"""Open-weight LLMs for the agent layer, through EZRA.

EZRA (https://github.com/Awwal41/EZRA) keeps a curated gallery of open
models, probes the host's RAM/GPUs, and downloads weights from Hugging Face.
It is GPL-3.0 and needs Python >= 3.10, so it is called as an external CLI
(`ezra ... --json`), never imported: install it in its own environment and
point `CHIMES_EZRA` at its `ezra` executable if it is not on PATH.

Serving uses an engine that speaks OpenAI tool calling, because an agent
driving `chimes-agent` needs tools: llama.cpp's `llama-server` for GGUF
weights (CPU or GPU) and `vllm` for safetensors on GPUs. `ezra serve` is the
fallback; it answers plain chat only.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shlex
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Optional

LOGIN_NODE_BUDGET_GB = 8.0
LOGIN_NODE_THREADS = 4
DEFAULT_CTX = 32768

ENDPOINT_FILE = "endpoint.json"
RUNTIME_FILE = "endpoint.runtime.json"
KEY_FILE = "api_key"
SERVE_SCRIPT = "serve.sh"
SERVER_LOG = "server.log"

ENGINES = ("auto", "llama-server", "vllm", "ezra")

SHIPPED_AGENTS = Path(__file__).resolve().parents[1] / "ezra_agents"
DEFAULT_AGENT = "chimes-assistant"


def ezra_root(explicit: Optional[str] = None) -> Path:
    """Where `llm-agent` installs EZRA agent profiles (`<root>/agents/<name>/`);
    passed to EZRA as EZRA_ROOT so your EZRA checkout stays untouched."""
    return Path(explicit or os.environ.get("CHIMES_EZRA_ROOT") or "~/.chimes-agent/ezra").expanduser().resolve()


def load_agent(name: str, root: Optional[str] = None) -> dict:
    import yaml

    path = ezra_root(root) / "agents" / name / "agent.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"no EZRA agent {name!r} at {path.parent}; create it with `chimes-agent llm-agent --name {name}`")
    data = yaml.safe_load(path.read_text()) or {}
    data["_dir"] = str(path.parent)
    return data


# --- EZRA CLI -----------------------------------------------------------------

def ezra_bin(explicit: Optional[str] = None) -> str:
    found = explicit or os.environ.get("CHIMES_EZRA") or shutil.which("ezra")
    if not found:
        raise FileNotFoundError(
            "EZRA is not installed (no `ezra` on PATH and CHIMES_EZRA unset). Install it in its own "
            "Python >= 3.10 environment: `python3 -m venv ~/ezra-venv && ~/ezra-venv/bin/pip install "
            "'ezra @ git+https://github.com/Awwal41/EZRA'`, then `export CHIMES_EZRA=~/ezra-venv/bin/ezra`."
        )
    return found


def _parse_json(text: str):
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        raise ValueError("no JSON in output")
    return json.loads(text[min(starts):])


def run_ezra(args: list, *, ezra: Optional[str] = None, models_dir: Optional[str] = None,
             ezra_root: Optional[str] = None, check: bool = True, parse: bool = True,
             timeout: Optional[float] = None):
    """Run `ezra <args>` and return its parsed JSON stdout (or the raw text
    with parse=False). EZRA's own stderr is echoed to stdout, which the CLI
    routes into the stage log. `ezra_root` sets EZRA_ROOT, where EZRA looks
    for `agents/`."""
    cmd = [ezra_bin(ezra)] + [str(a) for a in args]
    env = dict(os.environ)
    if models_dir:
        env["EZRA_MODELS_DIR"] = str(models_dir)
    if ezra_root:
        env["EZRA_ROOT"] = str(ezra_root)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    if proc.stderr:
        print(proc.stderr)
    if check and proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-5:]
        raise RuntimeError(f"`{' '.join(cmd)}` failed (exit {proc.returncode}): {' | '.join(tail)}")
    if not parse:
        return proc.stdout
    try:
        return _parse_json(proc.stdout)
    except ValueError as exc:
        raise RuntimeError(f"`{' '.join(cmd)}` did not print JSON ({exc}); is this EZRA build too old?") from exc


# --- where are we, and how much memory can a model use ------------------------

def host_context() -> str:
    """'allocation' inside a Slurm job (sbatch, salloc, TACC idev), 'login-node'
    on a cluster outside one, else 'workstation'."""
    if os.environ.get("SLURM_JOB_ID"):
        return "allocation"
    if shutil.which("sbatch") or shutil.which("squeue"):
        return "login-node"
    return "workstation"


def budget_gb(memory_gb: float, gpus: int, gpu_memory_gb: float, context: str) -> float:
    """Memory a model may use: total VRAM on a GPU node, else ~80% of RAM.
    Login nodes are shared, so they are capped regardless of what they have."""
    if gpus and gpu_memory_gb:
        budget = gpus * gpu_memory_gb
    else:
        budget = 0.8 * memory_gb
    if context == "login-node":
        budget = min(budget, LOGIN_NODE_BUDGET_GB)
    return round(budget, 1)


def weights_gb(model: dict) -> float:
    """RAM/VRAM the weights need. Uses the gallery's own figure when EZRA
    reports one, else EZRA's size rule (GGUF ~0.6 GB per 1B params,
    16-bit safetensors ~2 GB per 1B)."""
    if model.get("recommended_ram_gb"):
        return float(model["recommended_ram_gb"])
    size = float(model.get("size_b") or 1.0)
    fmt = (model.get("format") or "").lower()
    return max(1.0, round(size * (0.6 if fmt == "gguf" else 0.7 if fmt == "mlx" else 2.0), 1))


def active_b(model: dict) -> float:
    """Active parameters in billions: the "A17B" of a mixture-of-experts
    "109B-A17B", else the total size."""
    m = re.search(r"A(\d+(?:\.\d+)?)B", str(model.get("params") or ""))
    return float(m.group(1)) if m else float(model.get("size_b") or 1.0)


def kv_cache_gb(model: dict, ctx_size: int) -> float:
    """Rough KV-cache need for one sequence of ctx_size tokens (grouped-query
    attention models of 3-14B use ~0.25 GB per 1B params per 32k tokens;
    for mixture-of-experts models the active size is the better proxy)."""
    return round(max(0.5, 0.25 * active_b(model) * ctx_size / 32768), 1)


def is_chat_model(model: dict) -> bool:
    tags = {t.lower() for t in list(model.get("tags") or []) + list(model.get("science_tags") or [])}
    name = (model.get("name") or "").lower()
    if "encoder" in tags or "embedding" in tags:
        return False
    return bool(tags & {"instruct", "chat"}) or any(k in name for k in ("instruct", "-it", "chat"))


def rank_for_agents(models: list, *, budget: float, has_gpu: bool, ctx_size: int,
                    include_gated: bool = False, keyword: Optional[str] = None) -> list:
    """Chat models that fit `budget` with a `ctx_size` context, largest first
    (bigger models follow multi-step tool-calling instructions more reliably).
    GGUF is preferred without a GPU; reasoning distills rank lower because
    their long thinking output crowds out tool calls."""
    ranked = []
    for m in models:
        if not is_chat_model(m) or (m.get("gated") and not include_gated):
            continue
        if keyword and keyword.lower() not in json.dumps(m).lower():
            continue
        need = round(weights_gb(m) + kv_cache_gb(m, ctx_size), 1)
        if budget > 0 and need > budget:
            continue
        fmt = (m.get("format") or "").lower()
        if fmt not in ("gguf", "safetensors"):
            continue
        if fmt == "safetensors" and not has_gpu:
            continue
        reasons = [f"needs ~{need} GB ({weights_gb(m)} weights + {kv_cache_gb(m, ctx_size)} context) of {budget} GB"]
        score = float(m.get("size_b") or 0)
        tags = {t.lower() for t in m.get("tags") or []}
        if fmt == "gguf" and not has_gpu:
            reasons.append("GGUF runs on CPU with llama-server")
        if "instruct" in tags:
            score += 0.5  # general instruction tuning beats a domain chat model of the same size at tool calls
        if tags & {"function-calling", "tools", "tool-use"}:
            score += 1.0
            reasons.append("tagged for function calling")
        if "reasoning" in tags:
            score *= 0.5
            reasons.append("reasoning distill: weaker at tool calls")
        ranked.append({
            "name": m.get("name"), "repo_id": m.get("repo_id"), "format": fmt, "params": m.get("params"),
            "size_b": m.get("size_b"), "license": m.get("license"), "gated": bool(m.get("gated")),
            "installed": bool(m.get("installed")), "path": m.get("path"), "need_gb": need,
            "engine": "llama-server" if fmt == "gguf" else "vllm", "reasons": reasons, "_score": score,
        })
    ranked.sort(key=lambda r: (-r["_score"], r["need_gb"], r["name"] or ""))
    for r in ranked:
        r.pop("_score")
    return ranked


def available_engines() -> dict:
    return {
        "llama-server": shutil.which("llama-server"),
        "vllm": shutil.which("vllm"),
        "ezra": os.environ.get("CHIMES_EZRA") or shutil.which("ezra"),
    }


# --- weights and engine -------------------------------------------------------

def find_gguf(path: Path, quant: Optional[str] = None) -> Path:
    if path.is_file():
        return path
    files = sorted(p for p in path.rglob("*.gguf") if p.is_file())
    if quant:
        files = [p for p in files if quant.lower() in p.name.lower()] or files
    if not files:
        raise FileNotFoundError(f"no .gguf file under {path}")
    # Split GGUFs ("-00001-of-00003") load from their first shard.
    first = [p for p in files if re.search(r"-0+1-of-\d+\.gguf$", p.name)]
    return first[0] if first else max(files, key=lambda p: p.stat().st_size)


def weights_format(path: Path) -> str:
    if path.is_file() and path.suffix == ".gguf":
        return "gguf"
    if path.is_dir() and any(path.rglob("*.gguf")):
        return "gguf"
    if path.is_dir() and (path / "config.json").is_file():
        return "safetensors"
    raise ValueError(f"{path} is neither a .gguf file nor a directory with GGUF or Hugging Face (config.json) weights")


def choose_engine(engine: str, fmt: str, gpus: int, bins: dict) -> str:
    if engine != "auto":
        if not bins.get(engine):
            raise FileNotFoundError(f"engine {engine!r} not found on PATH; pass --engine-bin or install it")
        return engine
    if fmt == "gguf":
        if bins.get("llama-server"):
            return "llama-server"
        if bins.get("ezra"):
            return "ezra"
        raise FileNotFoundError("GGUF weights need llama.cpp's `llama-server` (or `ezra serve`) on PATH; pass --engine-bin")
    if not gpus:
        raise ValueError("safetensors weights need a GPU (vllm); use a GGUF model on CPU nodes, or pass --gpus")
    if bins.get("vllm"):
        return "vllm"
    raise FileNotFoundError("safetensors weights need `vllm` on PATH; pass --engine-bin")


def engine_command(engine: str, *, binary: str, weights: Path, alias: str, host: str, ctx_size: int,
                   threads: int, gpus: int, key_file: Path, tool_call_parser: str,
                   agent: Optional[str] = None) -> list:
    """The server command line, with `$PORT` left for serve.sh to fill in.
    API keys are read from key_file, never put on the command line where
    `ps` shows them to other users of the node."""
    q = shlex.quote
    if engine == "llama-server":
        cmd = [q(binary), "-m", q(str(weights)), "--alias", q(alias), "--host", host, "--port", '"$PORT"',
               "-c", str(ctx_size), "-t", str(threads), "--jinja", "--api-key-file", q(str(key_file))]
        if gpus:
            cmd += ["-ngl", "999"]
        return cmd
    if engine == "vllm":
        return [f'VLLM_API_KEY="$(cat {q(str(key_file))})"', q(binary), "serve", q(str(weights)),
                "--served-model-name", q(alias), "--host", host, "--port", '"$PORT"',
                "--max-model-len", str(ctx_size), "--tensor-parallel-size", str(max(1, gpus)),
                "--enable-auto-tool-choice", "--tool-call-parser", q(tool_call_parser)]
    if engine == "ezra":
        target = ["--agent", q(agent), "--model", q(alias)] if agent else ["--model", q(alias)]
        return [q(binary), "serve", *target, "--host", host, "--port", '"$PORT"']
    raise ValueError(f"unknown engine {engine!r}; choose from {ENGINES[1:]}")


def write_key(work_dir: Path) -> Path:
    path = work_dir / KEY_FILE
    if not path.exists():
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(secrets.token_urlsafe(32) + "\n")
    return path


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def render_serve_script(work_dir: Path, command: list, *, port: int, models_dir: Optional[str],
                        ezra_root_dir: Optional[str] = None) -> str:
    """serve.sh: records where the server came up (host, port, pid, job) in
    endpoint.runtime.json, then execs the engine in its place. The same
    script runs on a login node, inside an idev/salloc allocation, or as
    the body of an sbatch job."""
    q = shlex.quote
    py = q(sys.executable)
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        f"cd {q(str(work_dir))}",
    ]
    if models_dir:
        lines.append(f"export EZRA_MODELS_DIR={q(str(models_dir))}")
    if ezra_root_dir:
        lines.append(f"export EZRA_ROOT={q(str(ezra_root_dir))}")
    lines += [
        f"PORT={port}",
        'if [ "$PORT" = "0" ]; then PORT=$(' + py + " -c 'import socket; s=socket.socket(); s.bind((\"\", 0)); print(s.getsockname()[1])'); fi",
        'HOST=$(hostname)',
        "printf '{\"host\": \"%s\", \"port\": %s, \"pid\": %s, \"job_id\": \"%s\"}\\n' "
        f'"$HOST" "$PORT" "$$" "${{SLURM_JOB_ID:-}}" > {RUNTIME_FILE}',
        "exec " + " ".join(command),
    ]
    return "\n".join(lines) + "\n"


# --- status -------------------------------------------------------------------

def read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def probe(base_url: str, key: Optional[str], timeout: float = 3.0) -> dict:
    """GET <base_url>/v1/models: is the server up, and what does it serve."""
    import urllib.error
    import urllib.request

    req = urllib.request.Request(base_url.rstrip("/") + "/v1/models")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode() or "{}")
        return {"ready": True, "models": [m.get("id") for m in data.get("data", [])]}
    except urllib.error.HTTPError as exc:
        return {"ready": False, "http_status": exc.code}
    except (OSError, ValueError) as exc:
        return {"ready": False, "reason": str(exc)}


def client_settings(engine: str, base_url: str, alias: str, key_file: Path) -> dict:
    """Settings for agent harnesses: any OpenAI-compatible client, Cline (the
    tool-using agent EZRA pairs with), and Claude Code itself when
    llama-server provides the Anthropic Messages API."""
    key = f"$(cat {shlex.quote(str(key_file))})"
    env = {"OPENAI_BASE_URL": f"{base_url}/v1", "OPENAI_API_KEY": key, "OPENAI_MODEL": alias}
    if engine == "llama-server":
        env.update({"ANTHROPIC_BASE_URL": base_url, "ANTHROPIC_AUTH_TOKEN": key,
                    "ANTHROPIC_MODEL": alias, "ANTHROPIC_SMALL_FAST_MODEL": alias})
    cline = (f'cline auth --provider openai-compatible --apikey "{key}" '
             f"--modelid {shlex.quote(alias)} --baseurl {base_url}/v1")
    return {"env": env, "exports": [f'export {k}="{v}"' for k, v in env.items()], "cline_auth": cline}
