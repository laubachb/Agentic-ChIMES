"""Serve an open-weight LLM behind an OpenAI-compatible API for the agents.

Three places to run it (`--where`):

- `login`: a small model on the login node, bound to 127.0.0.1, 4 threads.
  Fine for trying things; login nodes are shared, so keep it short.
- `allocation`: inside an interactive allocation you already hold (TACC
  `idev`, `salloc`, LLNL `sxterm`). Launched in place, or with
  `srun --overlap` onto the allocated node when the shell is not on it.
- `batch`: an sbatch job on `--machine` (same renderer, account and
  approval path as every other submitting stage; `--dry-run` first).

Every mode runs the same `<work-dir>/serve.sh`, which records the host, port
and pid it came up on in `endpoint.runtime.json` and execs the engine. The
server requires the random key in `<work-dir>/api_key` (mode 0600). Like
`al-run`, launching returns at once; `--status` reports readiness and the
environment an agent harness needs, `--stop` ends it.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import time
from pathlib import Path

from .. import machines
from ..hpc import slurm
from ..io import atomic
from . import _llm
from .llm_download import models_dir_for

NAME = "llm-serve"
SUMMARY = "Serve an open-weight LLM (llama-server / vllm / ezra) on a login node, in an idev/salloc allocation, or as an sbatch job."
USES_OUTPUT_DIR = False
SUPPORTS_DRY_RUN = True
WHERE = ("auto", "login", "allocation", "batch")
SCHEMA = {
    "type": "object",
    "properties": {
        "work_dir": {"type": "string", "description": "Directory for serve.sh, api_key, endpoint.json and server.log (shared filesystem for batch)."},
        "model": {"type": ["string", "null"], "description": "EZRA gallery name (downloaded with llm-download); also the served model id."},
        "model_path": {"type": ["string", "null"], "description": "A .gguf file or Hugging Face model directory, instead of a gallery name."},
        "agent": {"type": ["string", "null"], "description": "EZRA agent (from llm-agent): serve its first downloaded preferred/fallback model; the ezra engine also applies its prompt."},
        "ezra_root": {"type": ["string", "null"], "description": "EZRA_ROOT holding agents/ (default CHIMES_EZRA_ROOT or ~/.chimes-agent/ezra)."},
        "models_dir": {"type": ["string", "null"], "description": "EZRA_MODELS_DIR the weights were downloaded to."},
        "where": {"type": "string", "enum": list(WHERE), "default": "auto", "description": "auto: batch with --machine, allocation inside a Slurm job, else login."},
        "machine": {"type": ["string", "null"], "description": "Machine profile for --where batch."},
        "queue": {"type": "string", "default": "batch"},
        "walltime_hours": {"type": "number", "default": 4.0},
        "engine": {"type": "string", "enum": list(_llm.ENGINES), "default": "auto"},
        "engine_bin": {"type": ["string", "null"], "description": "Path to the engine executable when it is not on PATH."},
        "gpus": {"type": ["integer", "null"], "description": "GPUs to use (batch: requested with --gres)."},
        "threads": {"type": ["integer", "null"], "description": "CPU threads for llama-server (default: login 4, else the node's cores)."},
        "ctx_size": {"type": "integer", "default": _llm.DEFAULT_CTX},
        "port": {"type": "integer", "default": 0, "description": "0 picks a free port when the server starts."},
        "quant": {"type": ["string", "null"], "description": "Pick the GGUF file whose name contains this, e.g. Q4_K_M."},
        "tool_call_parser": {"type": "string", "default": "hermes", "description": "vllm tool-call parser for the model family (hermes suits Qwen2.5)."},
        "ezra": {"type": ["string", "null"]},
        "status": {"type": "boolean", "default": False, "description": "Report on the server in work_dir instead of launching."},
        "stop": {"type": "boolean", "default": False, "description": "Stop the server in work_dir instead of launching."},
    },
}


def add_arguments(parser) -> None:
    parser.add_argument("--work-dir", dest="work_dir", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-path", dest="model_path", default=None)
    parser.add_argument("--agent", default=None)
    parser.add_argument("--ezra-root", dest="ezra_root", default=None)
    parser.add_argument("--models-dir", dest="models_dir", default=None)
    parser.add_argument("--where", choices=WHERE, default="auto")
    parser.add_argument("--machine", default=None)
    parser.add_argument("--queue", default="batch")
    parser.add_argument("--walltime-hours", dest="walltime_hours", type=float, default=4.0)
    parser.add_argument("--engine", choices=_llm.ENGINES, default="auto")
    parser.add_argument("--engine-bin", dest="engine_bin", default=None)
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--ctx-size", dest="ctx_size", type=int, default=_llm.DEFAULT_CTX)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--quant", default=None)
    parser.add_argument("--tool-call-parser", dest="tool_call_parser", default="hermes")
    parser.add_argument("--ezra", default=None)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--stop", action="store_true")


def _pid_alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
    except (OSError, TypeError, ValueError):
        return False
    return True


def _alive(ep: dict) -> bool:
    if ep.get("job_id"):
        return slurm.job_in_queue(str(ep["job_id"]))
    return _pid_alive(ep.get("pid"))


def _where(args) -> str:
    where = getattr(args, "where", "auto") or "auto"
    if where == "auto":
        where = "batch" if getattr(args, "machine", None) else "allocation" if os.environ.get("SLURM_JOB_ID") else "login"
    if where == "batch" and not getattr(args, "machine", None):
        raise ValueError("--where batch needs --machine <profile>")
    if where == "allocation" and not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("--where allocation must run inside an interactive allocation: start one first "
                         "(TACC: `idev -p <queue> -N 1 -t 02:00:00`; Slurm: `salloc -N 1 -p <partition> -t 02:00:00`), "
                         "then run this from that shell")
    return where


def _weights(args):
    """(weights path, format, served alias)."""
    if getattr(args, "model_path", None):
        path = Path(args.model_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"--model-path {path} does not exist")
        fmt = _llm.weights_format(path)
        alias = args.model or (path.stem if path.is_file() else path.name)
    else:
        ns = type("A", (), {"models_dir": getattr(args, "models_dir", None), "for_machine": getattr(args, "machine", None)})
        models_dir = models_dir_for(ns) or None
        if getattr(args, "model", None):
            candidates = [args.model]
        elif getattr(args, "agent", None):
            spec = _llm.load_agent(args.agent, getattr(args, "ezra_root", None))
            models = spec.get("models") or {}
            candidates = list(dict.fromkeys(list(models.get("preferred") or []) + list(models.get("fallback") or [])))
            if not candidates:
                raise ValueError(f"agent {args.agent!r} lists no models; re-run `chimes-agent llm-agent --replace`")
        else:
            raise ValueError("llm-serve needs --model (EZRA gallery name), --agent, or --model-path")
        info = None
        for name in candidates:
            found = _llm.run_ezra(["models", "info", name, "--json"], ezra=getattr(args, "ezra", None), models_dir=models_dir)
            if found.get("installed"):
                info = found
                break
        if info is None:
            raise FileNotFoundError(f"none of {candidates} is downloaded; run `chimes-agent llm-download --model {candidates[0]}` first")
        path = Path(info["path"])
        fmt = _llm.weights_format(path)
        alias = info["name"]
    if fmt == "gguf":
        path = _llm.find_gguf(path, getattr(args, "quant", None))
    return path, fmt, alias


def _gpus(args, where: str) -> int:
    if getattr(args, "gpus", None) is not None:
        return args.gpus
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if where == "allocation" and visible and visible not in ("-1", "NoDevFiles"):
        return len([d for d in visible.split(",") if d.strip()])
    return 0


def _threads(args, where: str, profile) -> int:
    if getattr(args, "threads", None):
        return min(args.threads, _llm.LOGIN_NODE_THREADS * 2) if where == "login" else args.threads
    if where == "login":
        return _llm.LOGIN_NODE_THREADS
    if where == "batch":
        return int(profile.node.get("cores") or profile.default_ntasks_per_node)
    return int(os.environ.get("SLURM_CPUS_ON_NODE") or os.cpu_count() or 4)


def _status(work_dir: Path) -> dict:
    ep = _llm.read_json(work_dir / _llm.ENDPOINT_FILE)
    if not ep:
        return {"work_dir": str(work_dir), "state": "none", "note": "no server was launched from this directory"}
    rt = _llm.read_json(work_dir / _llm.RUNTIME_FILE) or {}
    alive = _alive(ep)
    out = {"work_dir": str(work_dir), "where": ep.get("where"), "engine": ep.get("engine"), "model": ep.get("model"),
           "pid": ep.get("pid"), "job_id": ep.get("job_id"), "log": ep.get("log")}
    if not alive:
        out["state"] = "stopped"
    elif not rt.get("port"):
        out["state"] = "queued" if ep.get("job_id") else "starting"
    else:
        host = "127.0.0.1" if ep.get("where") == "login" else rt.get("host")
        base_url = f"http://{host}:{rt['port']}"
        key_file = Path(ep["key_file"])
        key = key_file.read_text().strip() if key_file.is_file() else None
        check = _llm.probe(base_url, key)
        out.update({"state": "ready" if check["ready"] else "loading", "host": rt.get("host"), "port": rt["port"],
                    "base_url": base_url, "served_models": check.get("models"),
                    "client": _llm.client_settings(ep.get("engine"), base_url, ep.get("model"), key_file)})
        if ep.get("where") != "login":
            out["ssh_tunnel"] = (f"ssh -N -L {rt['port']}:{rt.get('host')}:{rt['port']} <you>@<login-node>  "
                                 f"# then use http://127.0.0.1:{rt['port']} from your laptop")
    if out["state"] != "ready" and ep.get("log"):
        try:
            out["log_tail"] = "\n".join(Path(ep["log"]).read_text(errors="replace").splitlines()[-15:])
        except OSError:
            pass
    return out


def _stop(work_dir: Path) -> dict:
    ep = _llm.read_json(work_dir / _llm.ENDPOINT_FILE)
    if not ep:
        return {"work_dir": str(work_dir), "stopped": False, "note": "no server was launched from this directory"}
    if not _alive(ep):
        return {"work_dir": str(work_dir), "stopped": False, "note": "server was not running"}
    if ep.get("job_id"):
        slurm.cancel_job(str(ep["job_id"]))
    else:
        os.killpg(int(ep["pid"]), signal.SIGTERM)  # its own session: takes srun and the engine with it
    (work_dir / _llm.RUNTIME_FILE).unlink(missing_ok=True)
    return {"work_dir": str(work_dir), "stopped": True, "pid": ep.get("pid"), "job_id": ep.get("job_id")}


def run(args) -> dict:
    if not getattr(args, "work_dir", None):
        raise ValueError("llm-serve needs --work-dir (holds serve.sh, api_key, endpoint.json, server.log)")
    work_dir = Path(args.work_dir).expanduser().resolve()
    if getattr(args, "status", False):
        return _status(work_dir)
    if getattr(args, "stop", False):
        return _stop(work_dir)

    where = _where(args)
    dry = bool(getattr(args, "dry_run", False))
    prior = _llm.read_json(work_dir / _llm.ENDPOINT_FILE)
    if prior and _alive(prior):
        raise RuntimeError(f"a server from {work_dir} is still running (pid {prior.get('pid')}, job {prior.get('job_id')}); "
                           f"`chimes-agent llm-serve --stop --work-dir {work_dir}` first")
    profile = machines.load_profile(args.machine) if where == "batch" else None
    if where != "login":
        slurm.check_shared_dir(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    weights, fmt, alias = _weights(args)
    gpus = _gpus(args, where)
    bins = _llm.available_engines()
    engine_bin = getattr(args, "engine_bin", None)
    if engine_bin:
        target = args.engine if args.engine != "auto" else ("vllm" if fmt == "safetensors" else "llama-server")
        bins[target] = engine_bin
    engine = _llm.choose_engine(args.engine, fmt, gpus, bins)
    if engine == "ezra" and getattr(args, "model_path", None):
        raise ValueError("the ezra engine serves gallery names only; use --model, or llama-server for --model-path")
    threads = _threads(args, where, profile)
    key_file = _llm.write_key(work_dir)
    bind = "127.0.0.1" if where == "login" else "0.0.0.0"
    ns = type("A", (), {"models_dir": getattr(args, "models_dir", None), "for_machine": getattr(args, "machine", None)})
    models_dir = models_dir_for(ns) or None
    agent = getattr(args, "agent", None)
    agent_root = str(_llm.ezra_root(getattr(args, "ezra_root", None))) if agent else None
    command = _llm.engine_command(engine, binary=bins[engine], weights=weights, alias=alias, host=bind,
                                  ctx_size=args.ctx_size, threads=threads, gpus=gpus, key_file=key_file,
                                  tool_call_parser=args.tool_call_parser,
                                  agent=agent if engine == "ezra" else None)
    script = work_dir / _llm.SERVE_SCRIPT
    script.write_text(_llm.render_serve_script(work_dir, command, port=args.port, models_dir=models_dir,
                                               ezra_root_dir=agent_root if engine == "ezra" else None))
    script.chmod(0o755)
    log = work_dir / _llm.SERVER_LOG

    warnings = []
    if engine == "ezra":
        warnings.append("ezra serve answers plain chat only (no tool calls) and has no API key: an agent cannot "
                        "drive chimes-agent through it. Install llama.cpp's llama-server for GGUF models.")
    if where == "login":
        size_gb = weights.stat().st_size / 1e9 if weights.is_file() else None
        if size_gb and size_gb > _llm.LOGIN_NODE_BUDGET_GB:
            warnings.append(f"{size_gb:.1f} GB of weights on a shared login node; prefer --where allocation or batch.")
    served = f"agent:{agent}" if agent and engine == "ezra" else alias
    prompt = None
    if agent:
        spec = _llm.load_agent(agent, getattr(args, "ezra_root", None))
        prompt = str(Path(spec["_dir"]) / (spec.get("prompt") or "prompt.md"))
        if engine != "ezra":
            warnings.append(f"{engine} does not apply the agent's prompt; give {prompt} to your harness "
                            "(Cline: a .clinerules file; Claude Code reads CLAUDE.md already).")

    record = {"where": where, "engine": engine, "model": served, "agent": agent, "agent_prompt": prompt,
              "weights": str(weights), "format": fmt,
              "work_dir": str(work_dir), "script": str(script), "key_file": str(key_file), "log": str(log),
              "bind_host": bind, "threads": threads, "gpus": gpus, "ctx_size": args.ctx_size,
              "pid": None, "job_id": None, "machine": getattr(args, "machine", None), "launched_at": time.time()}

    if not dry:
        (work_dir / _llm.RUNTIME_FILE).unlink(missing_ok=True)
    if where == "batch":
        extra = [f"--gres=gpu:{gpus}"] if gpus else []
        job_name = re.sub(r"[^A-Za-z0-9_.-]", "-", f"llm-{alias}")[:40]
        handle = slurm.submit_job(profile, job_name=job_name, commands=[f"bash {script} > {log} 2>&1"],
                                  work_dir=work_dir, nodes=1, walltime_hours=args.walltime_hours,
                                  queue=args.queue, dry_run=dry, extra_directives=extra)
        record.update({"job_id": handle.job_id, "job_file": str(handle.job_file), "queue": handle.queue})
    else:
        launch = ["bash", str(script)]
        if where == "allocation" and not os.environ.get("SLURMD_NODENAME"):
            launch = ["srun", "--jobid", os.environ["SLURM_JOB_ID"], "--overlap", "-N", "1", "-n", "1",
                      f"--cpus-per-task={threads}"] + launch
        record["command"] = " ".join(launch)
        if not dry:
            with open(log, "w") as logf:
                proc = subprocess.Popen(launch, cwd=str(work_dir), stdout=logf, stderr=subprocess.STDOUT,
                                        start_new_session=True)
            record["pid"] = proc.pid

    out = dict(record, dry_run=dry, warnings=warnings, serve_script=script.read_text())
    if dry:
        out["note"] = "dry run: serve.sh (and for batch, the sbatch script) written; nothing started"
        return out
    atomic.write_json(work_dir / _llm.ENDPOINT_FILE, record, indent=1)
    out["next"] = (f"chimes-agent llm-serve --status --work-dir {work_dir}  "
                   "# 'ready' gives base_url and the environment for your agent harness")
    return out
