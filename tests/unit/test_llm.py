"""Open-weight LLM stages (llm-recommend, llm-download, llm-serve) against a
fake `ezra` CLI and a fake OpenAI-compatible engine, so nothing is downloaded
and no real model runs."""

import json
import os
import stat
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic_chimes import cli
from agentic_chimes.stages import _llm, llm_agent, llm_download, llm_recommend, llm_serve

GALLERY = [
    {"name": "qwen2.5-3b-instruct", "format": "gguf", "size_b": 3.0, "tags": ["instruct", "local"], "gated": False},
    {"name": "qwen2.5-14b-instruct", "format": "gguf", "size_b": 14.0, "tags": ["instruct"], "gated": False},
    {"name": "deepseek-r1-distill-qwen-14b", "format": "gguf", "size_b": 14.0, "tags": ["reasoning", "instruct"], "gated": False},
    {"name": "llama3.1-70b-instruct", "format": "gguf", "size_b": 70.0, "tags": ["instruct"], "gated": True},
    {"name": "qwen2.5-32b-instruct-hf", "format": "safetensors", "size_b": 32.0, "tags": ["instruct"], "gated": False},
    {"name": "matscibert", "format": "safetensors", "size_b": 0.1, "tags": ["encoder", "science"], "gated": False},
]


def _exe(path: Path, body: str) -> Path:
    path.write_text(f"#!{sys.executable}\n" + textwrap.dedent(body))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


@pytest.fixture
def fake_ezra(tmp_path, monkeypatch):
    models = tmp_path / "models"
    weights = models / "qwen2.5-3b-instruct"
    weights.mkdir(parents=True)
    (weights / "qwen2.5-3b-instruct-q4_k_m.gguf").write_bytes(b"\0" * 1024)
    calls = tmp_path / "ezra_calls.jsonl"
    exe = _exe(tmp_path / "ezra", f"""
        import json, os, sys
        a = sys.argv[1:]
        with open({str(calls)!r}, "a") as f:
            f.write(json.dumps({{"argv": a, "models_dir": os.environ.get("EZRA_MODELS_DIR")}}) + "\\n")
        if a[:2] == ["models", "list"]:
            print({json.dumps(json.dumps(GALLERY))})
        elif a[:2] == ["models", "recommend"]:
            print(json.dumps({{"system": {{"ram_total_gb": 32.0, "cpu_count": 8, "gpus": []}}, "recommendations": []}}))
        elif a[:2] == ["models", "download"]:
            print(json.dumps([{{"name": n, "status": "skipped" if "--dry-run" in a else "downloaded", "message": ""}}
                              for n in a[2:] if not n.startswith("--") and n not in ("Q4_K_M",)]))
        elif a[:2] == ["agents", "show"]:
            ok = os.path.isfile(os.path.join(os.environ.get("EZRA_ROOT", ""), "agents", a[2], "agent.yaml"))
            print("agent " + a[2] if ok else "")
            sys.exit(0 if ok else 1)
        elif a[:2] == ["models", "info"]:
            ok = a[2] == "qwen2.5-3b-instruct"
            print(json.dumps({{"name": a[2], "installed": ok, "format": "gguf",
                               "path": {str(weights)!r} if ok else None}}))
        else:
            sys.exit(2)
    """)
    monkeypatch.setenv("CHIMES_EZRA", str(exe))
    return SimpleNamespace(exe=exe, calls=calls, weights=weights)


def _calls(fake):
    return [json.loads(line) for line in fake.calls.read_text().splitlines()]


# --- sizing -----------------------------------------------------------------

def test_ranking_keeps_chat_models_that_fit_largest_first():
    ranked = _llm.rank_for_agents(GALLERY, budget=20.0, has_gpu=False, ctx_size=32768)
    names = [r["name"] for r in ranked]
    assert names[0] == "qwen2.5-14b-instruct"              # 8.4 GB weights + 3.5 GB context
    assert "matscibert" not in names                        # encoder
    assert "llama3.1-70b-instruct" not in names             # gated and too big
    assert "qwen2.5-32b-instruct-hf" not in names           # safetensors without a GPU
    assert names.index("deepseek-r1-distill-qwen-14b") > names.index("qwen2.5-14b-instruct")


def test_context_counts_against_memory():
    big_ctx = _llm.rank_for_agents(GALLERY, budget=10.0, has_gpu=False, ctx_size=131072)
    assert "qwen2.5-14b-instruct" not in [r["name"] for r in big_ctx]


def test_moe_context_uses_active_parameters():
    moe = {"size_b": 109.0, "params": "109B-A17B (MoE, 16 experts)"}
    assert _llm.active_b(moe) == 17.0
    assert _llm.kv_cache_gb(moe, 32768) == 4.2


def test_general_instruct_wins_a_size_tie():
    models = [{"name": "chemllm-7b-chat", "format": "gguf", "size_b": 7.0, "tags": ["chat", "science"]},
              {"name": "qwen2.5-7b-instruct", "format": "gguf", "size_b": 7.0, "tags": ["instruct"]}]
    ranked = _llm.rank_for_agents(models, budget=20.0, has_gpu=False, ctx_size=32768)
    assert [r["name"] for r in ranked] == ["qwen2.5-7b-instruct", "chemllm-7b-chat"]


def test_login_node_budget_is_capped():
    assert _llm.budget_gb(512, 0, 0, "login-node") == _llm.LOGIN_NODE_BUDGET_GB
    assert _llm.budget_gb(256, 0, 0, "compute-node") == pytest.approx(204.8)
    assert _llm.budget_gb(256, 4, 80, "allocation") == 320


# --- llm-recommend ----------------------------------------------------------

def test_recommend_for_a_profile_compute_node(fake_ezra):
    parser = cli.build_parser()
    args = parser.parse_args(["llm-recommend", "--for-machine", "dane", "--include-gated"])
    res = llm_recommend.run(args)
    assert res["hardware"]["context"] == "compute-node" and res["hardware"]["memory_gb"] == 256
    assert res["top_pick"] == "llama3.1-70b-instruct"       # 42 GB + context fits 204.8 GB
    assert res["next"] == "chimes-agent llm-download --model llama3.1-70b-instruct"


def test_recommend_probes_this_host_through_ezra(fake_ezra, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(_llm, "host_context", lambda: "workstation")
    res = llm_recommend.run(cli.build_parser().parse_args(["llm-recommend"]))
    assert res["hardware"]["memory_gb"] == 32.0 and res["hardware"]["budget_gb"] == 25.6
    assert res["top_pick"] == "qwen2.5-14b-instruct"
    assert any(c["argv"][:2] == ["models", "recommend"] for c in _calls(fake_ezra))


def test_missing_ezra_says_how_to_install(monkeypatch):
    monkeypatch.delenv("CHIMES_EZRA", raising=False)
    monkeypatch.setattr(_llm.shutil, "which", lambda name: None)
    with pytest.raises(FileNotFoundError, match="CHIMES_EZRA"):
        _llm.ezra_bin()


# --- llm-download -----------------------------------------------------------

def test_download_dry_run_goes_to_profile_scratch(fake_ezra, tmp_path):
    profile = tmp_path / "cluster.yaml"
    profile.write_text(textwrap.dedent(f"""
        name: cluster
        job_system: slurm
        launcher: srun
        account: acct
        hosttype: none
        partitions: {{debug: debug, batch: batch}}
        default_ntasks_per_node: 16
        filesystem: {{scratch_root: {tmp_path / 'scratch'}}}
    """))
    args = cli.build_parser().parse_args(["llm-download", "--model", "qwen2.5-3b-instruct",
                                          "--for-machine", str(profile), "--dry-run"])
    res = llm_download.run(args)
    assert res["dry_run"] is True
    assert res["models_dir"] == str(tmp_path / "scratch" / "ezra" / "models")
    dl = [c for c in _calls(fake_ezra) if c["argv"][:2] == ["models", "download"]][0]
    assert "--dry-run" in dl["argv"] and dl["models_dir"] == res["models_dir"]


# --- llm-serve --------------------------------------------------------------

def test_serve_batch_dry_run_renders_job_and_keeps_key_off_the_command_line(fake_ezra, tmp_path, monkeypatch):
    monkeypatch.setenv("CHIMES_ACCOUNT", "mybank")
    engine = _exe(tmp_path / "llama-server", "pass\n")
    work = tmp_path / "serve"
    args = cli.build_parser().parse_args([
        "llm-serve", "--work-dir", str(work), "--model", "qwen2.5-3b-instruct", "--machine", "dane",
        "--engine-bin", str(engine), "--gpus", "1", "--dry-run"])
    res = llm_serve.run(args)
    assert res["where"] == "batch" and res["engine"] == "llama-server" and res["dry_run"]
    job = Path(res["job_file"]).read_text()
    assert "#SBATCH --gres=gpu:1" in job and "--ntasks-per-node 112" in job and "-A mybank" in job
    assert f"bash {work / 'serve.sh'}" in job
    serve = (work / "serve.sh").read_text()
    assert "--jinja" in serve and "--api-key-file" in serve and "-ngl 999" in serve and "-t 112" in serve
    key = (work / "api_key").read_text().strip()
    assert key not in serve and key not in job
    assert oct((work / "api_key").stat().st_mode & 0o777) == "0o600"
    assert not (work / "endpoint.json").exists()             # a dry run records nothing


def test_allocation_mode_needs_an_allocation(fake_ezra, tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    args = cli.build_parser().parse_args(["llm-serve", "--work-dir", str(tmp_path / "s"),
                                          "--model", "qwen2.5-3b-instruct", "--where", "allocation"])
    with pytest.raises(ValueError, match="idev"):
        llm_serve.run(args)


def test_undownloaded_model_points_at_llm_download(fake_ezra, tmp_path):
    args = cli.build_parser().parse_args(["llm-serve", "--work-dir", str(tmp_path / "s"), "--model", "other"])
    with pytest.raises(FileNotFoundError, match="llm-download"):
        llm_serve.run(args)


@pytest.mark.skipif(sys.platform == "win32", reason="serve.sh needs bash")
def test_serve_on_login_node_status_and_stop(fake_ezra, tmp_path, monkeypatch):
    """Launch a fake engine through serve.sh, wait for --status to say ready,
    check the client environment, then stop it."""
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    engine = _exe(tmp_path / "llama-server", """
        import http.server, json, sys
        a = sys.argv[1:]
        port = int(a[a.index("--port") + 1])
        key = open(a[a.index("--api-key-file") + 1]).read().strip()
        alias = a[a.index("--alias") + 1]
        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                ok = self.headers.get("Authorization") == "Bearer " + key
                self.send_response(200 if ok else 401)
                self.end_headers()
                if ok:
                    self.wfile.write(json.dumps({"data": [{"id": alias}]}).encode())
            def log_message(self, *a):
                pass
        http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
    """)
    work = tmp_path / "serve"
    base = ["llm-serve", "--work-dir", str(work)]
    res = llm_serve.run(cli.build_parser().parse_args(base + ["--model", "qwen2.5-3b-instruct", "--engine-bin", str(engine)]))
    assert res["where"] == "login" and res["bind_host"] == "127.0.0.1" and res["threads"] == _llm.LOGIN_NODE_THREADS
    try:
        status = {}
        for _ in range(100):
            status = llm_serve.run(cli.build_parser().parse_args(base + ["--status"]))
            if status["state"] == "ready":
                break
            time.sleep(0.1)
        assert status["state"] == "ready", status
        assert status["served_models"] == ["qwen2.5-3b-instruct"]
        env = status["client"]["env"]
        assert env["ANTHROPIC_BASE_URL"] == status["base_url"] and env["OPENAI_BASE_URL"].endswith("/v1")
        with pytest.raises(RuntimeError, match="still running"):
            llm_serve.run(cli.build_parser().parse_args(base + ["--model", "qwen2.5-3b-instruct", "--engine-bin", str(engine)]))
    finally:
        stopped = llm_serve.run(cli.build_parser().parse_args(base + ["--stop"]))
    assert stopped["stopped"] is True
    for _ in range(50):
        if not llm_serve._pid_alive(res["pid"]):
            break
        time.sleep(0.1)
        try:
            os.waitpid(res["pid"], os.WNOHANG)
        except ChildProcessError:
            pass
    assert llm_serve.run(cli.build_parser().parse_args(base + ["--status"]))["state"] == "stopped"


def test_function_calling_tag_wins_a_size_tie():
    models = [{"name": "llama3.1-8b-instruct", "format": "gguf", "size_b": 8.0, "tags": ["instruct"]},
              {"name": "hermes-3-llama-3.1-8b", "format": "gguf", "size_b": 8.0, "tags": ["instruct", "function-calling"]}]
    ranked = _llm.rank_for_agents(models, budget=20.0, has_gpu=False, ctx_size=32768)
    assert ranked[0]["name"] == "hermes-3-llama-3.1-8b"


# --- llm-agent (EZRA personal agent) -------------------------------------------

def test_shipped_agent_follows_ezra_layout_and_names_real_stages():
    import importlib

    import yaml

    folder = _llm.SHIPPED_AGENTS / _llm.DEFAULT_AGENT
    assert {p.name for p in folder.iterdir()} >= {"agent.yaml", "prompt.md", "NOTES.md"}
    spec = yaml.safe_load((folder / "agent.yaml").read_text())
    assert spec["name"] == _llm.DEFAULT_AGENT and spec["prompt"] == "prompt.md"
    assert spec["models"]["preferred"] and spec["models"]["fallback"] and spec["tools"] == []
    real = {importlib.import_module(f"agentic_chimes.stages.{m}").NAME for m in cli.STAGE_MODULE_NAMES}
    import re
    named = set(re.findall(r"`([a-z]+(?:-[a-z]+)*)(?: [^`]*)?`", (folder / "prompt.md").read_text()))
    stage_like = {n for n in named if "-" in n and not n.startswith("chimes")}
    assert stage_like <= real, stage_like - real


def test_llm_agent_installs_profile_with_models_for_the_hardware(fake_ezra, tmp_path, monkeypatch):
    monkeypatch.setattr(_llm, "host_context", lambda: "workstation")
    root = tmp_path / "ezra-root"
    args = cli.build_parser().parse_args(["llm-agent", "--ezra-root", str(root)])
    res = llm_agent.run(args)
    agent_dir = root / "agents" / "chimes-assistant"
    assert (agent_dir / "prompt.md").is_file() and (agent_dir / "NOTES.md").is_file()
    import yaml
    spec = yaml.safe_load((agent_dir / "agent.yaml").read_text())
    assert spec["models"]["preferred"] == ["qwen2.5-14b-instruct", "deepseek-r1-distill-qwen-14b"]
    assert spec["models"]["fallback"] == ["qwen2.5-3b-instruct"]
    assert res["check"]["ok"] and res["check"]["unknown_aliases"] == []
    with pytest.raises(FileExistsError, match="--replace"):
        llm_agent.run(args)

    work = tmp_path / "serve"
    engine = _exe(tmp_path / "llama-server", "pass\n")
    served = llm_serve.run(cli.build_parser().parse_args([
        "llm-serve", "--work-dir", str(work), "--agent", "chimes-assistant", "--ezra-root", str(root),
        "--engine-bin", str(engine), "--dry-run"]))
    assert served["model"] == "qwen2.5-3b-instruct"            # first downloaded model in the agent's list
    assert served["agent_prompt"] == str(agent_dir / "prompt.md")
    assert any("does not apply the agent's prompt" in w for w in served["warnings"])


def test_ezra_engine_serves_the_agent_with_its_prompt(fake_ezra, tmp_path, monkeypatch):
    monkeypatch.setattr(_llm, "host_context", lambda: "workstation")
    monkeypatch.setattr(_llm.shutil, "which", lambda name: None)
    root = tmp_path / "ezra-root"
    llm_agent.run(cli.build_parser().parse_args(["llm-agent", "--ezra-root", str(root)]))
    work = tmp_path / "serve"
    res = llm_serve.run(cli.build_parser().parse_args([
        "llm-serve", "--work-dir", str(work), "--agent", "chimes-assistant", "--ezra-root", str(root), "--dry-run"]))
    assert res["engine"] == "ezra" and res["model"] == "agent:chimes-assistant"
    serve = (work / "serve.sh").read_text()
    assert "serve --agent chimes-assistant --model qwen2.5-3b-instruct" in serve
    assert f"export EZRA_ROOT={root}" in serve


def test_gguf_picker_prefers_quant_and_first_shard(tmp_path):
    for name, size in [("m-Q8_0.gguf", 8), ("m-Q4_K_M.gguf", 4), ("big-00001-of-00002.gguf", 1), ("big-00002-of-00002.gguf", 9)]:
        (tmp_path / name).write_bytes(b"\0" * size)
    assert _llm.find_gguf(tmp_path, "q4_k_m").name == "m-Q4_K_M.gguf"
    assert _llm.find_gguf(tmp_path).name == "big-00001-of-00002.gguf"
