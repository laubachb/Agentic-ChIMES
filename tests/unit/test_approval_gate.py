"""The PreToolUse hook must ask before every allocation-spending command,
including forms settings.json text rules cannot see."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("approval_gate", ROOT / ".claude" / "hooks" / "approval_gate.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def test_asks_for_submissions_in_every_form(tmp_path):
    job = tmp_path / "in.json"
    job.write_text(json.dumps({"machine": "dane", "queue": "debug"}))
    dry = tmp_path / "dry.json"
    dry.write_text(json.dumps({"machine": "dane", "dry_run": True}))
    ask = [
        "chimes-agent hyper-search --data-manifest m.json --machine dane",
        "python3 -m agentic_chimes.cli hyper-search --machine dane --output-dir x",
        f"chimes-agent hyper-search --json-in {job} --output-dir x",
        f"cd /p/x && python3 -m agentic_chimes.cli solve --json-in {job}",
        "chimes-agent qe-relabel --frames a.xyzf --machine dane",
        "chimes-agent al-run --config-py c.py",
        "chimes-agent auto-build --json-in whatever.json",
        "sbatch run.cmd",
        "chimes-agent submit --command 'echo hi'",
    ]
    for cmd in ask:
        assert gate.reason_to_ask(cmd), cmd


def test_lets_local_and_dry_runs_through(tmp_path):
    dry = tmp_path / "dry.json"
    dry.write_text(json.dumps({"machine": "dane", "dry_run": True}))
    ok = [
        "chimes-agent hyper-search --machine dane --dry-run --output-dir x",
        f"chimes-agent hyper-search --json-in {dry}",
        "chimes-agent evaluate --params p.txt --holdout-xyzf h.xyzf",
        "chimes-agent qe-relabel --collect run_dir",
        "chimes-agent al-run --status-of 1234",
        "chimes-agent benchmark --collect bench --machine dane",
        "chimes-agent solve --describe",
        "ls -la && git status",
    ]
    for cmd in ok:
        assert gate.reason_to_ask(cmd) is None, cmd
