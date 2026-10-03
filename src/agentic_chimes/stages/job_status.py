"""One verdict for a submitted job: is it done, did it work, and if not, why.

Reads the `job.json` every submission writes into its work directory (job
id, expected result files) and combines it with the scheduler's view.

- **State**: `squeue` while queued or running; otherwise `sacct`, with
  state, exit code, elapsed time and limit.
- **Expected results**: each must exist and, for JSON, parse.
- **Logs**: the tail of `stdoutmsg` and any `*.out` in the directory.

Verdicts:

| verdict | meaning |
|---|---|
| `QUEUED` / `RUNNING` | still in the scheduler |
| `SUCCEEDED` | finished, and every expected result exists and parses |
| `COMPLETED_WITHOUT_RESULTS` | Slurm says COMPLETED but results are missing: a non-shared job directory (/tmp), or an error the job script swallowed |
| `FAILED` | TIMEOUT, OUT_OF_MEMORY, NODE_FAIL, CANCELLED or a nonzero exit; `reason` and `fix` say what to do |
| `UNKNOWN` | the scheduler no longer knows the job (accounting purged) and results are missing |

Read-only: it never resubmits or cancels anything.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ..io import atomic

NAME = "job-status"
SUMMARY = "Verdict on a submitted job: queued/running/succeeded/failed (why, and the fix)/completed-without-results."
USES_OUTPUT_DIR = False
SCHEMA = {
    "type": "object",
    "properties": {
        "work_dir": {"type": ["string", "null"], "description": "Directory of a submission (holds job.json)."},
        "job_id": {"type": ["string", "null"], "description": "Slurm job id, when there is no job.json."},
        "expect": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Extra result files to require."},
    },
}

_FIX = {
    "TIMEOUT": "raise --walltime-hours; hyper-search and qe-relabel resume from their caches, so re-submitting repeats no finished work",
    "OUT_OF_MEMORY": "use fewer parallel workers per node (or a full node), or smaller frames/max_clusters",
    "NODE_FAIL": "re-submit unchanged: a hardware failure, not the job",
    "PREEMPTED": "re-submit unchanged",
    "CANCELLED": "cancelled by a user or the scheduler; re-submit if unintended",
}


def add_arguments(parser) -> None:
    parser.add_argument("--work-dir", dest="work_dir", default=None)
    parser.add_argument("--job-id", dest="job_id", default=None)
    parser.add_argument("--expect", action="append", default=None)


def _squeue(job_id):
    p = subprocess.run(["squeue", "-h", "-j", str(job_id), "-o", "%T|%M|%L|%R"], capture_output=True, text=True)
    line = p.stdout.strip().splitlines()
    return line[0].split("|") if line else None


def _sacct(job_id):
    p = subprocess.run(["sacct", "-j", str(job_id), "-X", "-n", "-P", "--format=State,ExitCode,Elapsed,Timelimit"],
                       capture_output=True, text=True)
    line = [ln for ln in p.stdout.strip().splitlines() if ln]
    return line[0].split("|") if line else None


def _tail(path: Path, n=15):
    try:
        return path.read_text(errors="replace").splitlines()[-n:]
    except OSError:
        return []


def _result_ok(path: Path):
    if not path.is_file() or path.stat().st_size == 0:
        return False, "missing" if not path.exists() else "empty"
    if path.suffix == ".json" and atomic.read_json(path) is None:
        return False, "unreadable JSON (truncated?)"
    return True, "ok"


def run(args) -> dict:
    work = Path(args.work_dir).resolve() if getattr(args, "work_dir", None) else None
    rec = atomic.read_json(work / "job.json") if work else None
    job_id = getattr(args, "job_id", None) or (rec or {}).get("job_id")
    if not job_id:
        raise ValueError("job-status needs --job-id, or --work-dir holding a job.json (written by every submission)")
    expect = [Path(e) for e in ((rec or {}).get("expect") or []) + list(getattr(args, "expect", None) or [])]

    out = {"job_id": str(job_id), "work_dir": str(work) if work else None, "job_name": (rec or {}).get("job_name")}
    q = _squeue(job_id)
    if q:
        state, elapsed, left, reason = q
        out.update({"verdict": "QUEUED" if state == "PENDING" else "RUNNING", "slurm_state": state,
                    "elapsed": elapsed, "time_left": left, "reason": reason if state == "PENDING" else None})
        return out

    acct = _sacct(job_id)
    results = {str(p): _result_ok(p)[1] for p in expect}
    all_ok = all(v == "ok" for v in results.values())
    logs = {}
    if work:
        for p in [work / "stdoutmsg"] + sorted(work.glob("*.out")):
            if p.is_file():
                logs[p.name] = _tail(p)
    out.update({"expected_results": results, "log_tail": logs})
    if not acct:
        out["verdict"] = "SUCCEEDED" if expect and all_ok else "UNKNOWN"
        if out["verdict"] == "UNKNOWN":
            out["reason"] = "sacct has no record (purged accounting?) and the expected results are missing"
        return out

    state, exit_code, elapsed, limit = acct
    base = state.split()[0]
    out.update({"slurm_state": state, "exit_code": exit_code, "elapsed": elapsed, "time_limit": limit})
    if base == "COMPLETED" and all_ok:
        out["verdict"] = "SUCCEEDED"
    elif base == "COMPLETED":
        out["verdict"] = "COMPLETED_WITHOUT_RESULTS"
        out["reason"] = ("Slurm reports success but expected results are " + ", ".join(f"{Path(k).name}: {v}" for k, v in results.items() if v != "ok"))
        out["fix"] = ("check log_tail for an error the job script swallowed; if the directory is under /tmp the compute "
                      "node wrote to its own disk (use a shared filesystem)")
    else:
        out["verdict"] = "FAILED"
        out["reason"] = base if base in _FIX else f"{base} (exit {exit_code})"
        out["fix"] = _FIX.get(base, "read log_tail; fix the error and re-submit")
    return out
