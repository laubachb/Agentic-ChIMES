"""CPU-hours spent on a study, by phase and by job.

Slurm jobs come from `sacct`, filtered by working directory: every job
whose WorkDir is inside the study (or one of its `extra_roots`) counts,
including jobs al_driver submits itself, with no registration needed.
Charged hours are allocated cores x elapsed time (sacct CPUTimeRAW), the
quantity allocations are billed in. Login-node work is read from
`usage/local.jsonl`, which the CLI appends to whenever a stage writes inside
a study outside Slurm.

Phase is taken from the path (01_data -> data, 02_fit -> fit, 03_al ->
active_learning, 04_md -> md, 05_bench -> benchmark, 06_deploy -> deploy).
Jobs in an extra root are attributed by `phase_hints` or by job name.
"""

from __future__ import annotations

import csv
import datetime
import json
import subprocess
from collections import defaultdict
from pathlib import Path

from . import study as study_stage
from ..io import atomic

NAME = "usage"
SUMMARY = "CPU-hours used by a study (Slurm via sacct + login-node ledger), by phase and job."
USES_OUTPUT_DIR = False
PHASE_DIRS = {"01_data": "data", "02_fit": "fit", "03_al": "active_learning", "04_md": "md",
              "05_bench": "benchmark", "06_deploy": "deploy"}
JOBNAME_PHASE = {"qe-relabel": "data", "hyper-search": "fit", "solver-study": "fit", "chimes-dlars": "fit",
                 "chimes-dlars-finalize": "fit", "amat-build": "fit", "benchmark": "benchmark", "chimes-bench": "benchmark"}
SCHEMA = {
    "type": "object",
    "required": ["study"],
    "properties": {
        "study": {"type": "string"},
        "since": {"type": ["string", "null"], "description": "sacct start date (YYYY-MM-DD); default: 30 days before the study was created."},
        "phase_hints": {"type": ["object", "null"], "description": "path prefix -> phase, for jobs outside the standard layout."},
    },
}


def add_arguments(parser) -> None:
    parser.add_argument("--study", default=None)
    parser.add_argument("--since", default=None)
    parser.add_argument("--phase-hints", dest="phase_hints", type=json.loads, default=None)


def _sacct(since: str) -> list:
    fields = "JobID,JobName,WorkDir,AllocCPUS,NNodes,ElapsedRaw,CPUTimeRAW,TotalCPU,State,Partition,Submit"
    # No -X: with it, TotalCPU is empty (consumed CPU is recorded on steps); the
    # job's own line (no "." in JobID) carries the step totals.
    proc = subprocess.run(["sacct", "-P", "-n", "-S", since, "-o", fields], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"sacct failed: {proc.stderr.strip()[:300]}")
    keys = fields.split(",")
    rows = [dict(zip(keys, line.split("|"))) for line in proc.stdout.splitlines() if line.count("|") == len(keys) - 1]
    return [r for r in rows if "." not in r["JobID"]]


def phase_of(path: str, root: Path, hints: dict, job_name: str = "") -> str:
    for prefix, phase in sorted((hints or {}).items(), key=lambda kv: -len(kv[0])):
        if path.startswith(str(Path(prefix).resolve())):
            return phase
    try:
        rel = Path(path).resolve().relative_to(root)
        if rel.parts and rel.parts[0] in PHASE_DIRS:
            return PHASE_DIRS[rel.parts[0]]
    except ValueError:
        pass
    return JOBNAME_PHASE.get(job_name, "other")


def parse_slurm_duration(s: str) -> float:
    """sacct TotalCPU/Elapsed text ([D-]HH:MM:SS[.mmm] or MM:SS.mmm) -> seconds."""
    s = (s or "").strip()
    if not s:
        return 0.0
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    parts = [float(x) for x in s.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, sec = parts
    return days * 86400 + h * 3600 + m * 60 + sec


def _hours(sec) -> float:
    return round(float(sec) / 3600.0, 3)


def summarize(jobs: list, local: list) -> dict:
    by_phase = defaultdict(lambda: {"cpu_hours": 0.0, "used_cpu_hours": 0.0, "node_hours": 0.0, "wall_hours": 0.0,
                                    "jobs": 0, "local_cpu_hours": 0.0})
    for j in jobs:
        ph = by_phase[j["phase"]]
        ph["cpu_hours"] += j["cpu_hours"]
        ph["used_cpu_hours"] += j["used_cpu_hours"]
        ph["node_hours"] += j["node_hours"]
        ph["wall_hours"] += j["wall_hours"]
        ph["jobs"] += 1
    for rec in local:
        by_phase[rec["phase"]]["local_cpu_hours"] += rec["cpu_s"] / 3600.0
    by_phase = {k: {kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in by_phase.items()}
    alloc = sum(j["cpu_hours"] for j in jobs)
    used = sum(j["used_cpu_hours"] for j in jobs)
    return {
        "total_cpu_hours": round(alloc, 3),
        "total_used_cpu_hours": round(used, 3),
        "allocation_efficiency": round(used / alloc, 3) if alloc else None,
        "total_node_hours": round(sum(j["node_hours"] for j in jobs), 3),
        "total_local_cpu_hours": round(sum(r["cpu_s"] for r in local) / 3600.0, 3),
        "n_jobs": len(jobs),
        "by_phase": by_phase,
    }


def run(args) -> dict:
    root = study_stage.find_study(getattr(args, "study", None) or ".")
    if root is None:
        raise ValueError("usage needs a study (a directory with study.json; create one with `study --init`)")
    data = study_stage.load(root)
    roots = [root] + [Path(p) for p in data.get("extra_roots", [])]
    since = getattr(args, "since", None)
    if not since:
        created = datetime.datetime.fromisoformat(data["created_at"])
        since = (created - datetime.timedelta(days=30)).date().isoformat()
    hints = getattr(args, "phase_hints", None) or data.get("phase_hints") or {}

    jobs = []
    for row in _sacct(since):
        wd = row.get("WorkDir") or ""
        if not any(wd == str(r) or wd.startswith(str(r) + "/") for r in roots):
            continue
        cpu_s, wall_s = int(row["CPUTimeRAW"] or 0), int(row["ElapsedRaw"] or 0)
        jobs.append({
            "job_id": row["JobID"], "name": row["JobName"], "work_dir": wd, "state": row["State"],
            "partition": row["Partition"], "submitted": row["Submit"], "alloc_cpus": int(row["AllocCPUS"] or 0),
            "nodes": int(row["NNodes"] or 0), "wall_hours": _hours(wall_s), "cpu_hours": _hours(cpu_s),
            "node_hours": _hours(wall_s * int(row["NNodes"] or 0)),
            "used_cpu_hours": _hours(parse_slurm_duration(row["TotalCPU"])),
            "phase": phase_of(wd, root, hints, row["JobName"]),
        })

    local = []
    ledger = root / "usage" / "local.jsonl"
    if ledger.is_file():
        for line in ledger.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                rec["phase"] = phase_of(rec.get("output_dir", ""), root, hints, rec.get("stage", ""))
                local.append(rec)

    summary = summarize(jobs, local)
    out = root / "usage"
    out.mkdir(exist_ok=True)
    report = {"study": str(root), "since": since, "roots": [str(r) for r in roots], **summary,
              "jobs": sorted(jobs, key=lambda j: j["submitted"]),
              "local_runs": len(local),
              "notes": ["cpu_hours = allocated cores x elapsed (what the allocation is charged); "
                        "used_cpu_hours = CPU time the processes consumed (sacct TotalCPU); "
                        "allocation_efficiency = used / allocated",
                        "jobs still running are counted up to now"]}
    path = out / "usage_report.json"
    atomic.write_json(path, report, indent=1)
    with open(out / "usage_jobs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["job_id", "name", "phase", "state", "partition", "nodes", "alloc_cpus",
                                          "wall_hours", "cpu_hours", "used_cpu_hours", "node_hours", "work_dir", "submitted"], extrasaction="ignore")
        w.writeheader()
        w.writerows(report["jobs"])
    return {"usage_report": str(path), **summary, "since": since}
