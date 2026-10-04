"""A study: one directory holding every phase of one ChIMES model's
development, marked by `study.json`.

`study.json` records the goal and a registry of the artifacts each phase
produced (`data_manifest`, `hyper_report`, `params`, `benchmark`, ...), so
later phases, the usage accounting and the final report find them without
guessing paths. Standard layout, which `--init` creates:

    <study>/study.json  STUDY.md
    01_data/  02_fit/  03_al/  04_md/  05_bench/  06_deploy/  usage/

Stages that write inside a study (any `--output-dir` below a `study.json`)
add their login-node CPU time to `usage/local.jsonl` automatically (see
cli.py); Slurm jobs are found by `usage` through sacct's working directory.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from ..io import atomic
from ..io import fs

NAME = "study"
SUMMARY = "Create a study directory, register phase artifacts, or show a study's status."
USES_OUTPUT_DIR = False
LAYOUT = ("01_data", "02_fit", "03_al", "04_md", "05_bench", "06_deploy", "usage")
KNOWN_ARTIFACTS = {
    "data_manifest": "01_data: data_manifest.json from data-curate",
    "hyper_report": "02_fit: hyper_report.json from hyper-search",
    "params": "final params.txt",
    "fm_setup": "final fm_setup.in",
    "al_run": "03_al: al_driver study directory",
    "md_runs": "04_md: lammps-run output directories (list)",
    "md_check": "04_md: md_check.json from md-check (the model's MD validation)",
    "eos_check": "04_md: eos_check.json from eos-check",
    "fingerprint": "fingerprint.json: coverage of MD/candidate frames vs training",
    "quests": "quests.json: QUESTS entropy/novelty of the data or a harvest",
    "committee": "committee.json: bootstrap-committee uncertainty",
    "learning_curve": "learning_curve.json: data-limited or plateau",
    "evaluate": "evaluate output directory with parity plots (evaluate --plot)",
    "al_status": "03_al/AL_STATUS.json from al-status",
    "benchmark": "05_bench: benchmark.json from benchmark --collect",
    "usage": "usage_report.json from usage",
    "deploy": "06_deploy: deployed model package",
}
SCHEMA = {
    "type": "object",
    "properties": {
        "init": {"type": ["string", "null"], "description": "Create a study at this directory (under /p/lustre2 for HPC work)."},
        "study": {"type": ["string", "null"], "description": "An existing study directory (or any path inside one)."},
        "name": {"type": ["string", "null"]},
        "goal": {"type": ["string", "null"], "description": "The user's request, verbatim."},
        "elements": {"type": ["array", "null"], "items": {"type": "string"}},
        "register": {"type": ["array", "null"], "items": {"type": "string"},
                     "description": "key=path entries; keys: " + ", ".join(KNOWN_ARTIFACTS) + " (md_runs appends)."},
        "extra_roots": {"type": ["array", "null"], "items": {"type": "string"},
                        "description": "Other directories whose Slurm jobs belong to this study (for usage accounting)."},
        "status": {"type": "boolean", "default": False,
                   "description": "Add a one-screen dashboard (phases, jobs, what waits on the user, CPU-hours) and write STATUS.md."},
    },
}


def add_arguments(parser) -> None:
    parser.add_argument("--init", default=None)
    parser.add_argument("--study", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--goal", default=None)
    parser.add_argument("--elements", type=lambda s: [x for x in s.split(",") if x], default=None)
    parser.add_argument("--register", action="append", default=None, help="key=path (repeatable)")
    parser.add_argument("--extra-root", dest="extra_roots", action="append", default=None)
    parser.add_argument("--status", action="store_true", default=False,
                        help="One-screen dashboard: phases, jobs, what waits on the user, CPU-hours (also written to STATUS.md).")


def find_study(path) -> Path | None:
    """The nearest ancestor (or self) containing study.json."""
    p = Path(path).resolve()
    for d in [p] + list(p.parents):
        if (d / "study.json").is_file():
            return d
    return None


def load(root: Path) -> dict:
    return json.loads((root / "study.json").read_text())


def save(root: Path, data: dict) -> None:
    atomic.write_json((root / "study.json"), data, indent=1)


def artifact(root: Path, key: str):
    """Registered path for `key`, else the standard-layout default if present."""
    data = load(root)
    reg = data.get("artifacts", {})
    if key in reg:
        return reg[key]
    defaults = {
        "data_manifest": root / "01_data" / "curate" / "data_manifest.json",
        "hyper_report": root / "02_fit" / "search" / "hyper_report.json",
        "params": root / "02_fit" / "search" / "best" / "params.txt",
        "fm_setup": root / "02_fit" / "search" / "best" / "fm_setup.in",
        "benchmark": root / "05_bench" / "benchmark.json",
        "usage": root / "usage" / "usage_report.json",
        "deploy": root / "06_deploy",
    }
    if key == "md_runs":
        md = root / "04_md"
        return sorted(str(p.parent) for p in md.glob("*/log.lammps")) if md.is_dir() else []
    d = defaults.get(key)
    return str(d) if d is not None and d.exists() else None


def _status(root: Path) -> dict:
    data = load(root)
    arts = {k: artifact(root, k) for k in KNOWN_ARTIFACTS}
    return {"study": str(root), "name": data.get("name"), "goal": data.get("goal"), "elements": data.get("elements"),
            "created_at": data.get("created_at"), "artifacts": arts,
            "missing": [k for k, v in arts.items() if not v],
            "extra_roots": data.get("extra_roots", [])}


PHASES = (
    ("Data", ("data_manifest",), "chimes-data-curator: data-search / data-fetch / data-curate"),
    ("Hyperparameters", ("hyper_report",), "chimes-hyperparameter-tuner: hyper-analyze, hyper-search --cv-folds 4"),
    ("Model", ("params", "fm_setup"), "register the chosen params.txt and fm_setup.in; chimes-fit-reviewer"),
    ("MD validation", ("md_check", "eos_check", "fingerprint", "quests"), "chimes-md-validator: md-check, eos-check, fingerprint, quests"),
    ("Active learning", ("al_status",), "chimes-active-learner: al-batch -> qe-relabel -> al-merge -> refit -> al-status"),
    ("Benchmark", ("benchmark",), "chimes-benchmark: benchmark, usage"),
    ("Deploy", ("deploy",), "deploy --study"),
)
_PRUNE = {"points", "amat", "__pycache__", ".git", "codes", "deps"}


def _read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError, TypeError):
        return None


def _artifact_json(path, name):
    """A registered artifact may be the JSON file or its directory (possibly with a run/ level)."""
    if not path:
        return None
    p = Path(path)
    if p.is_file():
        return _read(p)
    for cand in (p / name, p / "run" / name):
        if cand.is_file():
            return _read(cand)
    return None


def _headline(root: Path, arts: dict) -> dict:
    """One short fact per phase, read from the artifacts that exist."""
    out = {}
    m = _read(arts.get("data_manifest")) if arts.get("data_manifest") else None
    if m:
        out["Data"] = f"{m.get('n_train')} train / {m.get('n_holdout')} holdout frames, {len(m.get('warnings') or [])} warning(s)"
    h = _read(arts.get("hyper_report")) if arts.get("hyper_report") else None
    if h and h.get("final"):
        f = h["final"]
        err, se = f.get("holdout_relative_force_error"), f.get("holdout_relative_force_se")
        out["Hyperparameters"] = (f"relative force error {err:.3f}" + (f" ± {se:.3f}" if se is not None else "")
                                  + (" (cross-validated)" if h.get("cross_validation") else "")) if err is not None else "search finished"
    mc = _artifact_json(arts.get("md_check"), "md_check.json")
    if mc and mc.get("models"):
        stable = [bool(x.get("stable_at_all_temperatures")) for x in mc["models"]]
        below = sum(int(x.get("below_inner_cutoff_frames") or 0) for x in mc["models"])
        out["MD validation"] = f"{sum(stable)}/{len(stable)} model(s) stable at every temperature; {below} frame(s) inside an inner cutoff"
    al = _artifact_json(arts.get("al_status"), "AL_STATUS.json") or _read(root / "03_al" / "AL_STATUS.json")
    if al:
        out["Active learning"] = f"{al.get('verdict')} after {len(al.get('rounds') or [])} round(s)"
    b = _read(arts.get("benchmark")) if arts.get("benchmark") else None
    if b:
        unit = (b.get("cost_model") or {}).get("core_s_per_atom_step")
        out["Benchmark"] = f"{unit:.2e} core-s per atom-step" if isinstance(unit, float) else "benchmark collected"
    return out


def _jobs(roots) -> tuple:
    """(submitted jobs, prepared-but-unsubmitted job directories) under the study, with one squeue call."""
    import getpass
    import os
    import subprocess

    submitted, prepared = [], []
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in _PRUNE and not d.startswith("frame_")]
            if dirpath[len(str(root)):].count(os.sep) >= 6:
                dirnames[:] = []
            if "job.json" in filenames:
                rec = _read(Path(dirpath) / "job.json") or {}
                if rec.get("job_id"):
                    submitted.append({"work_dir": dirpath, **{k: rec.get(k) for k in ("job_id", "job_name", "submitted_at", "expect")}})
            elif "run.cmd" in filenames and "stdoutmsg" not in filenames:
                # a rendered job script with no job record and no Slurm output: a dry run nobody has submitted
                prepared.append(dirpath)
    queue = {}
    try:
        q = subprocess.run(["squeue", "-h", "-u", getpass.getuser(), "-o", "%i|%T|%M"], capture_output=True, text=True, timeout=20)
        for ln in q.stdout.splitlines():
            t = ln.split("|")
            if len(t) == 3:
                queue[t[0]] = (t[1], t[2])
    except (OSError, subprocess.TimeoutExpired):
        queue = None
    for j in submitted:
        expect = j.pop("expect") or []
        if queue is not None and str(j["job_id"]) in queue:
            state, elapsed = queue[str(j["job_id"])]
            j.update({"state": "QUEUED" if state == "PENDING" else "RUNNING", "elapsed": elapsed})
        elif expect and all(Path(e).is_file() for e in expect):
            j["state"] = "FINISHED"
        elif expect:
            j["state"] = "NO_RESULTS"       # left the queue without its result files: job-status says why
        else:
            j["state"] = "FINISHED?" if queue is not None else "UNKNOWN"
    submitted.sort(key=lambda j: -(j.get("submitted_at") or 0))
    return submitted, sorted(prepared)


def dashboard(root: Path) -> dict:
    data = load(root)
    arts = {k: artifact(root, k) for k in KNOWN_ARTIFACTS}
    facts = _headline(root, arts)
    phases = []
    for name, keys, how in PHASES:
        have = [k for k in keys if arts.get(k)]
        state = "done" if len(have) == len(keys) else "partial" if have else "not started"
        phases.append({"phase": name, "state": state, "have": have, "missing": [k for k in keys if k not in have],
                       "headline": facts.get(name), "how": how})
    report = root / "REPORT.md"
    narrative_left = report.is_file() and "<!-- NARRATIVE" in report.read_text()
    phases.append({"phase": "Report", "state": "partial" if narrative_left else "done" if report.is_file() else "not started",
                   "have": ["REPORT.md"] if report.is_file() else [], "missing": [] if report.is_file() else ["REPORT.md"],
                   "headline": "narrative placeholders remain" if narrative_left else None,
                   "how": "study-report --study, then chimes-report-writer"})
    roots = [root] + [Path(p) for p in data.get("extra_roots", []) if Path(p).is_dir()]
    submitted, prepared = _jobs(roots)
    active = [j for j in submitted if j["state"] in ("QUEUED", "RUNNING")]
    problems = [j for j in submitted if j["state"] == "NO_RESULTS"]
    waiting = [f"approve or discard the prepared job in {p}" for p in prepared]
    waiting += [f"job {j['job_id']} ({j['job_name']}) left the queue without results: chimes-agent job-status --work-dir {j['work_dir']}"
                for j in problems]
    usage = _read(arts.get("usage")) if arts.get("usage") else None
    local_s = 0.0
    ledger = root / "usage" / "local.jsonl"
    if ledger.is_file():
        for ln in ledger.read_text().splitlines():
            try:
                local_s += float(json.loads(ln).get("cpu_s", 0.0))
            except ValueError:
                pass
    cpu = {"login_node_cpu_hours": round(local_s / 3600.0, 3)}
    if usage:
        cpu.update({"slurm_charged_cpu_hours": usage.get("total_cpu_hours"), "slurm_used_cpu_hours": usage.get("total_used_cpu_hours")})
    else:
        cpu["note"] = "Slurm CPU-hours not collected yet: chimes-agent usage --study <dir>"
    nxt = next((p for p in phases if p["state"] != "done"), None)
    return {"phases": phases, "jobs_active": active, "jobs_recent": submitted[:5], "n_jobs_submitted": len(submitted),
            "waiting_on_user": waiting, "cpu_hours": cpu,
            "next": f"{nxt['phase']} ({nxt['state']}): {nxt['how']}" if nxt else "all phases done"}


def render_dashboard(status: dict, dash: dict) -> str:
    mark = {"done": "[x]", "partial": "[~]", "not started": "[ ]"}
    L = [f"# {status.get('name')}: status", "", f"Goal: {status.get('goal') or '(none recorded)'}", "", "## Phases", ""]
    for p in dash["phases"]:
        tail = f": {p['headline']}" if p.get("headline") else ""
        miss = f" (missing: {', '.join(p['missing'])})" if p["state"] == "partial" and p["missing"] else ""
        L.append(f"- {mark[p['state']]} {p['phase']}{tail}{miss}")
    L += ["", f"Next: {dash['next']}", "", "## Jobs", ""]
    if dash["jobs_active"]:
        L += [f"- {j['state']} {j['job_id']} {j['job_name']} ({j.get('elapsed')}) in {j['work_dir']}" for j in dash["jobs_active"]]
    else:
        L.append(f"- none queued or running ({dash['n_jobs_submitted']} submitted so far)")
    L += ["", "## Waiting on you", ""] + ([f"- {w}" for w in dash["waiting_on_user"]] or ["- nothing"])
    c = dash["cpu_hours"]
    L += ["", "## CPU-hours", ""]
    if "slurm_charged_cpu_hours" in c:
        L.append(f"- Slurm: {c['slurm_charged_cpu_hours']} charged, {c['slurm_used_cpu_hours']} used")
    else:
        L.append(f"- {c['note']}")
    L.append(f"- login node: {c['login_node_cpu_hours']}")
    return "\n".join(L) + "\n"


def run(args) -> dict:
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    if getattr(args, "init", None):
        root = Path(args.init).resolve()
        fs.ensure_dir(root)
        if (root / "study.json").is_file():
            raise ValueError(f"{root} is already a study; use --study {root}")
        for d in LAYOUT:
            (root / d).mkdir(exist_ok=True)
        data = {"name": getattr(args, "name", None) or root.name, "goal": getattr(args, "goal", None),
                "elements": getattr(args, "elements", None), "created_at": now, "artifacts": {},
                "extra_roots": [str(Path(p).resolve()) for p in (getattr(args, "extra_roots", None) or [])],
                "log": [{"at": now, "event": "created"}]}
        save(root, data)
        if not (root / "STUDY.md").exists():
            (root / "STUDY.md").write_text(f"# {data['name']}\n\n**Goal:** {data['goal'] or '(fill in)'}\n\n"
                                           "## Decisions\n\n## Phase status\n")
    else:
        where = getattr(args, "study", None)
        if not where:
            raise ValueError("give --init DIR or --study DIR")
        root = find_study(where)
        if root is None:
            raise ValueError(f"no study.json at or above {where}")
        data = load(root)
        changed = False
        for k in ("name", "goal", "elements"):
            if getattr(args, k, None):
                data[k] = getattr(args, k)
                changed = True
        for p in getattr(args, "extra_roots", None) or []:
            rp = str(Path(p).resolve())
            if rp not in data.setdefault("extra_roots", []):
                data["extra_roots"].append(rp)
                changed = True
        for entry in getattr(args, "register", None) or []:
            key, _, path = entry.partition("=")
            if key not in KNOWN_ARTIFACTS:
                raise ValueError(f"unknown artifact key {key!r}; known: {sorted(KNOWN_ARTIFACTS)}")
            if not Path(path).exists():
                raise FileNotFoundError(f"{key}: {path} does not exist")
            path = str(Path(path).resolve())
            if key == "md_runs":
                runs = data.setdefault("artifacts", {}).setdefault("md_runs", [])
                if path not in runs:
                    runs.append(path)
            else:
                data.setdefault("artifacts", {})[key] = path
            data.setdefault("log", []).append({"at": now, "event": f"registered {key}={path}"})
            changed = True
        if changed:
            save(root, data)
    status = _status(root)
    if getattr(args, "status", False):
        dash = dashboard(root)
        text = render_dashboard(status, dash)
        atomic.write_text(root / "STATUS.md", text)
        status.update({"dashboard": dash, "dashboard_text": text, "status_md": str(root / "STATUS.md")})
    return status
