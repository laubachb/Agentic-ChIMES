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
    (root / "study.json").write_text(json.dumps(data, indent=1))


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


def run(args) -> dict:
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    if getattr(args, "init", None):
        root = Path(args.init).resolve()
        root.mkdir(parents=True, exist_ok=True)
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
    return _status(root)
