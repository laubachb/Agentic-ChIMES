"""Solve the ChIMES least-squares fit via chimes_lsq.py.

Local algorithms (svd, fast_svd, ridge, fast_ridge, ridgecv, lasso,
lassolars) run directly through chimes_lsq.py's numpy/scikit-learn code
paths as a plain local subprocess. `dlars`/`dlasso` (the DLARS-cliff-prone
path) require `machine` and submit via stages/_dlars_hpc.py, which polls
the live job's dlars.log with stages/_cliff_monitor.py and auto-finalizes
on a detected cliff -- see docs/commands/solve.md.

Exact CLI confirmed by reading codes/chimes_lsq-LLfork/src/chimes_lsq.py's
argparse block directly: --A --b --header --map --algorithm --alpha --eps
--folds --weights (all `str2bool` flags take "true"/"false" as *strings*).
The script prints the params.txt body to stdout, ending with a literal
"ENDFILE" line, and writes force.txt itself as a side-effect file in its
cwd -- the local path here captures stdout into params.txt and leaves
force.txt where the script puts it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .. import config, machines
from . import _dlars_hpc

NAME = "solve"
SUMMARY = "Solve for params.txt from A.txt/b.txt (local: svd/ridge/lassolars/...; dlars/dlasso via --machine)."
SCHEMA = {
    "type": "object",
    "required": ["A", "b", "header", "map"],
    "properties": {
        "A": {"type": "string"},
        "b": {"type": "string"},
        "header": {"type": "string"},
        "map": {"type": "string"},
        "dim": {"type": ["string", "null"], "description": "dim.txt from amat-build; required for algorithm=dlars/dlasso."},
        "algorithm": {
            "type": "string",
            "enum": ["svd", "fast_svd", "ridge", "fast_ridge", "ridgecv", "lasso", "lassolars", "dlars", "dlasso"],
            "default": "svd",
        },
        "alpha": {"type": "number", "default": 1.0e-4},
        "eps": {"type": "number", "default": 1.0e-5},
        "weights": {"type": ["string", "null"]},
        "folds": {"type": "integer", "default": 4},
        "normalize": {"type": "boolean", "default": False, "description": "dlars/dlasso only -- dlars itself warns 'normalize should not be used with chimes_lsq' (confirmed on a real run: normalize=true causes an immediate MKL error / zero-variable stall)."},
        "split_files": {"type": "boolean", "default": False, "description": "dlars/dlasso only; must match fm_setup.in's SPLITFI."},
        "machine": {"type": ["string", "null"], "description": "Required for algorithm=dlars/dlasso."},
        "queue": {"type": "string", "default": "batch"},
        "walltime_hours": {"type": "number", "default": 2.0},
        "nodes": {"type": "integer", "default": 1},
        "ntasks_per_node": {"type": ["integer", "null"]},
        "poll_interval_s": {"type": "integer", "default": 60},
    },
}

_LOCAL_ALGORITHMS = {"svd", "fast_svd", "ridge", "fast_ridge", "ridgecv", "lasso", "lassolars"}
_HPC_ALGORITHMS = {"dlars", "dlasso"}


def add_arguments(parser) -> None:
    parser.add_argument("--A", dest="A", default=None)
    parser.add_argument("--b", dest="b", default=None)
    parser.add_argument("--header", default=None)
    parser.add_argument("--map", default=None)
    parser.add_argument("--dim", default=None)
    parser.add_argument("--algorithm", default="svd")
    parser.add_argument("--alpha", type=float, default=1.0e-4)
    parser.add_argument("--eps", type=float, default=1.0e-5)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--normalize", type=lambda s: s.lower() == "true", default=False)
    parser.add_argument("--split-files", dest="split_files", action="store_true")
    parser.add_argument("--machine", default=None)
    parser.add_argument("--queue", default="batch")
    parser.add_argument("--walltime-hours", dest="walltime_hours", type=float, default=2.0)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--ntasks-per-node", dest="ntasks_per_node", type=int, default=None)
    parser.add_argument("--poll-interval-s", dest="poll_interval_s", type=int, default=60)


def _run_local(args, algorithm: str, work_dir: Path) -> dict:
    chimes_lsq_py = config.CHIMES_LSQ_ROOT / "src" / "chimes_lsq.py"
    if not chimes_lsq_py.is_file():
        raise FileNotFoundError(f"chimes_lsq.py not found at {chimes_lsq_py}")

    cmd = [
        sys.executable,
        str(chimes_lsq_py),
        "--A", str(Path(args.A).resolve()),
        "--b", str(Path(args.b).resolve()),
        "--header", str(Path(args.header).resolve()),
        "--map", str(Path(args.map).resolve()),
        "--algorithm", algorithm,
        "--alpha", str(args.alpha),
        "--eps", str(args.eps),
        "--folds", str(args.folds),
    ]
    if args.weights:
        cmd += ["--weights", str(Path(args.weights).resolve())]

    proc = subprocess.run(cmd, cwd=str(work_dir), capture_output=True, text=True)
    log_path = work_dir / "solve.log"
    log_path.write_text(proc.stderr or "")

    if proc.returncode != 0:
        raise RuntimeError(f"chimes_lsq.py exited {proc.returncode}; see {log_path}\n{(proc.stderr or '')[-2000:]}")

    params_path = work_dir / "params.txt"
    params_path.write_text(proc.stdout)

    if "ENDFILE" not in proc.stdout:
        raise RuntimeError(f"chimes_lsq.py did not emit an ENDFILE-terminated params.txt; see {params_path}")

    force_path = work_dir / "force.txt"

    return {
        "params": str(params_path),
        "force": str(force_path) if force_path.is_file() else None,
        "algorithm": algorithm,
        "log": str(log_path),
    }


def _ensure_canonical_link(src: str, work_dir: Path, canonical_name: str) -> None:
    """dlars/chimes_lsq.py's fit_dlars hardcodes 'dim.txt' regardless of
    what --A/--b were given -- safest is to always invoke it with plain
    "A.txt"/"b.txt"/"dim.txt" relative to the job's cwd (work_dir), matching
    amat-build's own output naming. Symlink the given file in under that
    name if it isn't already there (e.g. solve's --output-dir differs from
    amat-build's).

    Split output (A.0000.txt, ...) is not relinked here -- for
    algorithm=dlars/dlasso with split_files=true, run solve with the same
    --output-dir amat-build used, so the split files are already present
    under their canonical names."""
    src_path = Path(src).resolve()
    dst_path = work_dir / canonical_name
    if dst_path.resolve() == src_path:
        return
    if dst_path.exists() or dst_path.is_symlink():
        dst_path.unlink()
    dst_path.symlink_to(src_path)


def run(args) -> dict:
    for req in ("A", "b", "header", "map"):
        if not getattr(args, req, None):
            raise ValueError(f"solve requires --{req} (or {req!r} in --json-in)")

    algorithm = args.algorithm or "svd"
    work_dir = Path(getattr(args, "output_dir", None) or Path(args.A).resolve().parent)
    work_dir.mkdir(parents=True, exist_ok=True)

    if algorithm in _LOCAL_ALGORITHMS:
        return _run_local(args, algorithm, work_dir)

    if algorithm not in _HPC_ALGORITHMS:
        raise ValueError(f"unknown algorithm {algorithm!r}; known: {sorted(_LOCAL_ALGORITHMS | _HPC_ALGORITHMS)}")

    machine = getattr(args, "machine", None)
    if not machine:
        raise ValueError(f"algorithm={algorithm!r} requires --machine (dlars/dlasso run via HPC submission)")
    if not getattr(args, "dim", None):
        raise ValueError(f"algorithm={algorithm!r} requires --dim (dim.txt from amat-build)")

    _ensure_canonical_link(args.A, work_dir, "A.txt")
    _ensure_canonical_link(args.b, work_dir, "b.txt")
    _ensure_canonical_link(args.dim, work_dir, "dim.txt")

    profile = machines.load_profile(machine)
    result = _dlars_hpc.run_dlars_hpc(
        profile=profile,
        work_dir=work_dir,
        header=str(Path(args.header).resolve()),
        map_file=str(Path(args.map).resolve()),
        algorithm=algorithm,
        alpha=args.alpha,
        normalize=getattr(args, "normalize", False),
        split_files=bool(getattr(args, "split_files", False)),
        weights=str(Path(args.weights).resolve()) if getattr(args, "weights", None) else None,
        nodes=getattr(args, "nodes", 1) or 1,
        ntasks_per_node=getattr(args, "ntasks_per_node", None),
        walltime_hours=getattr(args, "walltime_hours", 2.0) or 2.0,
        queue=getattr(args, "queue", "batch") or "batch",
        poll_interval_s=getattr(args, "poll_interval_s", 60) or 60,
        dry_run=bool(getattr(args, "dry_run", False)),
    )
    result["algorithm"] = algorithm
    return result
