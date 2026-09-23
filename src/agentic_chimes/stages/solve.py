"""Solve the ChIMES least-squares fit via chimes_lsq.py.

Local algorithms only in this phase: svd, fast_svd, ridge, fast_ridge,
ridgecv, lasso, lassolars -- these all run directly through chimes_lsq.py's
numpy/scikit-learn code paths as a plain local subprocess. dlars/dlasso
(the DLARS-cliff-prone path -- see stages/_cliff_monitor.py, built and
unit-tested ahead of this wiring) needs the HPC submission layer and multi-
node srun/ibrun launch, not implemented here yet.

Exact CLI confirmed by reading
codes/chimes_lsq-LLfork/src/chimes_lsq.py's argparse block directly (not
just the earlier audit summary): --A --b --header --map --algorithm --alpha
--eps --folds --weights (all `str2bool` flags take "true"/"false" as
*strings*). The script prints the params.txt body to stdout, ending with a
literal "ENDFILE" line, and writes force.txt itself as a side-effect file in
its cwd -- this wrapper captures stdout into params.txt and leaves force.txt
where the script puts it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .. import config

NAME = "solve"
SUMMARY = "Solve for params.txt from A.txt/b.txt (local: svd/ridge/lassolars/...; dlars/dlasso pending HPC layer)."
SCHEMA = {
    "type": "object",
    "required": ["A", "b", "header", "map"],
    "properties": {
        "A": {"type": "string"},
        "b": {"type": "string"},
        "header": {"type": "string"},
        "map": {"type": "string"},
        "algorithm": {
            "type": "string",
            "enum": ["svd", "fast_svd", "ridge", "fast_ridge", "ridgecv", "lasso", "lassolars", "dlars", "dlasso"],
            "default": "svd",
        },
        "alpha": {"type": "number", "default": 1.0e-4},
        "eps": {"type": "number", "default": 1.0e-5},
        "weights": {"type": ["string", "null"]},
        "folds": {"type": "integer", "default": 4},
    },
}

_LOCAL_ALGORITHMS = {"svd", "fast_svd", "ridge", "fast_ridge", "ridgecv", "lasso", "lassolars"}


def add_arguments(parser) -> None:
    parser.add_argument("--A", dest="A", default=None)
    parser.add_argument("--b", dest="b", default=None)
    parser.add_argument("--header", default=None)
    parser.add_argument("--map", default=None)
    parser.add_argument("--algorithm", default="svd")
    parser.add_argument("--alpha", type=float, default=1.0e-4)
    parser.add_argument("--eps", type=float, default=1.0e-5)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--folds", type=int, default=4)


def run(args) -> dict:
    for req in ("A", "b", "header", "map"):
        if not getattr(args, req, None):
            raise ValueError(f"solve requires --{req} (or {req!r} in --json-in)")

    algorithm = args.algorithm or "svd"
    if algorithm not in _LOCAL_ALGORITHMS:
        raise NotImplementedError(
            f"algorithm={algorithm!r} needs the DLARS/HPC solve path, not wired up yet in this phase "
            f"(cliff-detection logic already exists and is unit-tested in stages/_cliff_monitor.py, "
            f"ahead of the HPC wiring). Supported now: {sorted(_LOCAL_ALGORITHMS)}."
        )

    chimes_lsq_py = config.CHIMES_LSQ_ROOT / "src" / "chimes_lsq.py"
    if not chimes_lsq_py.is_file():
        raise FileNotFoundError(f"chimes_lsq.py not found at {chimes_lsq_py}")

    work_dir = Path(args.output_dir) if getattr(args, "output_dir", None) else Path(args.A).resolve().parent
    work_dir.mkdir(parents=True, exist_ok=True)

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
