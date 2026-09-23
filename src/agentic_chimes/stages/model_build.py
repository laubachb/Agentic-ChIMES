"""Complete ChIMES model build: amat-build then solve, sequentially, in one
call. Composes those two stages' `run()` functions directly (see
stages/_compose.py) rather than requiring the caller to chain two separate
`chimes-agent` invocations for the common case of "I have an fm_setup.in,
give me a params.txt."

Does not include fm-setup-gen (call that separately first -- this stage
takes an already-written fm_setup.in) or evaluate (holdout scoring is a
deliberately separate decision, not bundled into every build).
"""

from __future__ import annotations

from pathlib import Path

from . import amat_build, solve
from ._compose import ns

NAME = "model-build"
SUMMARY = "Complete ChIMES model build: amat-build then solve, sequentially."
SCHEMA = {
    "type": "object",
    "required": ["fm_setup_in"],
    "properties": {
        "fm_setup_in": {"type": "string"},
        "chimes_lsq_bin": {"type": ["string", "null"]},
        "algorithm": {
            "type": "string",
            "enum": ["svd", "fast_svd", "ridge", "fast_ridge", "ridgecv", "lasso", "lassolars", "dlars", "dlasso"],
            "default": "svd",
        },
        "alpha": {"type": "number", "default": 1.0e-4},
        "eps": {"type": "number", "default": 1.0e-5},
        "weights": {"type": ["string", "null"]},
        "folds": {"type": "integer", "default": 4},
        "normalize": {"type": "boolean", "default": False, "description": "dlars/dlasso only."},
        "machine": {"type": ["string", "null"], "description": "If given, both amat-build and a dlars/dlasso solve submit via Slurm."},
        "queue": {"type": "string", "default": "batch"},
        "walltime_hours": {"type": "number", "default": 2.0},
        "nodes": {"type": "integer", "default": 1},
        "ntasks_per_node": {"type": ["integer", "null"]},
        "poll_interval_s": {"type": "integer", "default": 60},
    },
}


def add_arguments(parser) -> None:
    parser.add_argument("--fm-setup-in", dest="fm_setup_in", default=None)
    parser.add_argument("--chimes-lsq-bin", dest="chimes_lsq_bin", default=None)
    parser.add_argument("--algorithm", default="svd")
    parser.add_argument("--alpha", type=float, default=1.0e-4)
    parser.add_argument("--eps", type=float, default=1.0e-5)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--normalize", type=lambda s: s.lower() == "true", default=False)
    parser.add_argument("--machine", default=None)
    parser.add_argument("--queue", default="batch")
    parser.add_argument("--walltime-hours", dest="walltime_hours", type=float, default=2.0)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--ntasks-per-node", dest="ntasks_per_node", type=int, default=None)
    parser.add_argument("--poll-interval-s", dest="poll_interval_s", type=int, default=60)


def run(args) -> dict:
    if not getattr(args, "fm_setup_in", None):
        raise ValueError("model-build requires --fm-setup-in (or 'fm_setup_in' in --json-in)")

    output_dir = getattr(args, "output_dir", None)
    machine = getattr(args, "machine", None)

    amat_result = amat_build.run(
        ns(
            fm_setup_in=args.fm_setup_in,
            chimes_lsq_bin=getattr(args, "chimes_lsq_bin", None),
            machine=machine,
            queue=getattr(args, "queue", "batch"),
            walltime_hours=getattr(args, "walltime_hours", 2.0),
            nodes=getattr(args, "nodes", 1),
            ntasks_per_node=getattr(args, "ntasks_per_node", None),
            dry_run=False,
            output_dir=output_dir,
        )
    )

    work_dir = Path(amat_result["work_dir"])

    algorithm = getattr(args, "algorithm", "svd") or "svd"
    if amat_result.get("split") and algorithm not in ("dlars", "dlasso"):
        raise NotImplementedError(
            "fm_setup.in has SPLITFI true, which chimes_lsq only produces for the DLARS/DLASSO "
            f"path -- algorithm={algorithm!r} doesn't read split A/b files. Use --algorithm dlars/dlasso."
        )

    # in split mode (SPLITFI true), amat-build's output has no "A"/"b"/"dim"
    # keys at all (only A.0000.txt etc exist) -- fall back to the canonical
    # work_dir/A.txt path string, which _ensure_canonical_link's early-return
    # (dst resolves to the same path) correctly treats as a no-op, since the
    # split files are already sitting in work_dir under dlars' own expected names
    A = amat_result.get("A") or str(work_dir / "A.txt")
    b = amat_result.get("b") or str(work_dir / "b.txt")
    dim = amat_result.get("dim") or str(work_dir / "dim.txt")

    solve_result = solve.run(
        ns(
            A=A,
            b=b,
            header=amat_result["params_header"],
            map=amat_result["ff_groups_map"],
            dim=dim,
            algorithm=algorithm,
            alpha=getattr(args, "alpha", 1.0e-4),
            eps=getattr(args, "eps", 1.0e-5),
            weights=getattr(args, "weights", None),
            folds=getattr(args, "folds", 4),
            normalize=getattr(args, "normalize", False),
            split_files=bool(amat_result.get("split")),
            machine=machine,
            queue=getattr(args, "queue", "batch"),
            walltime_hours=getattr(args, "walltime_hours", 2.0),
            nodes=getattr(args, "nodes", 1),
            ntasks_per_node=getattr(args, "ntasks_per_node", None),
            poll_interval_s=getattr(args, "poll_interval_s", 60),
            dry_run=False,
            output_dir=str(work_dir),
        )
    )

    return {"work_dir": str(work_dir), "amat_build": amat_result, "solve": solve_result, "params": solve_result["params"]}
