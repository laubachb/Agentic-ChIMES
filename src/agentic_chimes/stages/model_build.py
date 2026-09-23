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


def run(args) -> dict:
    if not getattr(args, "fm_setup_in", None):
        raise ValueError("model-build requires --fm-setup-in (or 'fm_setup_in' in --json-in)")

    output_dir = getattr(args, "output_dir", None)

    amat_result = amat_build.run(
        ns(
            fm_setup_in=args.fm_setup_in,
            chimes_lsq_bin=getattr(args, "chimes_lsq_bin", None),
            output_dir=output_dir,
        )
    )

    work_dir = Path(amat_result["work_dir"])
    if amat_result.get("split"):
        raise NotImplementedError(
            "fm_setup.in has SPLITFI true (needed for dlars on a large basis); "
            "the split A/b path isn't wired into model-build's solve step yet -- "
            "run amat-build and solve separately for split output."
        )

    solve_result = solve.run(
        ns(
            A=amat_result["A"],
            b=amat_result["b"],
            header=amat_result["params_header"],
            map=amat_result["ff_groups_map"],
            algorithm=getattr(args, "algorithm", "svd") or "svd",
            alpha=getattr(args, "alpha", 1.0e-4),
            eps=getattr(args, "eps", 1.0e-5),
            weights=getattr(args, "weights", None),
            folds=getattr(args, "folds", 4),
            output_dir=str(work_dir),
        )
    )

    return {"work_dir": str(work_dir), "amat_build": amat_result, "solve": solve_result, "params": solve_result["params"]}
