"""Bootstrap committee of one ChIMES basis: uncertainty for active learning.

Each member refits the same basis (one design matrix, built once) on a
bootstrap resample of the *training frames*. A frame drawn k times gets
weight sqrt(k) on all its rows, which is what duplicating its rows does to a
least-squares objective, through the solver's ordinary `--weights` input.
Members therefore use exactly the solver and alpha of the original fit.

Where members disagree, the data does not pin the model down. With
`--candidates-xyzf` (e.g. an md-check harvest or a fingerprint `novel.xyzf`),
the stage ranks candidate frames by committee spread:

- force spread: RMS over atoms and components of the standard deviation
  across members;
- energy spread: standard deviation per atom.

It writes the most uncertain frames to `uncertain.xyzf`, ready for QE. This
is query-by-committee, which al_driver documents but never finished; it
complements fingerprint novelty (structure) with model disagreement.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ..io import atomic, fs
from ..io import xyzf as xyzf_io

NAME = "committee"
SUMMARY = "Bootstrap committee of one basis (frame-resampled refits) and candidate ranking by model disagreement."
SCHEMA = {
    "type": "object",
    "required": ["fm_setup_in"],
    "properties": {
        "fm_setup_in": {"type": "string", "description": "The basis to refit (e.g. hyper-search best/fm_setup.in)."},
        "algorithm": {"type": "string", "default": "lassolars"},
        "alpha": {"type": "number", "default": 1e-5},
        "n_models": {"type": "integer", "default": 5},
        "seed": {"type": "integer", "default": 0},
        "candidates_xyzf": {"type": ["string", "null"], "description": "Frames to rank by committee disagreement."},
        "n_select": {"type": "integer", "default": 20, "description": "Most uncertain frames written to uncertain.xyzf."},
    },
}


def add_arguments(parser) -> None:
    parser.add_argument("--fm-setup-in", dest="fm_setup_in", default=None)
    parser.add_argument("--algorithm", default="lassolars")
    parser.add_argument("--alpha", type=float, default=1e-5)
    parser.add_argument("--n-models", dest="n_models", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--candidates-xyzf", dest="candidates_xyzf", default=None)
    parser.add_argument("--n-select", dest="n_select", type=int, default=20)


def bootstrap_weights(groups: np.ndarray, rng) -> np.ndarray:
    """Row weights sqrt(k_f) for frame f drawn k_f times in a bootstrap of the frames."""
    n_frames = int(groups.max()) + 1
    counts = np.bincount(rng.integers(0, n_frames, size=n_frames), minlength=n_frames)
    return np.sqrt(counts[groups].astype(float))


def spread(models_out: list) -> tuple:
    """(force spread, energy spread per atom) per frame from [model][frame] -> (E, F array)."""
    n_frames = len(models_out[0])
    fs_, es = [], []
    for i in range(n_frames):
        F = np.stack([np.asarray(m[i][1]) for m in models_out])        # (models, atoms, 3)
        E = np.array([m[i][0] for m in models_out])
        fs_.append(float(np.sqrt(np.mean(F.std(axis=0) ** 2))))
        es.append(float(E.std() / F.shape[1]))
    return fs_, es


def run(args) -> dict:
    from ._compose import ns
    from . import amat_build, evaluate, solve
    from ._solvers import frame_groups

    if not getattr(args, "fm_setup_in", None):
        raise ValueError("committee needs --fm-setup-in")
    n_models = max(2, int(getattr(args, "n_models", 5) or 5))
    out = Path(getattr(args, "output_dir", None) or ".").resolve()
    fs.ensure_dir(out)
    amat_dir = fs.ensure_dir(out / "amat")
    amat_build.run(ns(fm_setup_in=str(Path(args.fm_setup_in).resolve()), chimes_lsq_bin=None, machine=None, queue="batch",
                      walltime_hours=1.0, nodes=1, ntasks_per_node=None, dry_run=False, timeout_s=None,
                      output_dir=str(amat_dir)))
    groups = frame_groups(amat_dir / "b-labeled.txt", amat_dir / "natoms.txt")
    rng = np.random.default_rng(getattr(args, "seed", 0) or 0)
    members = []
    try:
        for k in range(n_models):
            d = fs.ensure_dir(out / f"member{k}")
            np.savetxt(d / "weights.dat", bootstrap_weights(groups, rng), fmt="%.6g")
            r = solve.run(ns(A=str(amat_dir / "A.txt"), b=str(amat_dir / "b.txt"), header=str(amat_dir / "params.header"),
                             map=str(amat_dir / "ff_groups.map"), dim=str(amat_dir / "dim.txt"), algorithm=args.algorithm,
                             alpha=args.alpha, eps=1e-5, weights=str(d / "weights.dat"), folds=4, normalize=False,
                             split_files=False, machine=None, queue="batch", walltime_hours=1.0, nodes=1,
                             ntasks_per_node=None, poll_interval_s=60, dry_run=False, output_dir=str(d)))
            members.append(r["params"])
    finally:
        (amat_dir / "A.txt").unlink(missing_ok=True)

    result = {"members": members, "n_models": n_models, "algorithm": args.algorithm, "alpha": args.alpha}
    cand = getattr(args, "candidates_xyzf", None)
    if cand:
        frames = xyzf_io.read_xyzf(cand)
        evaluate.check_frames_against_models(frames, members)
        w = evaluate._load_wrapper()
        outs = []
        for p in members:
            ptr = w.chimes_open_instance()
            w.set_chimes_instance(ptr, small=False)
            w.init_chimes_instance(ptr, p, 0)
            cut = evaluate.max_outer_cutoff(p)
            try:
                outs.append([evaluate.predict(w, ptr, f, cut) for f in frames])
            finally:
                w.chimes_close_instance(ptr)
        f_spread, e_spread = spread(outs)
        order = sorted(range(len(frames)), key=lambda i: -f_spread[i])
        n_sel = min(int(getattr(args, "n_select", 20) or 20), len(frames))
        picked = order[:n_sel]
        xyzf_io.write_xyzf([frames[i] for i in picked], out / "uncertain.xyzf")
        result.update({
            "candidates": len(frames),
            "force_spread_kcal_mol_ang": {"median": float(np.median(f_spread)), "max": float(max(f_spread))},
            "energy_spread_kcal_mol_atom": {"median": float(np.median(e_spread)), "max": float(max(e_spread))},
            "uncertain_xyzf": str(out / "uncertain.xyzf"),
            "selected": [{"frame": i, "force_spread": round(f_spread[i], 4), "energy_spread_per_atom": round(e_spread[i], 5)}
                         for i in picked],
        })
    atomic.write_json(out / "committee.json", result, indent=1)
    return result
