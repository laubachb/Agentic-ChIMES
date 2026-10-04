"""Learning curve: holdout error of one basis fitted on growing subsets of the
training data. It answers the question every active-learning decision
depends on: is the model data-limited (error still falling) or
basis-limited (plateau), and how much would doubling the data help?

The design matrix is built once. Each subset fit sets row weight 0 for the
frames left out (the same device as cross-validation), so a 5-point curve
costs one A-matrix build and five solves. Subsets are nested and group-aware
(correlated frames enter together, see dataset_select.correlated_groups);
each pair's closest-contact frame is in every subset, since the inner cutoff
is set from it.

The tail of the curve is fitted as err = a * n^(-b) (log-log slope over the
last three points) and extrapolated to 2x and 4x the data. The verdict:

  data-limited  slope <= -0.1 (doubling the data would cut the error by
                more than ~7 %);
  plateau       otherwise: more of the same data will not help much; change
                the basis, the coverage (new conditions) or the labels.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np

from ..io import atomic, fs
from ..io import xyzf as xyzf_io

NAME = "learning-curve"
SUMMARY = "Holdout error vs training-set size for one basis (nested subsets from one design matrix): data-limited or plateau?"
SCHEMA = {
    "type": "object",
    "required": ["fm_setup_in", "holdout_xyzf"],
    "properties": {
        "fm_setup_in": {"type": "string", "description": "The basis (e.g. hyper-search best/fm_setup.in; its TRJFILE is the training set)."},
        "holdout_xyzf": {"type": "string"},
        "fractions": {"type": "array", "items": {"type": "number"}, "default": [0.125, 0.25, 0.5, 0.75, 1.0]},
        "repeats": {"type": "integer", "default": 2, "description": "Independent nested orderings; the curve reports the mean and spread."},
        "algorithm": {"type": "string", "default": "lassolars"},
        "alpha": {"type": "number", "default": 1e-5},
        "weights_preset": {"type": ["string", "null"], "description": "Fitting weights preset (weights stage), if the final model uses one."},
        "stress_weight": {"type": ["number", "null"]},
        "seed": {"type": "integer", "default": 0},
        "plot": {"type": "boolean", "default": True},
    },
}


def add_arguments(parser) -> None:
    parser.add_argument("--fm-setup-in", dest="fm_setup_in", default=None)
    parser.add_argument("--holdout-xyzf", dest="holdout_xyzf", default=None)
    parser.add_argument("--fractions", type=lambda s: [float(x) for x in s.split(",") if x], default=None)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--algorithm", default="lassolars")
    parser.add_argument("--alpha", type=float, default=1e-5)
    parser.add_argument("--weights-preset", dest="weights_preset", default=None)
    parser.add_argument("--stress-weight", dest="stress_weight", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-plot", dest="plot", action="store_false", default=True)


def nested_subsets(frames, fractions, seed: int) -> list:
    """[(fraction, frame indices)] nested, group-aware, closest contacts always included."""
    from .dataset_select import closest_contact_frames, correlated_groups

    groups = correlated_groups(frames)
    protected = {groups[i] for i in closest_contact_frames(frames)}
    others = sorted({g for g in groups if g not in protected})
    random.Random(seed).shuffle(others)
    order = sorted(protected) + others
    members = {}
    for i, g in enumerate(groups):
        members.setdefault(g, []).append(i)
    out = []
    for frac in sorted(fractions):
        n_target = max(len(protected) + 1, round(frac * len(frames)))
        chosen, count = [], 0
        for g in order:
            if count >= n_target and g not in protected:
                break
            chosen += members[g]
            count += len(members[g])
        out.append((frac, sorted(chosen)))
    return out


def fit_tail(ns, errs) -> dict:
    """Power-law tail err = a n^-b over the last three points, with 2x/4x extrapolation."""
    pts = [(n, e) for n, e in zip(ns, errs) if e and e > 0][-3:]
    if len(pts) < 2:
        return {"slope": None}
    x = np.log([p[0] for p in pts])
    y = np.log([p[1] for p in pts])
    b, a = np.polyfit(x, y, 1)
    n_last, e_last = pts[-1]
    return {"slope": round(float(b), 3), "err_at_2x": round(float(e_last * 2 ** b), 4),
            "err_at_4x": round(float(e_last * 4 ** b), 4), "n_last": int(n_last)}


def run(args) -> dict:
    from ._compose import ns
    from . import amat_build, evaluate, solve
    from .weights import row_frames, build as build_weights
    from ..io import fm_setup as fm_io

    if not getattr(args, "fm_setup_in", None) or not getattr(args, "holdout_xyzf", None):
        raise ValueError("learning-curve needs --fm-setup-in and --holdout-xyzf")
    fm_path = Path(args.fm_setup_in).resolve()
    setup = fm_io.parse(fm_path.read_text())
    train_path = Path(str(setup["trjfile"]))
    if not train_path.is_absolute():
        train_path = fm_path.parent / train_path
    frames = xyzf_io.read_xyzf(train_path)[: int(setup.get("nframes") or 0) or None]
    holdout = xyzf_io.read_xyzf(args.holdout_xyzf)
    fractions = sorted(set(getattr(args, "fractions", None) or [0.125, 0.25, 0.5, 0.75, 1.0]))
    repeats = max(1, int(getattr(args, "repeats", 2) or 1))
    out = fs.ensure_dir(Path(getattr(args, "output_dir", None) or ".").resolve())
    amat = fs.ensure_dir(out / "amat")
    amat_build.run(ns(fm_setup_in=str(fm_path), chimes_lsq_bin=None, machine=None, queue="batch", walltime_hours=1.0,
                      nodes=1, ntasks_per_node=None, dry_run=False, timeout_s=None, output_dir=str(amat)))
    tags = [ln.split()[0] for ln in (amat / "b-labeled.txt").read_text().splitlines() if ln.strip()]
    natoms = [float(x) for x in (amat / "natoms.txt").read_text().split()]
    rows_frame = np.asarray(row_frames(tags, natoms))
    base_w = np.ones(len(tags))
    preset, sw = getattr(args, "weights_preset", None), getattr(args, "stress_weight", None)
    if (preset and preset != "uniform") or sw is not None:
        over = {"stress": ["A", [float(sw)]]} if sw is not None else None
        base_w = np.loadtxt(build_weights(amat, preset=preset or "uniform", overrides=over, out_path=out / "weights.base.dat")["weights"])

    curve = {}
    try:
        for rep in range(repeats):
            for frac, idx in nested_subsets(frames, fractions, int(getattr(args, "seed", 0) or 0) + rep):
                keep = np.isin(rows_frame, idx)
                wpath = out / "weights.sub.dat"
                np.savetxt(wpath, np.where(keep, base_w, 0.0), fmt="%.10g")
                d = out / f"fit_{rep}_{frac:g}"
                r = solve.run(ns(A=str(amat / "A.txt"), b=str(amat / "b.txt"), header=str(amat / "params.header"),
                                 map=str(amat / "ff_groups.map"), dim=str(amat / "dim.txt"), algorithm=args.algorithm,
                                 alpha=args.alpha, eps=1e-5, weights=str(wpath), folds=4, normalize=False, split_files=False,
                                 machine=None, queue="batch", walltime_hours=1.0, nodes=1, ntasks_per_node=None,
                                 poll_interval_s=60, dry_run=False, output_dir=str(d)))
                ev = evaluate.evaluate_frames(holdout, [r["params"]])["results"][0]
                curve.setdefault(frac, []).append({"n_frames": len(idx), "relative_force_error": ev["relative_force_error"],
                                                   "rmse_energy_per_atom": ev["rmse_energy_kcal_mol_per_atom"],
                                                   "by_composition": {g: v["relative_force_error"] for g, v in ev["by_composition"].items()}})
                import shutil

                shutil.rmtree(d, ignore_errors=True)
    finally:
        (amat / "A.txt").unlink(missing_ok=True)
        (out / "weights.sub.dat").unlink(missing_ok=True)

    table = []
    for frac in fractions:
        reps = curve.get(frac) or []
        if not reps:
            continue
        fe = [x["relative_force_error"] for x in reps]
        ee = [x["rmse_energy_per_atom"] for x in reps if x["rmse_energy_per_atom"] is not None]
        table.append({"fraction": frac, "n_frames": int(np.mean([x["n_frames"] for x in reps])),
                      "relative_force_error": round(float(np.mean(fe)), 4),
                      "relative_force_error_spread": round(float(np.std(fe)), 4) if len(fe) > 1 else None,
                      "rmse_energy_per_atom": round(float(np.mean(ee)), 4) if ee else None})
    tail = fit_tail([t["n_frames"] for t in table], [t["relative_force_error"] for t in table])
    verdict = ("data-limited" if tail.get("slope") is not None and tail["slope"] <= -0.1 else
               "plateau" if tail.get("slope") is not None else "undetermined")
    notes = []
    if verdict == "data-limited":
        notes.append(f"error still falls with data (log-log slope {tail['slope']}): doubling the training set would give "
                     f"~{tail['err_at_2x']} (from {table[-1]['relative_force_error']}); active learning on the same "
                     "conditions pays")
    elif verdict == "plateau":
        notes.append(f"error has flattened (slope {tail['slope']}): more frames of the same kind will not help much; "
                     "a richer basis, new conditions (coverage) or better labels would")
    if len(table) > 1 and table[0]["relative_force_error"] < table[-1]["relative_force_error"]:
        notes.append("error at the smallest subset is lower than at the full set: the holdout is closer to a subset of "
                     "the data than to all of it, or the fit is unstable at small sizes; look at the spread")
    result = {"basis": str(fm_path), "n_train_total": len(frames), "n_holdout": len(holdout), "repeats": repeats,
              "curve": table, "tail_fit": tail, "verdict": verdict, "notes": notes}
    if getattr(args, "plot", True):
        from ..io import plots

        png = plots.lines({"holdout": ([t["n_frames"] for t in table], [t["relative_force_error"] for t in table])},
                          out / "learning_curve.png", xlabel="training frames", ylabel="relative force error",
                          title=f"Learning curve ({verdict})", logx=True, logy=True)
        if png:
            result["plot"] = png
    atomic.write_json(out / "learning_curve.json", result, indent=1)
    return result
