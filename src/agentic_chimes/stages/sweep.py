"""Grid sweep over ChIMES hyperparameters, composing fm-setup-gen +
model-build (amat-build+solve) + evaluate for each grid point and reporting
a comparison table. **Not an auto-tuner** -- consistent with this repo's
design stance that hyperparameter choices stay a human/agent judgment
call: this produces a table (JSON + CSV) and a `best_by` pointer per
metric to look at, it does not pick a "best" model as a final answer.

Sweepable dimensions (any subset; anything not given uses `base`'s value):
  - order_2b, order_3b, order_4b (independently -- order_4b entries of
    null/0 mean "no 4-body term" for that grid point, matching
    fm_setup_gen's own convention for omitting PAIRTYP's 4-body order)
  - default_s_minim, default_s_maxim (the outer/inner cutoff fm-setup-gen
    falls back to for any pair not given an explicit pair_cutoffs entry)
  - alpha (solve's regularization strength)
  - algorithm (solve's algorithm choice)

Grid points run sequentially (no parallelism in this phase), each into its
own `point_NNNN/` subdirectory under --output-dir, calling
fm_setup_gen/model_build/evaluate's `run()` functions directly (see
stages/_compose.py). A failed grid point is recorded with
`"status": "failed"` and does not stop the rest of the sweep.
"""

from __future__ import annotations

import csv
import itertools
import json
import time
from pathlib import Path

from . import evaluate, fm_setup_gen, model_build
from ._compose import ns

NAME = "sweep"
SUMMARY = "Grid sweep over 2b/3b/4b order, cutoffs, alpha/algorithm; reports a comparison table (not auto-tuning)."

GRID_KEYS = ["order_2b", "order_3b", "order_4b", "default_s_minim", "default_s_maxim", "alpha", "algorithm"]

SCHEMA = {
    "type": "object",
    "required": ["base", "grid", "holdout_xyzf"],
    "properties": {
        "base": {
            "type": "object",
            "description": "Shared fm-setup-gen + solve parameters (trjfile, nframes, elements, masses, charges, pair_cutoffs, morse_lambda, default_s_minim, default_s_maxim, default_morse_lambda, s_delta, wraptrj, nlayers, fitcoul, fitstrs, fitener, fitpovr, chbtype, fcuttyp, exclude_3b, exclude_4b, special_maxim_3b, special_maxim_4b, special_blocks, algorithm, alpha, eps, weights, folds, max_frames) -- see fm-setup-gen/solve/evaluate --describe.",
        },
        "grid": {
            "type": "object",
            "description": "Any subset of: order_2b, order_3b, order_4b, default_s_minim, default_s_maxim, alpha, algorithm -- each a list of values to sweep. Cartesian product across all given keys.",
        },
        "holdout_xyzf": {"type": "string"},
    },
}


def add_arguments(parser) -> None:
    parser.add_argument("--base", type=json.loads, default=None, help="JSON object -- see --describe.")
    parser.add_argument("--grid", type=json.loads, default=None, help="JSON object -- see --describe.")
    parser.add_argument("--holdout-xyzf", dest="holdout_xyzf", default=None)


def _grid_points(grid: dict):
    keys = [k for k in GRID_KEYS if grid.get(k)]
    if not keys:
        yield {}
        return
    for combo in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, combo))


def _resolve_order(base: dict, overrides: dict) -> dict:
    base_order = dict(base.get("order") or {})
    order = {
        "2": overrides.get("order_2b", base_order.get("2", base.get("order_2b", 12))),
        "3": overrides.get("order_3b", base_order.get("3", base.get("order_3b", 5))),
    }
    order4 = overrides.get("order_4b", base_order.get("4", base.get("order_4b")))
    if order4:
        order["4"] = order4
    return order


def _run_one_point(base: dict, overrides: dict, holdout_xyzf: str, point_dir: Path) -> dict:
    fm_args = ns(
        trjfile=base.get("trjfile"),
        nframes=base.get("nframes"),
        elements=base.get("elements"),
        order=_resolve_order(base, overrides),
        masses=base.get("masses"),
        charges=base.get("charges"),
        pair_cutoffs=base.get("pair_cutoffs"),
        morse_lambda=base.get("morse_lambda"),
        default_s_minim=overrides.get("default_s_minim", base.get("default_s_minim", 1.0)),
        default_s_maxim=overrides.get("default_s_maxim", base.get("default_s_maxim", 6.0)),
        default_morse_lambda=base.get("default_morse_lambda", 1.5),
        s_delta=base.get("s_delta", 0.01),
        wraptrj=base.get("wraptrj", True),
        nlayers=base.get("nlayers", 1),
        fitcoul=base.get("fitcoul", False),
        fitstrs=base.get("fitstrs", "false"),
        fitener=base.get("fitener", "false"),
        fitpovr=base.get("fitpovr", False),
        chbtype=base.get("chbtype", "MORSE"),
        fcuttyp=base.get("fcuttyp", "CUBIC"),
        exclude_3b=base.get("exclude_3b"),
        exclude_4b=base.get("exclude_4b"),
        special_maxim_3b=base.get("special_maxim_3b"),
        special_maxim_4b=base.get("special_maxim_4b"),
        special_blocks=base.get("special_blocks"),
        cheby_range=base.get("cheby_range", [-1, 1]),
        output_dir=str(point_dir),
    )
    fm_result = fm_setup_gen.run(fm_args)

    mb_args = ns(
        fm_setup_in=fm_result["fm_setup_in"],
        chimes_lsq_bin=base.get("chimes_lsq_bin"),
        algorithm=overrides.get("algorithm", base.get("algorithm", "svd")),
        alpha=overrides.get("alpha", base.get("alpha", 1.0e-4)),
        eps=base.get("eps", 1.0e-5),
        weights=base.get("weights"),
        folds=base.get("folds", 4),
        normalize=base.get("normalize", False),
        machine=base.get("machine"),
        queue=base.get("queue", "batch"),
        walltime_hours=base.get("walltime_hours", 2.0),
        nodes=base.get("nodes", 1),
        ntasks_per_node=base.get("ntasks_per_node"),
        poll_interval_s=base.get("poll_interval_s", 60),
        output_dir=str(point_dir),
    )
    mb_result = model_build.run(mb_args)

    ev_args = ns(params=[mb_result["params"]], holdout_xyzf=holdout_xyzf, max_frames=base.get("max_frames"))
    ev_result = evaluate.run(ev_args)

    return {
        "params": mb_result["params"],
        "rmse_force_kcal_mol_ang": ev_result["results"][0]["rmse_force_kcal_mol_ang"],
        "rmse_energy_kcal_mol": ev_result["results"][0]["rmse_energy_kcal_mol"],
    }


def _write_csv(path: Path, results: list) -> None:
    fieldnames = ["index", "status"] + GRID_KEYS + ["rmse_force_kcal_mol_ang", "rmse_energy_kcal_mol", "wall_time_s", "params", "error"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = {"index": r["index"], "status": r["status"], **r.get("overrides", {})}
            row.update({k: r.get(k) for k in ("rmse_force_kcal_mol_ang", "rmse_energy_kcal_mol", "wall_time_s", "params", "error")})
            writer.writerow(row)


def run(args) -> dict:
    base = getattr(args, "base", None) or {}
    grid = getattr(args, "grid", None) or {}
    holdout_xyzf = getattr(args, "holdout_xyzf", None)

    if not base:
        raise ValueError("sweep requires 'base' (fm-setup-gen + solve parameters; see --describe)")
    if not holdout_xyzf:
        raise ValueError("sweep requires 'holdout_xyzf'")

    unknown = [k for k in grid if k not in GRID_KEYS]
    if unknown:
        raise ValueError(f"unknown grid key(s) {unknown}; sweepable: {GRID_KEYS}")

    out_root = Path(getattr(args, "output_dir", None) or ".")
    out_root.mkdir(parents=True, exist_ok=True)

    points = list(_grid_points(grid))
    results = []

    for i, overrides in enumerate(points):
        point_dir = out_root / f"point_{i:04d}"
        t0 = time.time()
        row = {"index": i, "overrides": overrides}
        try:
            row.update(_run_one_point(base, overrides, holdout_xyzf, point_dir))
            row["status"] = "done"
        except Exception as exc:  # noqa: BLE001 - one bad grid point must not kill the sweep
            row["status"] = "failed"
            row["error"] = str(exc)
        row["wall_time_s"] = time.time() - t0
        results.append(row)

    csv_path = out_root / "sweep_results.csv"
    _write_csv(csv_path, results)

    best_by = {}
    done = [r for r in results if r["status"] == "done"]
    if done:
        best_by["rmse_force"] = min(done, key=lambda r: r["rmse_force_kcal_mol_ang"])["index"]
        with_energy = [r for r in done if r.get("rmse_energy_kcal_mol") is not None]
        if with_energy:
            best_by["rmse_energy"] = min(with_energy, key=lambda r: r["rmse_energy_kcal_mol"])["index"]

    return {"n_points": len(points), "n_done": len(done), "n_failed": len(points) - len(done), "results": results, "table_csv": str(csv_path), "best_by": best_by}
