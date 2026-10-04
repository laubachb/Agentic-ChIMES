"""End-to-end ChIMES model-building pipeline: unlabeled configs -> QE
labeling -> data-driven cutoff/lambda determination (implementing
documented ChIMES guidance -- see docs/concepts/cutoffs_and_lambdas.md
and stages/hyper_analyze.py) -> one model chosen by `hyper-search` (data-driven cutoffs and
lambdas, noise-aware ties, per-composition guard, alpha and stress stages)
-> optionally, stage the winning fit
as ALC-0 for al_driver and launch it (al-run) to stabilize the model via
further active-learning cycles.

Unlike model-build/sweep (which deliberately stay hands-off on
hyperparameter judgment calls), this pipeline is explicitly asked to
return one answer -- the full hyper_report.json (every stage table and the
reason for each choice) stays alongside the pick.

Blocking/synchronous: runs the whole pipeline top to bottom in one
process, polling the QE Slurm job to completion (hpc.poll_job) rather
than splitting into separate submit/collect calls. Background the whole
`chimes-agent auto-build` invocation yourself (nohup, or your own
background-process mechanism) if you don't want to wait on it
interactively -- see docs/commands/auto-build.md. `--dry-run` only covers
the QE submission call (returns after rendering/submitting-preview,
before polling); it does not preview the rest of the pipeline.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .. import hpc, machines
from ..io import xyzf as xyzf_io
from . import _cutoffs, al_run, dataset_select, qe_relabel
from ._compose import ns
from ..io import fs

NAME = "auto-build"
SUMMARY = "Unlabeled configs -> QE labeling -> hyper-search (cutoffs, lambdas, orders, alpha) -> one model (-> optional AL stabilization)."
SUPPORTS_DRY_RUN = True

DEFAULT_ORDER_GRID = {"2": [10, 12, 14], "3": [5, 7, 9], "4": [None, 2, 3]}

SCHEMA = {
    "type": "object",
    "required": ["elements", "masses"],
    "properties": {
        "unlabeled_xyzf": {"type": ["string", "null"], "description": "Requires pseudopotentials/ecutwfc/machine to label it via QE."},
        "labeled_xyzf": {"type": ["string", "null"], "description": "Skip QE labeling; use this pool directly."},
        "elements": {"type": "array", "items": {"type": "string"}},
        "masses": {"type": "object", "additionalProperties": {"type": "number"}},
        "charges": {"type": ["object", "null"]},
        "pseudopotentials": {"type": ["object", "null"]},
        "ecutwfc": {"type": ["number", "null"], "description": "Ry"},
        "ecutrho": {"type": ["number", "null"]},
        "kpoints": {"type": "array", "default": [1, 1, 1]},
        "smearing": {"type": "string", "default": "gaussian"},
        "degauss": {"type": "number", "default": 0.01},
        "conv_thr": {"type": "number", "default": 1.0e-8},
        "machine": {"type": ["string", "null"]},
        "queue": {"type": "string", "default": "batch"},
        "walltime_hours": {"type": "number", "default": 4.0},
        "nodes": {"type": "integer", "default": 1},
        "ntasks_per_node": {"type": ["integer", "null"]},
        "holdout_xyzf": {"type": ["string", "null"], "description": "If omitted, carved from the labeled pool via dataset-select stratified_holdout."},
        "holdout_fraction": {"type": "number", "default": 0.2},
        "split_seed": {"type": "integer", "default": 42},
        "search_stages": {"type": ["array", "null"], "items": {"type": "string"}, "description": "hyper-search stages to run (default: all of hyper-search's defaults)."},
        "s_minim_delta": {"type": "number", "default": _cutoffs.DEFAULT_S_MINIM_DELTA, "description": "Unused since auto-build delegates to hyper-search (which places inner cutoffs 0.02 A below the closest contact)."},
        "s_maxim_2b_default": {"type": "number", "default": _cutoffs.DOCUMENTED_S_MAXIM_2B_DEFAULT},
        "nlayers": {"type": "integer", "default": 1},
        "order_grid": {"type": "object", "default": DEFAULT_ORDER_GRID, "description": '{"2":[...], "3":[...], "4":[null or int, ...]}'},
        "algorithm": {"type": "string", "default": "lassolars"},
        "alpha": {"type": "number", "default": 1.0e-5, "description": "Documented normalized-DLARS/LASSO default."},
        "fitener": {"type": "string", "default": "false"},
        "fitstrs": {"type": "string", "default": "false"},
        "max_frames": {"type": ["integer", "null"]},
        "stabilize": {
            "type": ["object", "null"],
            "description": '{"alc0_dir", "al_run_work_dir", "config_py", "cycles"} -- see docs/commands/auto-build.md',
        },
    },
}


def add_arguments(parser) -> None:
    parser.add_argument("--unlabeled-xyzf", dest="unlabeled_xyzf", default=None)
    parser.add_argument("--labeled-xyzf", dest="labeled_xyzf", default=None)
    parser.add_argument("--elements", type=lambda s: s.split(","), default=None)
    parser.add_argument("--masses", type=json.loads, default=None)
    parser.add_argument("--charges", type=json.loads, default=None)
    parser.add_argument("--pseudopotentials", type=json.loads, default=None)
    parser.add_argument("--ecutwfc", type=float, default=None)
    parser.add_argument("--ecutrho", type=float, default=None)
    parser.add_argument("--kpoints", type=lambda s: [int(x) for x in s.split(",")], default=[1, 1, 1])
    parser.add_argument("--smearing", default="gaussian")
    parser.add_argument("--degauss", type=float, default=0.01)
    parser.add_argument("--conv-thr", dest="conv_thr", type=float, default=1.0e-8)
    parser.add_argument("--machine", default=None)
    parser.add_argument("--queue", default="batch")
    parser.add_argument("--walltime-hours", dest="walltime_hours", type=float, default=4.0)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--ntasks-per-node", dest="ntasks_per_node", type=int, default=None)
    parser.add_argument("--holdout-xyzf", dest="holdout_xyzf", default=None)
    parser.add_argument("--holdout-fraction", dest="holdout_fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", dest="split_seed", type=int, default=42)
    parser.add_argument("--search-stages", dest="search_stages", type=lambda v: [x for x in v.split(",") if x], default=None)
    parser.add_argument("--s-minim-delta", dest="s_minim_delta", type=float, default=_cutoffs.DEFAULT_S_MINIM_DELTA)
    parser.add_argument("--s-maxim-2b-default", dest="s_maxim_2b_default", type=float, default=_cutoffs.DOCUMENTED_S_MAXIM_2B_DEFAULT)
    parser.add_argument("--nlayers", type=int, default=1)
    parser.add_argument("--order-grid", dest="order_grid", type=json.loads, default=None)
    parser.add_argument("--algorithm", default="lassolars")
    parser.add_argument("--alpha", type=float, default=1.0e-5)
    parser.add_argument("--fitener", default="false")
    parser.add_argument("--fitstrs", default="false")
    parser.add_argument("--max-frames", dest="max_frames", type=int, default=None)
    parser.add_argument("--stabilize", type=json.loads, default=None)


def run(args) -> dict:
    a = vars(args)
    out_dir = Path(a.get("output_dir") or ".")
    fs.ensure_dir(out_dir)

    elements = a.get("elements")
    masses = a.get("masses")
    if not elements or not masses:
        raise ValueError("auto-build requires elements and masses")

    trace = {}

    # ---- Phase 1: labeling ----
    labeled_xyzf = a.get("labeled_xyzf")
    if not labeled_xyzf:
        unlabeled = a.get("unlabeled_xyzf")
        if not unlabeled:
            raise ValueError("auto-build requires labeled_xyzf, or unlabeled_xyzf + QE params to label it")
        for req in ("pseudopotentials", "ecutwfc", "machine"):
            if not a.get(req):
                raise ValueError(f"auto-build requires {req!r} to label unlabeled_xyzf via QE")

        qe_dir = out_dir / "qe"
        submit_result = qe_relabel.run(
            ns(
                structure_xyzf=unlabeled,
                frame_indices=None,
                elements=elements,
                masses=masses,
                pseudopotentials=a["pseudopotentials"],
                ecutwfc=a["ecutwfc"],
                ecutrho=a.get("ecutrho"),
                kpoints=a.get("kpoints") or [1, 1, 1],
                smearing=a.get("smearing", "gaussian"),
                degauss=a.get("degauss", 0.01),
                conv_thr=a.get("conv_thr", 1.0e-8),
                machine=a["machine"],
                queue=a.get("queue", "batch"),
                walltime_hours=a.get("walltime_hours", 4.0),
                nodes=a.get("nodes", 1),
                ntasks_per_node=a.get("ntasks_per_node"),
                collect=None,
                dry_run=bool(a.get("dry_run", False)),
                output_dir=str(qe_dir),
            )
        )
        trace["qe_submit"] = submit_result

        if a.get("dry_run"):
            return {"phase": "qe_submit_dry_run", "trace": trace}

        profile = machines.load_profile(a["machine"])
        handle = hpc.JobHandle(
            job_id=submit_result["job_id"],
            job_name="qe-relabel",
            work_dir=Path(submit_result["work_dir"]),
            job_file=Path(submit_result["job_file"]),
            dry_run=submit_result["dry_run"],
            machine=profile.name,
            queue=a.get("queue", "batch"),
        )
        hpc.poll_job(profile, handle, verbose=True)

        collect_result = qe_relabel.run(ns(collect=str(qe_dir)))
        trace["qe_collect"] = collect_result
        labeled_xyzf = collect_result["labeled_xyzf"]

        if collect_result["n_converged"] == 0:
            raise RuntimeError(f"QE labeling produced zero converged frames -- see trace.qe_collect.report: {collect_result['report']}")

    # ---- Phase 2: holdout split ----
    holdout_xyzf = a.get("holdout_xyzf")
    if not holdout_xyzf:
        split_dir = out_dir / "split"
        split_result = dataset_select.run(
            ns(
                frames=labeled_xyzf,
                method="stratified_holdout",
                n_select=None,
                holdout_fraction=a.get("holdout_fraction", 0.2),
                seed=a.get("split_seed", 42),
                descriptor="composition",
                output_dir=str(split_dir),
            )
        )
        trace["split"] = split_result
        train_xyzf = split_result["selected_xyzf"]
        holdout_xyzf = split_result["holdout_xyzf"]
    else:
        train_xyzf = labeled_xyzf

    # ---- Phases 3-4: hyperparameters and fit, via hyper-search ----
    # (formerly an order sweep at orthorhombic-only derived cutoffs, picking the lowest raw holdout RMSE;
    # hyper-search derives cutoffs/lambdas from the data for any cell, treats statistical ties and per-composition
    # regressions properly, and tunes alpha and stress weights)
    from . import hyper_search

    order_grid = a.get("order_grid") or DEFAULT_ORDER_GRID
    o4 = [v for v in order_grid.get("4", DEFAULT_ORDER_GRID["4"]) if v]
    search_dir = out_dir / "search"
    search = hyper_search.run(ns(
        data_manifest=None, train_xyzf=str(Path(train_xyzf).resolve()), holdout_xyzf=str(Path(holdout_xyzf).resolve()),
        elements=elements, masses=masses, hyper_analysis=None, stages=a.get("search_stages"), four_body="auto" if o4 else "off",
        orders_2b=[v for v in order_grid.get("2", DEFAULT_ORDER_GRID["2"]) if v],
        orders_3b=[v for v in order_grid.get("3", DEFAULT_ORDER_GRID["3"]) if v], orders_4b=o4 or None,
        s_maxim_2b=None, s_maxim_3b=None, s_maxim_4b=None, lambda_scales=None,
        objective="auto", energy_weight=0.1, tolerance=0.03, min_gain=0.05, min_signal=1e-9, exclude_inert=False,
        max_param_ratio=0.5, fitener=(str(a.get("fitener", "false")).lower() == "true") if a.get("fitener") is not None else None,
        fitstrs=a.get("fitstrs"), algorithm=a.get("algorithm", "lassolars"), alpha=a.get("alpha", 1.0e-5),
        workers=a.get("workers", 4), machine=None, max_fit_seconds=a.get("max_fit_seconds", 600), prefer="cheaper",
        output_dir=str(search_dir)))
    trace["search"] = {k: search.get(k) for k in ("hyper_report", "final", "hyperparameters", "notes")}
    chosen = search["final"]
    winning_fm_setup = Path(search["fm_setup_in"])

    result = {
        "trace": trace,
        "train_xyzf": train_xyzf,
        "holdout_xyzf": holdout_xyzf,
        "chosen": chosen,
        "hyperparameters": search["hyperparameters"],
        "params": search["params"],
        "hyper_report": search["hyper_report"],
    }

    # ---- Phase 5: optional AL stabilization ----
    stabilize = a.get("stabilize")
    if stabilize:
        alc0_dir = Path(stabilize["alc0_dir"])
        fs.ensure_dir(alc0_dir)
        shutil.copy(winning_fm_setup, alc0_dir / "fm_setup.in")
        shutil.copy(train_xyzf, alc0_dir / Path(train_xyzf).name)

        al_result = al_run.run(
            ns(
                work_dir=stabilize["al_run_work_dir"],
                config_py=stabilize.get("config_py"),
                cycles=stabilize.get("cycles", [0]),
                python_bin=stabilize.get("python_bin"),
                status_of=None,
                stop=None,
            )
        )
        result["stabilize"] = al_result

    return result
