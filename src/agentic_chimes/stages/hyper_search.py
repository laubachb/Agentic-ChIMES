"""Staged hyperparameter search for a ChIMES model: cutoffs, Morse lambdas
and polynomial orders, chosen on holdout error with a preference for the
cheapest model that is nearly as good.

Stages (each a small grid; later stages hold earlier choices fixed):
  2b      order_2b x s_maxim_2b, no many-body terms
  3b      order_3b x s_maxim_3b, 2-body fixed; kept only if it beats the
          2-body model by more than `tolerance`
  4b      order_4b x s_maxim_4b, 2+3-body fixed; `four_body`: off | on |
          auto (run only if 3-body improved the score by > min_gain)
  lambda  one scale factor on every pair's Morse lambda
  refine  order_2b +/- 2 around the choice, many-body terms fixed
          (coordinate search can leave 2b slightly off after 3b is added)

Inner cutoffs and base lambdas come from `hyper-analyze` (data-driven, not
searched: an inner cutoff above the smallest sampled distance throws data
away, below it extrapolates). Outer-cutoff candidates come from the RDF
shells there too. N_LAYERS is set per fit from the cutoff and the thinnest
cell.

Selection in every stage: best score, then the cheapest point (fewest
coefficients, then shortest cutoffs) within `tolerance` of it. Points with
more than `max_param_ratio` coefficients per equation are reported but never
chosen. The score is the holdout force error relative to the holdout
reference force RMS, plus `energy_weight` x per-atom energy RMSE
(kcal/mol/atom) when `objective` is `force+energy`.

Every fit is cached by configuration under `points/`, so re-running resumes.
`workers` fits run in parallel. With `--machine`, the search is submitted as
one Slurm job (dry-run first) instead of running here.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ..io import xyzf as xyzf_io
from . import _hyper, hyper_analyze

NAME = "hyper-search"
SUMMARY = "Staged search over cutoffs, Morse lambdas and 2b/3b/4b orders; picks the cheapest near-best model on holdout error."

DEFAULTS = {
    "orders_2b": [8, 10, 12, 14, 16, 18],
    "orders_3b": [4, 6, 8, 10],
    "orders_4b": [2, 3, 4],
    "lambda_scales": [0.9, 1.0, 1.1],
    "stages": ["2b", "3b", "4b", "lambda", "refine"],
}

SCHEMA = {
    "type": "object",
    "properties": {
        "data_manifest": {"type": ["string", "null"], "description": "From data-curate: train/holdout paths, elements, fitener."},
        "train_xyzf": {"type": ["string", "null"]},
        "holdout_xyzf": {"type": ["string", "null"]},
        "elements": {"type": ["array", "null"], "items": {"type": "string"}},
        "hyper_analysis": {"type": ["string", "null"], "description": "hyper_analysis.json; computed if omitted."},
        "stages": {"type": "array", "items": {"type": "string"}, "default": DEFAULTS["stages"]},
        "four_body": {"type": "string", "enum": ["off", "on", "auto"], "default": "auto"},
        "orders_2b": {"type": "array", "items": {"type": "integer"}, "default": DEFAULTS["orders_2b"]},
        "orders_3b": {"type": "array", "items": {"type": "integer"}, "default": DEFAULTS["orders_3b"]},
        "orders_4b": {"type": "array", "items": {"type": "integer"}, "default": DEFAULTS["orders_4b"]},
        "s_maxim_2b": {"type": ["array", "null"], "items": {"type": "number"}, "description": "Default: hyper-analyze candidates."},
        "s_maxim_3b": {"type": ["array", "null"], "items": {"type": "number"}},
        "s_maxim_4b": {"type": ["array", "null"], "items": {"type": "number"}},
        "lambda_scales": {"type": "array", "items": {"type": "number"}, "default": DEFAULTS["lambda_scales"]},
        "objective": {"type": "string", "enum": ["auto", "force", "force+energy"], "default": "auto", "description": "auto = force+energy when energies are fitted, else force."},
        "energy_weight": {"type": "number", "default": 0.1, "description": "Score per kcal/mol/atom of energy RMSE, for objective force+energy."},
        "tolerance": {"type": "number", "default": 0.03, "description": "Accept a cheaper model scoring within this fraction of the best, or within one bootstrap standard error of it if that is larger."},
        "min_signal": {"type": "number", "default": 1.0e-6, "description": "3-/4-body columns whose median norm is below this fraction of the 2-body median are numerically inert; such points are never chosen."},
        "min_gain": {"type": "number", "default": 0.05, "description": "four_body=auto runs 4b only if 3b improved the score by at least this fraction."},
        "max_param_ratio": {"type": "number", "default": 0.5, "description": "Coefficients per equation above which a point is never chosen."},
        "fitener": {"type": ["boolean", "null"], "description": "Default: data_manifest fit_hints.fitener."},
        "algorithm": {"type": "string", "default": "lassolars"},
        "alpha": {"type": "number", "default": 1.0e-5},
        "masses": {"type": ["object", "null"]},
        "workers": {"type": "integer", "default": 4},
        "machine": {"type": ["string", "null"], "description": "Submit the whole search as one Slurm job on this machine."},
        "queue": {"type": "string", "default": "batch"},
        "walltime_hours": {"type": "number", "default": 4.0},
    },
}


def _ints(s):
    return [int(x) for x in s.split(",") if x]


def _floats(s):
    return [float(x) for x in s.split(",") if x]


def add_arguments(parser) -> None:
    parser.add_argument("--data-manifest", dest="data_manifest", default=None)
    parser.add_argument("--train-xyzf", dest="train_xyzf", default=None)
    parser.add_argument("--holdout-xyzf", dest="holdout_xyzf", default=None)
    parser.add_argument("--elements", type=lambda s: [x for x in s.split(",") if x], default=None)
    parser.add_argument("--hyper-analysis", dest="hyper_analysis", default=None)
    parser.add_argument("--stages", type=lambda s: [x for x in s.split(",") if x], default=None)
    parser.add_argument("--four-body", dest="four_body", choices=["off", "on", "auto"], default="auto")
    parser.add_argument("--orders-2b", dest="orders_2b", type=_ints, default=None)
    parser.add_argument("--orders-3b", dest="orders_3b", type=_ints, default=None)
    parser.add_argument("--orders-4b", dest="orders_4b", type=_ints, default=None)
    parser.add_argument("--s-maxim-2b", dest="s_maxim_2b", type=_floats, default=None)
    parser.add_argument("--s-maxim-3b", dest="s_maxim_3b", type=_floats, default=None)
    parser.add_argument("--s-maxim-4b", dest="s_maxim_4b", type=_floats, default=None)
    parser.add_argument("--lambda-scales", dest="lambda_scales", type=_floats, default=None)
    parser.add_argument("--objective", choices=["auto", "force", "force+energy"], default="auto")
    parser.add_argument("--energy-weight", dest="energy_weight", type=float, default=0.1)
    parser.add_argument("--tolerance", type=float, default=0.03)
    parser.add_argument("--min-gain", dest="min_gain", type=float, default=0.05)
    parser.add_argument("--min-signal", dest="min_signal", type=float, default=1.0e-6)
    parser.add_argument("--max-param-ratio", dest="max_param_ratio", type=float, default=0.5)
    parser.add_argument("--fitener", type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
    parser.add_argument("--algorithm", default="lassolars")
    parser.add_argument("--alpha", type=float, default=1.0e-5)
    parser.add_argument("--masses", type=json.loads, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--machine", default=None)
    parser.add_argument("--queue", default="batch")
    parser.add_argument("--walltime-hours", dest="walltime_hours", type=float, default=4.0)


def _get(args, key):
    v = getattr(args, key, None)
    return DEFAULTS.get(key) if v is None else v


def _masses(elements, given):
    from ase.data import atomic_masses, atomic_numbers

    given = given or {}
    return {e: float(given.get(e, round(atomic_masses[atomic_numbers[e]], 4))) for e in elements}


def _submit(args, out: Path) -> dict:
    from .. import hpc, machines

    payload = {k: v for k, v in vars(args).items()
               if not k.startswith("_") and k not in ("machine", "json_in", "json_out", "describe", "force", "dry_run",
                                                      "output_dir", "stage", "queue", "walltime_hours")}
    profile = machines.load_profile(args.machine)
    payload["workers"] = profile.default_ntasks_per_node or payload.get("workers", 4)
    inp = out / "hyper_search_input.json"
    inp.write_text(json.dumps(payload, indent=1, default=str))
    run_dir = out / "search"
    cmd = f"{sys.executable} -m agentic_chimes.cli hyper-search --json-in {inp} --output-dir {run_dir} > hyper_search.out 2>&1"
    handle = hpc.submit_job(profile, job_name="hyper-search", commands=[f"cd {out.resolve()}", cmd], work_dir=out,
                            nodes=1, walltime_hours=getattr(args, "walltime_hours", 4.0) or 4.0,
                            queue=getattr(args, "queue", "batch") or "batch", dry_run=bool(getattr(args, "dry_run", False)))
    return {"submitted": not handle.dry_run, "dry_run": handle.dry_run, "job_id": handle.job_id,
            "job_file": str(handle.job_file), "input": str(inp),
            "results_when_done": str(run_dir / "hyper_report.json"),
            "note": "The search runs inside the job; read hyper_report.json when it finishes."}


class _Runner:
    def __init__(self, out, base_task, workers):
        self.out, self.base, self.workers, self.all = out, base_task, workers, {}

    def fit(self, cfgs):
        tasks = []
        for cfg in cfgs:
            key = _hyper.config_key(cfg)
            tasks.append({**self.base, "cfg": cfg, "point_dir": str(self.out / "points" / key)})
        if self.workers > 1 and len(tasks) > 1:
            with ProcessPoolExecutor(max_workers=min(self.workers, len(tasks))) as pool:
                results = list(pool.map(_hyper.run_point, tasks))
        else:
            results = [_hyper.run_point(t) for t in tasks]
        for r in results:
            self.all[r["key"]] = r
        return results


def _row(r):
    c = r["cfg"]
    keep = {k: c.get(k) for k in ("order_2b", "order_3b", "order_4b", "s_maxim_2b", "s_maxim_3b", "s_maxim_4b", "lambda_scale")}
    keep.update({k: r.get(k) for k in ("status", "n_params", "params_per_equation", "nlayers", "holdout_relative_force_error",
                                        "holdout_relative_force_se", "holdout_rmse_force", "holdout_rmse_energy_per_atom",
                                        "train_relative_force_error", "score", "key", "error")})
    sig = r.get("signal") or {}
    for body in ("3b", "4b"):
        if sig.get(f"signal_{body}") is not None:
            keep[f"signal_{body}"] = float(f"{sig[f'signal_{body}']:.2g}")
    return keep


def _edge_notes(stage, chosen, grid_key, values):
    if chosen is None or not values or len(values) < 2:
        return []
    v = chosen["cfg"].get(grid_key)
    if v == max(values):
        return [f"{stage}: chose the largest {grid_key} ({v}); the optimum may lie beyond the grid"]
    return []


def run(args) -> dict:
    out = Path(getattr(args, "output_dir", None) or ".").resolve()
    out.mkdir(parents=True, exist_ok=True)
    if getattr(args, "machine", None):
        return _submit(args, out)

    train, elements, manifest = hyper_analyze.resolve_inputs(args)
    holdout = getattr(args, "holdout_xyzf", None) or manifest.get("holdout_xyzf")
    if not holdout:
        raise ValueError("hyper-search needs a holdout set (data_manifest with a split, or --holdout-xyzf)")

    if getattr(args, "hyper_analysis", None):
        analysis = json.loads(Path(args.hyper_analysis).read_text())
    else:
        from ._compose import ns

        analysis = hyper_analyze.run(ns(data_manifest=None, train_xyzf=train, elements=elements, r_max=8.0,
                                        s_minim_delta=0.02, output_dir=str(out)))
    unsampled = [p for p, v in analysis["pairs"].items() if not v.get("sampled")]
    if unsampled:
        raise ValueError(f"pairs {unsampled} have no data within the analysis radius; fix the dataset first")

    fitener = getattr(args, "fitener", None)
    if fitener is None:
        fitener = bool((manifest.get("fit_hints") or {}).get("fitener", analysis.get("n_energy_equations", 0) > 0))
    objective = getattr(args, "objective", "auto") or "auto"
    if objective == "auto":
        objective = "force+energy" if fitener else "force"
    min_signal = getattr(args, "min_signal", 1.0e-6)
    energy_weight = getattr(args, "energy_weight", 0.1)
    tol = getattr(args, "tolerance", 0.03)
    min_gain = getattr(args, "min_gain", 0.05)
    max_ratio = getattr(args, "max_param_ratio", 0.5)
    cands = analysis["candidates"]
    grid = {
        "orders_2b": _get(args, "orders_2b"), "orders_3b": _get(args, "orders_3b"), "orders_4b": _get(args, "orders_4b"),
        "s_maxim_2b": getattr(args, "s_maxim_2b", None) or cands["s_maxim_2b"],
        "s_maxim_3b": getattr(args, "s_maxim_3b", None) or cands["s_maxim_3b"],
        "s_maxim_4b": getattr(args, "s_maxim_4b", None) or cands["s_maxim_4b"],
        "lambda_scales": _get(args, "lambda_scales"),
    }
    stages = _get(args, "stages")
    four_body = getattr(args, "four_body", "auto") or "auto"

    base_cfg = {
        "elements": elements,
        "s_minim": {p: v["suggested"]["s_minim"] for p, v in analysis["pairs"].items()},
        "morse_lambda": {p: v["suggested"]["morse_lambda"] for p, v in analysis["pairs"].items()},
        "fitener": fitener, "lambda_scale": 1.0,
        "order_3b": 0, "s_maxim_3b": None, "order_4b": 0, "s_maxim_4b": None,
    }
    n_train = len(xyzf_io.read_xyzf(train))
    runner = _Runner(out, {
        "train_xyzf": str(Path(train).resolve()), "holdout_xyzf": str(Path(holdout).resolve()), "n_train": n_train,
        "masses": _masses(elements, getattr(args, "masses", None)), "thinnest": analysis["thinnest_cell_width"],
        "algorithm": getattr(args, "algorithm", "lassolars") or "lassolars", "alpha": getattr(args, "alpha", 1e-5),
    }, max(1, getattr(args, "workers", 4) or 1))

    inert = []

    def usable(r):
        sig = r.get("signal") or {}
        for body in ("3b", "4b"):
            if r["cfg"].get(f"order_{body}") and sig.get(f"signal_{body}") is not None and sig[f"signal_{body}"] < min_signal:
                inert.append((body, r["cfg"].get(f"s_maxim_{body}"), sig[f"signal_{body}"]))
                return False
        return True

    def pick(results):
        results = [r for r in results if r.get("status") != "done" or usable(r)]
        return _hyper.select(results, objective=objective, energy_weight=energy_weight, tolerance=tol, max_param_ratio=max_ratio)

    report_stages, notes = [], []
    current = None

    def record(name, results, sel, extra=None):
        entry = {"stage": name, "n_points": len(results), "n_failed": sum(r["status"] != "done" for r in results),
                 "chosen": _row(sel["chosen"]) if sel["chosen"] else None, "best": _row(sel["best"]) if sel["best"] else None,
                 "reason": sel["reason"], "table": [_row(r) for r in sorted(results, key=lambda r: r.get("score", 9e9))]}
        if extra:
            entry.update(extra)
        report_stages.append(entry)

    # ---- 2-body
    if "2b" in stages or current is None:
        cfgs = [{**base_cfg, "order_2b": o, "s_maxim_2b": c} for o in grid["orders_2b"] for c in grid["s_maxim_2b"]]
        res = runner.fit(cfgs)
        sel = pick(res)
        record("2b", res, sel)
        if not sel["chosen"]:
            raise RuntimeError(f"2-body stage produced no usable fit: {[r.get('error') for r in res][:3]}")
        current = sel["chosen"]
        notes += _edge_notes("2b", current, "order_2b", grid["orders_2b"])
        notes += _edge_notes("2b", current, "s_maxim_2b", grid["s_maxim_2b"])
    score_2b = current["score"]

    # ---- 3-body
    if "3b" in stages:
        c2 = current["cfg"]
        cfgs = [{**c2, "order_3b": o, "s_maxim_3b": c} for o in grid["orders_3b"] for c in grid["s_maxim_3b"] if c <= c2["s_maxim_2b"]]
        res = runner.fit(cfgs)
        sel = pick(res + [current])
        chosen = sel["chosen"]
        kept = chosen is not None and chosen["cfg"].get("order_3b")
        record("3b", res, sel, {"kept_three_body": bool(kept)})
        if kept:
            current = chosen
            notes += _edge_notes("3b", current, "order_3b", grid["orders_3b"])
            notes += _edge_notes("3b", current, "s_maxim_3b", [c for c in grid["s_maxim_3b"] if c <= current["cfg"]["s_maxim_2b"]])
        else:
            notes.append("3b: no 3-body model beat the 2-body model by more than the tolerance; kept 2-body only")

    # ---- 4-body
    gain_3b = (score_2b - current["score"]) / score_2b if score_2b else 0.0
    run_4b = "4b" in stages and current["cfg"].get("order_3b") and (
        four_body == "on" or (four_body == "auto" and gain_3b >= min_gain))
    if "4b" in stages and not run_4b:
        report_stages.append({"stage": "4b", "skipped": True,
                              "reason": ("four_body=off" if four_body == "off" else "no 3-body terms" if not current["cfg"].get("order_3b")
                                         else f"3-body improved the score by {gain_3b:.1%} < min_gain {min_gain:.0%}")})
    if run_4b:
        c3 = current["cfg"]
        cfgs = [{**c3, "order_4b": o, "s_maxim_4b": c} for o in grid["orders_4b"] for c in grid["s_maxim_4b"] if c <= c3["s_maxim_3b"]]
        res = runner.fit(cfgs)
        sel = pick(res + [current])
        record("4b", res, sel, {"kept_four_body": bool(sel["chosen"] and sel["chosen"]["cfg"].get("order_4b"))})
        if sel["chosen"] and sel["chosen"]["cfg"].get("order_4b"):
            current = sel["chosen"]
        else:
            notes.append("4b: no 4-body model beat the current model by more than the tolerance; no 4-body terms")

    # ---- Morse lambda
    if "lambda" in stages:
        cfgs = [{**current["cfg"], "lambda_scale": s} for s in grid["lambda_scales"]]
        res = runner.fit(cfgs)
        done = [r for r in res if r["status"] == "done"]
        for r in done:
            r["score"] = _hyper.score(r, objective, energy_weight)
            r["params_per_equation"] = round(r["n_params"] / r["n_equations"], 3)
        best = min(done, key=lambda r: r["score"]) if done else None
        if best and best["score"] < current["score"] * (1 - tol):
            current = best
            reason = f"lambda scale {best['cfg']['lambda_scale']} improved the score by more than {tol:.0%}"
        else:
            reason = "no lambda scale beat 1.0 by more than the tolerance; kept first-RDF-peak lambdas"
        report_stages.append({"stage": "lambda", "n_points": len(res), "reason": reason,
                              "table": [_row(r) for r in sorted(done, key=lambda r: r["score"])]})

    # ---- refine 2-body order with many-body terms in place
    if "refine" in stages and (current["cfg"].get("order_3b") or current["cfg"].get("order_4b")):
        o = current["cfg"]["order_2b"]
        cfgs = [{**current["cfg"], "order_2b": v} for v in sorted({o - 2, o, o + 2}) if v >= 4]
        res = runner.fit(cfgs)
        sel = pick(res)
        record("refine", res, sel)
        if sel["chosen"]:
            current = sel["chosen"]
            if current["cfg"]["order_2b"] < min(grid["orders_2b"]):
                notes.append(f"refine moved order_2b below the 2b grid ({current['cfg']['order_2b']}); lower orders may be worth trying")
            elif current["cfg"]["order_2b"] > max(grid["orders_2b"]):
                notes.append(f"refine moved order_2b above the 2b grid ({current['cfg']['order_2b']}); higher orders may be worth trying")

    if inert:
        seen = sorted({(b, c) for b, c, _ in inert})
        notes.append("numerically inert many-body terms (columns < min_signal of the 2-body scale) at "
                     + ", ".join(f"{b} cutoff {c}" for b, c in seen)
                     + ": ChIMES' cubic smoothing multiplies one (1 - r/r_c)^3 factor per cluster distance, so these cutoffs "
                       "barely exceed the neighbour distances. Such points were excluded; try longer many-body cutoffs")

    # ---- final artifacts
    final = current
    best_dir = out / "best"
    best_dir.mkdir(exist_ok=True)
    shutil.copy(final["params"], best_dir / "params.txt")
    shutil.copy(final["fm_setup_in"], best_dir / "fm_setup.in")
    c = final["cfg"]
    choice = {
        "elements": elements,
        "order": {"2": c["order_2b"], "3": c.get("order_3b") or 0, **({"4": c["order_4b"]} if c.get("order_4b") else {})},
        "pair_cutoffs": {p: [c["s_minim"][p], c["s_maxim_2b"]] for p in c["s_minim"]},
        "morse_lambda": {p: round(c["morse_lambda"][p] * c.get("lambda_scale", 1.0), 4) for p in c["morse_lambda"]},
        "special_maxim_3b": c.get("s_maxim_3b") if c.get("order_3b") else None,
        "special_maxim_4b": c.get("s_maxim_4b") if c.get("order_4b") else None,
        "nlayers": final["nlayers"],
        "fitener": c["fitener"],
        "algorithm": runner.base["algorithm"], "alpha": runner.base["alpha"],
        "weights": "default (uniform)",
    }
    (best_dir / "hyper_choice.json").write_text(json.dumps(choice, indent=1))

    gap = final["holdout_relative_force_error"] - final["train_relative_force_error"]
    if final["train_relative_force_error"] and gap > 0.5 * final["train_relative_force_error"]:
        notes.append(f"holdout error exceeds training error by {gap:.3f} (relative): possible overfitting or train/holdout mismatch")
    if final["holdout_relative_force_error"] > 0.3:
        notes.append(f"final relative force error {final['holdout_relative_force_error']:.2f} is high: the data (coverage, size, "
                     "consistency) is more likely the limit than these hyperparameters")
    if analysis["n_frames"] < 200:
        notes.append(f"only {analysis['n_frames']} training frames and a small holdout: differences of a few percent between "
                     "points are within noise, which is why the tolerance favours smaller models")

    report = {
        "final": _row(final),
        "hyperparameters": choice,
        "best_dir": str(best_dir),
        "stages": report_stages,
        "notes": notes + analysis.get("notes", []),
        "settings": {"objective": objective, "energy_weight": energy_weight, "tolerance": tol, "min_gain": min_gain, "min_signal": min_signal,
                     "max_param_ratio": max_ratio, "grid": grid, "stages": stages, "four_body": four_body,
                     "train_xyzf": str(train), "holdout_xyzf": str(holdout)},
        "n_fits": len(runner.all),
    }
    path = out / "hyper_report.json"
    path.write_text(json.dumps(report, indent=1, default=str))
    return {"hyper_report": str(path), "params": str(best_dir / "params.txt"), "fm_setup_in": str(best_dir / "fm_setup.in"),
            "hyper_choice": str(best_dir / "hyper_choice.json"), "final": report["final"], "hyperparameters": choice,
            "stage_summary": [{k: s[k] for k in ("stage", "n_points", "skipped", "reason", "kept_three_body", "kept_four_body") if k in s} for s in report_stages],
            "notes": report["notes"], "n_fits": report["n_fits"]}
