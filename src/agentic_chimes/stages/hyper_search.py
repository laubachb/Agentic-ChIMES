"""Staged hyperparameter search for a ChIMES model: cutoffs, Morse lambdas
and polynomial orders, chosen on holdout error with a preference for the
cheapest model that is nearly as good.

Stages (each a small grid; later stages hold earlier choices fixed):
  2b      order_2b x s_maxim_2b, no many-body terms
  3b      order_3b x s_maxim_3b, 2-body fixed; kept only if it beats the
          2-body model by more than `tolerance`
  4b      order_4b x s_maxim_4b, 2+3-body fixed; `four_body`: off | on |
          auto (run only if 3-body improved the score by > min_gain)
  exclude leave-one-type-out over the model's 3-/4-body cluster types (EXCLUDE
          blocks), greedily dropping types whose removal leaves the fit
          statistically tied; reports each type's data coverage and the
          score change when it is removed
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
from ..io.pool import process_pool
from pathlib import Path

from ..io import xyzf as xyzf_io
from . import _hyper, hyper_analyze
from ..io import atomic
from ..io import fs

NAME = "hyper-search"
SUMMARY = "Staged search over cutoffs, Morse lambdas and 2b/3b/4b orders; picks the cheapest near-best model on holdout error."
SUPPORTS_DRY_RUN = True

DEFAULTS = {
    "orders_2b": [8, 10, 12, 14, 16, 18],
    "orders_3b": [4, 6, 8, 10],
    "orders_4b": [2, 3],  # 4-body builds scale steeply; order 4 is ~3,000+ coefficients for a binary
    "lambda_scales": [0.9, 1.0, 1.1],
    "stages": ["2b", "3b", "3b_pairs", "4b", "smoothing", "exclude", "lambda", "refine", "alpha", "stress"],
    "smoothings": ["TERSOFF 0.5", "TERSOFF 0.75"],
    "alphas": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3],
    "stress_weights": [1.0, 3.0, 10.0, 30.0],
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
        "four_body_solvers": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Solvers the 4b stage compares; each extra solver refits the 3-body baseline too, so every 4-body point is compared like for like, and doubles the (expensive) 4-body builds. Default: only the search solver. Add blocklasso to test whether raw lassolars is hiding 4-body signal (on Cu-Zr it was not: blocklasso was worse everywhere)."},
        "exclude_rounds": {"type": "integer", "default": 2, "description": "Greedy rounds of the exclude stage (each round tries removing every remaining 3-/4-body cluster type)."},
        "s_maxim_2b": {"type": ["array", "null"], "items": {"type": "number"}, "description": "Default: hyper-analyze candidates."},
        "s_maxim_3b": {"type": ["array", "null"], "items": {"type": "number"}},
        "s_maxim_4b": {"type": ["array", "null"], "items": {"type": "number"}},
        "lambda_scales": {"type": "array", "items": {"type": "number"}, "default": DEFAULTS["lambda_scales"]},
        "prefer": {"type": "string", "enum": ["cheaper", "richer"], "default": "cheaper", "description": "Among statistically tied points: cheaper = lowest estimated MD cost (a final model); richer = most coefficients (a model that will go through active learning: the literature errs toward complexity before AL and prunes once the data is final, Lindsey et al. 2025). richer also skips the exclusion stage."},
        "weights_preset": {"type": "string", "default": "uniform", "description": "Fitting weights for every fit (weights stage presets: uniform, al_driver, lindsey2020, carbon2_large, hierarchical2026). Matters when energies (or stresses) are fitted alongside forces. Changing it refits every point."},
        "cv_folds": {"type": "integer", "default": 0, "description": "k > 1: score every point by k-fold cross-validation over the training frames (held-out frames get row weight 0 in the same design matrix; correlated frames share a fold; closest contacts stay in training). ~k times the statistical power of a holdout; the external holdout is still reported as ext_holdout_*."},
        "cv_seed": {"type": "integer", "default": 0},
        "smoothings": {"type": "array", "items": {"type": "string"}, "default": DEFAULTS["smoothings"], "description": "FCUTTYP values the `smoothing` stage compares with the current one (literature: TERSOFF 0.5-0.75 for many-body models)."},
        "alphas": {"type": "array", "items": {"type": "number"}, "default": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3], "description": "Regularization strengths the `alpha` stage re-solves the chosen basis with (LASSO/ridge solvers). The cheapest statistically tied model wins, and MD cost counts nonzero coefficients, so ties go to sparser models."},
        "stress_weights": {"type": "array", "items": {"type": "number"}, "default": [1.0, 3.0, 10.0, 30.0], "description": "Stress-row weights the `stress` stage tries (only when fitting stresses)."},
        "smoothing": {"type": "string", "default": "CUBIC", "description": "fm_setup.in FCUTTYP for every fit: CUBIC (chimes_lsq default) or 'TERSOFF <f_O>' with 0 < f_O < 1. The cubic form multiplies one smoothing factor per cluster distance and shrinks 3-/4-body terms; published many-body ChIMES models use TERSOFF 0.5-0.75 (Lindsey et al., JCP 153, 134117, 2020). Changing it refits every point."},
        "objective": {"type": "string", "enum": ["auto", "force", "force+energy"], "default": "auto", "description": "auto = force+energy when energies are fitted, else force."},
        "energy_weight": {"type": "number", "default": 0.1, "description": "Score per kcal/mol/atom of energy RMSE, for objective force+energy."},
        "tolerance": {"type": "number", "default": 0.03, "description": "Accept a cheaper model scoring within this fraction of the best, or within one bootstrap standard error of it if that is larger."},
        "min_signal": {"type": "number", "default": 1.0e-9, "description": "Flag 3-/4-body terms whose median column norm is below this fraction of the 2-body median: they can only act through very large coefficients. Flagged, not excluded, unless exclude_inert."},
        "exclude_inert": {"type": "boolean", "default": False},
        "max_fit_seconds": {"type": ["number", "null"], "default": 600, "description": "Abandon a fit whose design-matrix build exceeds this (reported as status timeout)."},
        "min_gain": {"type": "number", "default": 0.05, "description": "four_body=auto runs 4b only if 3b improved the score by at least this fraction."},
        "max_param_ratio": {"type": "number", "default": 0.5, "description": "Coefficients per equation above which a point is never chosen."},
        "fitener": {"type": ["boolean", "null"], "description": "Default: data_manifest fit_hints.fitener."},
        "fitstrs": {"type": ["string", "boolean", "null"], "description": "Fit stresses: ALL (full tensor), true (diagonal) or false. Default: data_manifest fit_hints.fitstrs. Use with a weights preset that weights stresses (hierarchical2026, carbon2_large)."},
        "algorithm": {"type": "string", "default": "lassolars"},
        "alpha": {"type": "number", "default": 1.0e-5},
        "masses": {"type": ["object", "null"]},
        "workers": {"type": "integer", "default": 4},
        "machine": {"type": ["string", "null"], "description": "Submit the whole search as one Slurm job on this machine."},
        "queue": {"type": "string", "default": "batch"},
        "walltime_hours": {"type": "number", "default": 4.0},
        "cores": {"type": ["integer", "null"], "description": "Cores to request with --machine; default = the largest stage's fit count (capped at a node). Charged hours scale with this."},
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
    parser.add_argument("--four-body-solvers", dest="four_body_solvers", type=lambda s: [x for x in s.split(",") if x], default=None)
    parser.add_argument("--exclude-rounds", dest="exclude_rounds", type=int, default=2)
    parser.add_argument("--s-maxim-2b", dest="s_maxim_2b", type=_floats, default=None)
    parser.add_argument("--s-maxim-3b", dest="s_maxim_3b", type=_floats, default=None)
    parser.add_argument("--s-maxim-4b", dest="s_maxim_4b", type=_floats, default=None)
    parser.add_argument("--lambda-scales", dest="lambda_scales", type=_floats, default=None)
    parser.add_argument("--prefer", choices=["cheaper", "richer"], default="cheaper")
    parser.add_argument("--weights-preset", dest="weights_preset", default="uniform")
    parser.add_argument("--stress-weights", dest="stress_weights", type=_floats, default=None)
    parser.add_argument("--alphas", type=_floats, default=None)
    parser.add_argument("--cv-folds", dest="cv_folds", type=int, default=0)
    parser.add_argument("--cv-seed", dest="cv_seed", type=int, default=0)
    parser.add_argument("--smoothings", type=lambda s: [x.strip() for x in s.split(",") if x.strip()], default=None)
    parser.add_argument("--smoothing", default="CUBIC", help="CUBIC or 'TERSOFF <f_O>'")
    parser.add_argument("--objective", choices=["auto", "force", "force+energy"], default="auto")
    parser.add_argument("--energy-weight", dest="energy_weight", type=float, default=0.1)
    parser.add_argument("--tolerance", type=float, default=0.03)
    parser.add_argument("--min-gain", dest="min_gain", type=float, default=0.05)
    parser.add_argument("--min-signal", dest="min_signal", type=float, default=1.0e-9)
    parser.add_argument("--exclude-inert", dest="exclude_inert", action="store_true", default=False)
    parser.add_argument("--max-fit-seconds", dest="max_fit_seconds", type=float, default=600)
    parser.add_argument("--max-param-ratio", dest="max_param_ratio", type=float, default=0.5)
    parser.add_argument("--fitener", type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
    parser.add_argument("--fitstrs", default=None, help="ALL | true | false (default: data manifest fit_hints)")
    parser.add_argument("--algorithm", default="lassolars")
    parser.add_argument("--alpha", type=float, default=1.0e-5)
    parser.add_argument("--masses", type=json.loads, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--machine", default=None)
    parser.add_argument("--queue", default="batch")
    parser.add_argument("--walltime-hours", dest="walltime_hours", type=float, default=4.0)
    parser.add_argument("--cores", type=int, default=None)


def _smoothing(value) -> str:
    """Normalize and validate an FCUTTYP value: CUBIC or 'TERSOFF <f_O>'."""
    parts = str(value or "CUBIC").split()
    kind = parts[0].upper()
    if kind == "CUBIC" and len(parts) == 1:
        return "CUBIC"
    if kind == "TERSOFF" and len(parts) == 2:
        try:
            f_o = float(parts[1])
        except ValueError:
            f_o = -1.0
        if 0.0 < f_o < 1.0:
            return f"TERSOFF {f_o:g}"
    raise ValueError(f"smoothing must be CUBIC or 'TERSOFF <f_O>' with 0 < f_O < 1, got {value!r}")


def _get(args, key):
    v = getattr(args, key, None)
    return DEFAULTS.get(key) if v is None else v


def _masses(elements, given):
    from ase.data import atomic_masses, atomic_numbers

    given = given or {}
    return {e: float(given.get(e, round(atomic_masses[atomic_numbers[e]], 4))) for e in elements}


def _max_stage_points(args) -> int:
    """Largest number of fits any stage runs in parallel. Allocations are
    charged per core: a full 112-core node for ~30 short fits was ~1-3 %
    utilized on Cu-Zr (77.7 CPU-hours charged, 1.3 used)."""
    def n(key, default_len):
        v = getattr(args, key, None)
        return len(v) if v else default_len

    solvers = len(getattr(args, "four_body_solvers", None) or [1])
    return max(n("orders_2b", len(DEFAULTS["orders_2b"])) * n("s_maxim_2b", 5),
               n("orders_3b", len(DEFAULTS["orders_3b"])) * n("s_maxim_3b", 3),
               n("orders_4b", len(DEFAULTS["orders_4b"])) * n("s_maxim_4b", 2) * solvers, 4)


def _submit(args, out: Path) -> dict:
    from .. import hpc, machines

    payload = {k: v for k, v in vars(args).items()
               if not k.startswith("_") and k not in ("machine", "json_in", "json_out", "describe", "force", "dry_run",
                                                      "output_dir", "stage", "queue", "walltime_hours")}
    profile = machines.load_profile(args.machine)
    cores = getattr(args, "cores", None) or min(profile.default_ntasks_per_node or 112, _max_stage_points(args))
    payload["workers"] = cores
    inp = out / "hyper_search_input.json"
    atomic.write_json(inp, payload, indent=1, default=str)
    run_dir = out / "search"
    cmd = f"{sys.executable} -m agentic_chimes.cli hyper-search --json-in {inp} --output-dir {run_dir} --force > hyper_search.out 2>&1"
    handle = hpc.submit_job(profile, job_name="hyper-search", commands=[f"cd {out.resolve()}", cmd], work_dir=out,
                            nodes=1, ntasks_per_node=cores, walltime_hours=getattr(args, "walltime_hours", 4.0) or 4.0,
                            queue=getattr(args, "queue", "batch") or "batch", dry_run=bool(getattr(args, "dry_run", False)),
                            expect=[run_dir / "hyper_report.json"])
    return {"submitted": not handle.dry_run, "dry_run": handle.dry_run, "job_id": handle.job_id,
            "job_file": str(handle.job_file), "input": str(inp),
            "results_when_done": str(run_dir / "hyper_report.json"), "cores_requested": cores,
            "note": "The search runs inside the job; read hyper_report.json when it finishes."}


SOLVE_KEYS = ("solver", "alpha", "stress_weight", "weights_preset")  # change the solve, not the design matrix
# Peak memory of one fit, measured: chimes_lsq.py parses A.txt with genfromtxt, so a 48 MB dense matrix peaked
# at 936 MB (~15-20x). Budget 20x the dense size plus 0.25 GB of interpreter per fit.
FIT_MEMORY_FACTOR = 20.0
FIT_MEMORY_BASE = 0.25e9


def available_memory() -> float:
    """Bytes this process may use: the Slurm/cgroup limit when there is one, else free physical memory."""
    import os

    limits = []
    for f in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = open(f).read().strip()
            if v.isdigit() and int(v) < 1 << 60:
                limits.append(int(v))
        except OSError:
            pass
    try:
        limits.append(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError):
        pass
    return float(min(limits)) if limits else float("inf")


def fit_memory(rows: int, cols: int) -> float:
    return FIT_MEMORY_BASE + FIT_MEMORY_FACTOR * 8.0 * rows * cols


def _size_hint(cfg) -> int:
    o = [cfg.get("order_2b") or 0, cfg.get("order_3b") or 0, cfg.get("order_4b") or 0]
    return o[0] + o[1] ** 3 + o[2] ** 6


class _Runner:
    def __init__(self, out, base_task, workers):
        self.out, self.base, self.workers, self.all = out, base_task, workers, {}
        self.memory_notes = []

    def _parallel(self, tasks):
        """Run tasks with at most self.workers in parallel, capped by memory: the largest task runs first,
        alone, and its measured design-matrix size sets how many fits fit in memory at once."""
        if not tasks:
            return []
        tasks = sorted(tasks, key=lambda t: -_size_hint(t["cfg"]))
        if self.workers <= 1 or len(tasks) == 1:
            return [_hyper.run_point(t) for t in tasks]
        first = _hyper.run_point(tasks[0])
        workers = self.workers
        dims = None
        for d in (Path(tasks[0].get("amat_dir") or tasks[0]["point_dir"]) / "dim.txt",):
            try:
                cols, rows = (int(x) for x in d.read_text().split()[:2])
                dims = (rows, cols)
            except (OSError, ValueError):
                pass
        if dims:
            per_fit = fit_memory(*dims)
            fits = int(0.8 * available_memory() // per_fit)
            if fits < 1:
                self.memory_notes.append(f"one fit of a {dims[0]} x {dims[1]} design matrix needs ~{per_fit / 1e9:.1f} GB, "
                                         "more than this node offers: fit this basis with solve --algorithm dlars "
                                         "--machine (distributed) instead")
            if fits < workers:
                workers = max(1, fits)
                self.memory_notes.append(f"{dims[0]} x {dims[1]} design matrix: ~{per_fit / 1e9:.1f} GB per fit, so "
                                         f"{workers} parallel fits instead of {self.workers}")
        rest = tasks[1:]
        if workers > 1 and len(rest) > 1:
            with process_pool(min(workers, len(rest))) as pool:
                return [first] + list(pool.map(_hyper.run_point, rest))
        return [first] + [_hyper.run_point(t) for t in rest]

    def fit(self, cfgs, *, share_amat: bool = False):
        """Fit configurations. With share_amat, configurations that differ only in the solve
        (SOLVE_KEYS) share one design matrix: built once per basis, re-solved, then deleted."""
        tasks = []
        for cfg in cfgs:
            key = _hyper.config_key(cfg)
            t = {**self.base, "cfg": cfg, "point_dir": str(self.out / "points" / key)}
            if share_amat:
                basis = {k: v for k, v in cfg.items() if k not in SOLVE_KEYS}
                t.update({"amat_dir": str(self.out / "amat" / _hyper.config_key(basis)), "keep_amat": True})
            tasks.append(t)
        if share_amat:
            seen, firsts, rest = set(), [], []
            for t in tasks:
                (rest if t["amat_dir"] in seen else firsts).append(t)
                seen.add(t["amat_dir"])
            results = self._parallel(firsts) + self._parallel(rest)
            for d in seen:
                (Path(d) / "A.txt").unlink(missing_ok=True)
            order = {id(t): i for i, t in enumerate(firsts + rest)}
            by_key = {r["key"]: r for r in results}
            results = [by_key[_hyper.config_key(t["cfg"])] for t in tasks]
        else:
            results = self._parallel(tasks)
            by_key = {r["key"]: r for r in results}
            results = [by_key[_hyper.config_key(t["cfg"])] for t in tasks]
        for r in results:
            self.all[r["key"]] = r
        return results


def _number_density(frames) -> float:
    import numpy as np

    vals = []
    for f in frames:
        cell = np.asarray(f.box if f.non_ortho else np.diag(f.box), dtype=float)
        vals.append(f.natoms / abs(np.linalg.det(cell)))
    return float(np.median(vals))


def _f(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "" if x is None else str(x)


def render_markdown(report: dict) -> str:
    """HYPER_REPORT.md: the search's own account of what it tried and chose."""
    d, f, hp = report.get("data") or {}, report["final"], report["hyperparameters"]
    cv = report.get("cross_validation")
    L = ["# Hyperparameter search report", "",
         f"*Generated by `chimes-agent hyper-search`; numbers from `hyper_report.json` ({report['n_fits']} fits).*", "",
         "## Data budget", "",
         f"- training frames: {d.get('n_train')}; holdout frames: {d.get('n_holdout')}; elements: {', '.join(d.get('elements') or [])}",
         f"- equations: {d.get('n_force_equations')} force + {d.get('n_energy_equations')} energy",
         f"- scoring: {'%d-fold cross-validation over %s training frames (external holdout reported separately)' % (cv['folds'], cv.get('n_frames')) if cv else 'the holdout set'}",
         "", "## Chosen model", "",
         "| setting | value |", "|---|---|",
         f"| orders 2b/3b/4b | {hp['order'].get('2')}/{hp['order'].get('3') or 0}/{hp['order'].get('4') or 0} |",
         f"| inner cutoffs (Å) | {', '.join(f'{p} {v[0]}' for p, v in hp['pair_cutoffs'].items())} |",
         f"| 2-body outer cutoff (Å) | {next(iter(hp['pair_cutoffs'].values()))[1]} |",
         f"| 3-body cutoff (Å) | {hp.get('special_maxim_3b_pairs') or hp.get('special_maxim_3b') or '-'} |",
         f"| 4-body cutoff (Å) | {hp.get('special_maxim_4b') or '-'} |",
         f"| Morse λ (Å) | {', '.join(f'{p} {v}' for p, v in hp['morse_lambda'].items())} |",
         f"| smoothing | {hp.get('fcuttyp', 'CUBIC')} |",
         f"| excluded 3-body / 4-body types | {', '.join(' '.join(t) for t in hp.get('exclude_3b') or []) or 'none'} / {', '.join(' '.join(t) for t in hp.get('exclude_4b') or []) or 'none'} |",
         f"| solver / α | {hp.get('algorithm')} / {hp.get('alpha')} |",
         f"| weights / stress weight | {hp.get('weights')} / {hp.get('stress_weight') or '-'} |",
         f"| N_LAYERS | {hp.get('nlayers')} |",
         f"| coefficients (fitted) | {f.get('n_params')} |", "",
         "## Accuracy of the chosen model", "",
         "| metric | value |", "|---|---|",
         f"| relative force error ({'CV' if cv else 'holdout'}) | {_f(f.get('holdout_relative_force_error'))} ± {_f(f.get('holdout_relative_force_se'), 3)} |",
         f"| training relative force error | {_f(f.get('train_relative_force_error'))} |",
         f"| energy RMSE (kcal/mol/atom) | {_f(f.get('holdout_rmse_energy_per_atom'))} |"]
    if cv:
        L += [f"| CV median per-frame error / fragile frames | {_f(f.get('cv_median_frame_error'))} / {f.get('cv_fragile_frames', 0)} |"]
    if f.get("holdout_rmse_pressure_gpa") is not None:
        L += [f"| pressure RMSE (GPa) | {_f(f.get('holdout_rmse_pressure_gpa'), 2)} |"]
    if cv:
        L += [f"| external holdout relative force error | {_f(cv.get('ext_holdout_relative_force_error'))} |",
              f"| external holdout energy RMSE | {_f(cv.get('ext_holdout_rmse_energy_per_atom'))} |"]
    L += [""]
    bc = f.get("by_composition") or {}
    if bc:
        L += ["By composition:", "", "| composition | relative force error |", "|---|---|"]
        L += [f"| {g} | {_f(v)} |" for g, v in bc.items()] + [""]
    L += ["## Stages", ""]
    cols = [("order_2b", "2b"), ("order_3b", "3b"), ("order_4b", "4b"), ("s_maxim_2b", "r2b"), ("s_maxim_3b", "r3b"),
            ("s_maxim_4b", "r4b"), ("lambda_scale", "λ×"), ("fcuttyp", "smooth"), ("solver_alpha", "α"),
            ("stress_weight", "w_s"), ("exclude_3b", "excl3b"), ("holdout_relative_force_error", "F err"),
            ("holdout_relative_force_se", "±"), ("holdout_rmse_energy_per_atom", "E err"), ("n_params", "n"),
            ("md_cost", "MD cost"), ("group_regressions", "group regressions")]
    for st in report["stages"]:
        L += [f"### {st['stage']}", ""]
        if st.get("skipped"):
            L += [f"Skipped: {st.get('reason')}", ""]
            continue
        L += [f"{st.get('n_points')} fits. Decision: {st.get('reason')}", ""]
        rows = st.get("table") or []
        if st["stage"] == "exclude":
            for rnd in st.get("rounds") or []:
                rows = rnd.get("table") or rows
            ct = st.get("cluster_types") or []
            if ct:
                L += ["| type | instances | coefficients saved | score change when removed | excluded |", "|---|---|---|---|---|"]
                L += [f"| {c['type']} | {c.get('instances')} | {c.get('coefficients_saved')} | "
                      f"{', '.join(f'{v:+.3f}' for v in (c.get('score_change_when_removed') or {}).values()) or '-'} | "
                      f"{'yes' if c.get('excluded') else 'no'} |" for c in ct] + [""]
        if rows:
            use = [(k, h) for k, h in cols if any(r.get(k) not in (None, "", [], {}) for r in rows)]
            L += ["| " + " | ".join(h for _, h in use) + " |", "|" + "---|" * len(use)]
            for r in rows[:12]:
                cells = []
                for k, _ in use:
                    v = r.get(k)
                    if k == "exclude_3b" and v:
                        v = "; ".join(v)
                    elif k == "group_regressions" and v:
                        v = "; ".join(f"{g['group']} {g['error']:.3f} vs {g['best']:.3f}" for g in v)
                    elif k in ("solver_alpha", "stress_weight") and isinstance(v, (int, float)):
                        v = f"{v:g}"
                    elif k == "md_cost" and isinstance(v, (int, float)):
                        v = f"{v:.3g}"
                    elif isinstance(v, float):
                        v = _f(v, 4 if k in ("holdout_relative_force_se",) else 3)
                    cells.append("" if v is None else str(v))
                L += ["| " + " | ".join(cells) + " |"]
            if len(rows) > 12:
                L += [f"| … {len(rows) - 12} more rows in hyper_report.json |" + " |" * (len(use) - 1)]
            L += [""]
    prof = report.get("profiles") or {}
    if prof:
        L += ["## Sensitivity profiles", "", "Each table varies one setting with every other setting at its chosen value "
              "(`*` marks the chosen value).", ""]
        for key, rows in prof.items():
            L += [f"**{key}**", "", "| value | F err | E err | coefficients | MD cost |", "|---|---|---|---|---|"]
            L += [f"| {r['value']}{'*' if r['chosen'] else ''} | {_f(r['holdout_relative_force_error'])} | "
                  f"{_f(r['holdout_rmse_energy_per_atom'])} | {r['n_params']} | {_f(r['md_cost'], 0)} |" for r in rows] + [""]
    if report.get("notes"):
        L += ["## Notes", ""] + [f"- {n}" for n in report["notes"]] + [""]
    return "\n".join(L)


def _row(r):
    c = r["cfg"]
    keep = {k: c.get(k) for k in ("order_2b", "order_3b", "order_4b", "s_maxim_2b", "s_maxim_3b", "s_maxim_4b", "lambda_scale")}
    for k in ("fcuttyp", "lambda_scale_pairs"):
        if c.get(k):
            keep[k] = c[k]
    for k in ("cv_folds", "ext_holdout_relative_force_error", "ext_holdout_rmse_energy_per_atom", "cv_median_frame_error"):
        if r.get(k) is not None:
            keep[k] = r[k]
    if r.get("cv_fragile_frames"):
        keep["cv_fragile_frames"] = len(r["cv_fragile_frames"])
    if c.get("s_maxim_3b_pairs"):
        keep["s_maxim_3b_pairs"] = c["s_maxim_3b_pairs"]
    for k in ("solver", "exclude_3b", "exclude_4b"):
        if c.get(k):
            keep[k] = c[k] if k == "solver" else [" ".join(e) for e in c[k]]
    keep.update({k: r.get(k) for k in ("status", "n_params", "params_per_equation", "nlayers", "holdout_relative_force_error",
                                        "holdout_relative_force_se", "holdout_rmse_force", "holdout_rmse_energy_per_atom",
                                        "train_relative_force_error", "solver_alpha", "score", "md_cost", "key", "error",
                                        "holdout_rmse_pressure_gpa")})
    if c.get("stress_weight") is not None:
        keep["stress_weight"] = c["stress_weight"]
    if r.get("group_regressions"):
        keep["group_regressions"] = r["group_regressions"]
    if r.get("holdout_by_composition"):
        keep["by_composition"] = {g: v.get("relative_force_error") for g, v in r["holdout_by_composition"].items()}
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
    fs.ensure_dir(out)
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
    fitstrs = getattr(args, "fitstrs", None)
    if fitstrs is None:
        fitstrs = (manifest.get("fit_hints") or {}).get("fitstrs") or False
    fitstrs = {True: "true", False: "false"}.get(fitstrs, str(fitstrs))
    if fitstrs.lower() not in ("false", "true", "all"):
        raise ValueError(f"fitstrs must be ALL, true or false, got {fitstrs!r}")
    objective = getattr(args, "objective", "auto") or "auto"
    if objective == "auto":
        objective = "force+energy" if fitener else "force"
    min_signal = getattr(args, "min_signal", 1.0e-9)
    exclude_inert = bool(getattr(args, "exclude_inert", False))
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

    smoothing = _smoothing(getattr(args, "smoothing", None))
    prefer = getattr(args, "prefer", None) or "cheaper"
    if prefer not in ("cheaper", "richer"):
        raise ValueError(f"prefer must be cheaper or richer, got {prefer!r}")
    base_cfg = {
        "elements": elements,
        "s_minim": {p: v["suggested"]["s_minim"] for p, v in analysis["pairs"].items()},
        "morse_lambda": {p: v["suggested"]["morse_lambda"] for p, v in analysis["pairs"].items()},
        "fitener": fitener, "lambda_scale": 1.0,
        "order_3b": 0, "s_maxim_3b": None, "order_4b": 0, "s_maxim_4b": None,
    }
    if smoothing != "CUBIC":  # only non-default values enter the cache key, so CUBIC caches stay valid
        base_cfg["fcuttyp"] = smoothing
    if fitstrs.lower() != "false":  # same: stress-free caches stay valid
        base_cfg["fitstrs"] = fitstrs.upper() if fitstrs.lower() == "all" else "true"
    from . import weights as _weights

    weights_preset = getattr(args, "weights_preset", None) or "uniform"
    if weights_preset not in _weights.PRESETS:
        raise ValueError(f"weights_preset must be one of {sorted(_weights.PRESETS)}, got {weights_preset!r}")
    if weights_preset != "uniform":  # same trick: uniform caches stay valid
        base_cfg["weights_preset"] = weights_preset
    train_frames = xyzf_io.read_xyzf(train)
    n_train = len(train_frames)
    density = _number_density(train_frames)
    cv_folds = int(getattr(args, "cv_folds", 0) or 0)
    cv_extra = {}
    if cv_folds > 1:
        assign = _hyper.cv_assignment(train_frames, cv_folds, int(getattr(args, "cv_seed", 0) or 0))
        cv_extra = {"cv_folds": cv_folds, "cv_seed": int(getattr(args, "cv_seed", 0) or 0), "cv_assign": assign,
                    "train_frames": train_frames}
    elif cv_folds == 1:
        raise ValueError("cv_folds must be 0 (holdout only) or >= 2")
    runner = _Runner(out, {
        "train_xyzf": str(Path(train).resolve()), "holdout_xyzf": str(Path(holdout).resolve()), "n_train": n_train,
        "masses": _masses(elements, getattr(args, "masses", None)), "thinnest": analysis["thinnest_cell_width"],
        "algorithm": getattr(args, "algorithm", "lassolars") or "lassolars", "alpha": getattr(args, "alpha", 1e-5),
        "timeout_s": getattr(args, "max_fit_seconds", 600), **cv_extra,
    }, max(1, getattr(args, "workers", 4) or 1))

    inert = []

    def usable(r):
        sig = r.get("signal") or {}
        for body in ("3b", "4b"):
            if r["cfg"].get(f"order_{body}") and sig.get(f"signal_{body}") is not None and sig[f"signal_{body}"] < min_signal:
                inert.append((body, r["cfg"].get(f"s_maxim_{body}"), sig[f"signal_{body}"]))
                return not exclude_inert
        return True

    def pick(results):
        results = [r for r in results if r.get("status") != "done" or usable(r)]
        return _hyper.select(results, objective=objective, energy_weight=energy_weight, tolerance=tol,
                             max_param_ratio=max_ratio, density=density, prefer=prefer)

    report_stages, notes = [], []
    current = None

    def record(name, results, sel, extra=None):
        entry = {"stage": name, "n_points": len(results), "n_failed": sum(r["status"] == "failed" for r in results),
                 "n_timeout": sum(r["status"] == "timeout" for r in results),
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

    # ---- per-pair 3-body cutoffs: each pair at its own shell (Lindsey et al., JCTC 15, 436, 2019)
    if "3b_pairs" in stages and current["cfg"].get("order_3b") and len(analysis["pairs"]) > 1:
        c3 = current["cfg"]
        g = c3["s_maxim_3b"]
        g2 = (analysis.get("global") or {}).get("second_shell_end")
        shells = {p: v.get("suggested", {}) for p, v in analysis["pairs"].items()}

        def clamp(p, r):
            lo = shells[p].get("first_shell_end") or 0.0
            return round(min(max(r, lo), c3["s_maxim_2b"]), 2)

        cands = []
        own = {p: clamp(p, s.get("second_shell_end") or g) for p, s in shells.items()}
        cands.append(own)
        if g2:
            cands.append({p: clamp(p, g * (s.get("second_shell_end") or g2) / g2) for p, s in shells.items()})
        uniq = []
        for cand in cands:
            if cand not in uniq and any(abs(v - g) > 1e-6 for v in cand.values()):
                uniq.append(cand)
        if uniq:
            res = runner.fit([{**c3, "s_maxim_3b_pairs": cand} for cand in uniq])
            sel = pick(res + [current])
            record("3b_pairs", res, sel)
            if sel["chosen"]:
                current = sel["chosen"]
            if current["cfg"].get("s_maxim_3b_pairs"):
                notes.append(f"3b_pairs: per-pair 3-body cutoffs {current['cfg']['s_maxim_3b_pairs']} replace the global "
                             f"{g} A (statistically tied or better, cheaper in MD)")
        else:
            report_stages.append({"stage": "3b_pairs", "skipped": True,
                                  "reason": "per-pair shell cutoffs coincide with the global 3-body cutoff"})

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
        base_solver = c3.get("solver", runner.base["algorithm"])
        solvers = getattr(args, "four_body_solvers", None) or [base_solver]

        def with_solver(cfg, s):
            return cfg if s == base_solver else {**cfg, "solver": s}

        baselines = [with_solver(c3, s) for s in solvers if s != base_solver]
        cfgs = [with_solver({**c3, "order_4b": o, "s_maxim_4b": c}, s)
                for s in solvers for o in grid["orders_4b"] for c in grid["s_maxim_4b"] if c <= c3["s_maxim_3b"]]
        res = runner.fit(baselines + cfgs, share_amat=True)  # solver variants re-solve one matrix per basis
        sel = pick(res + [current])
        per_solver = {}
        for s in solvers:
            base = current if s == base_solver else next((r for r in res if r["cfg"].get("solver") == s and not r["cfg"].get("order_4b")), None)
            fours = [r for r in res if r.get("status") == "done" and r["cfg"].get("order_4b") and r["cfg"].get("solver", base_solver) == s]
            if base and base.get("status") == "done" and fours:
                best4 = min(fours, key=lambda r: _hyper.score(r, objective, energy_weight))
                per_solver[s] = {"three_body_score": round(_hyper.score(base, objective, energy_weight), 4),
                                 "best_four_body_score": round(_hyper.score(best4, objective, energy_weight), 4),
                                 "best_four_body": {k: best4["cfg"].get(k) for k in ("order_4b", "s_maxim_4b")}}
        kept = bool(sel["chosen"] and sel["chosen"]["cfg"].get("order_4b"))
        record("4b", res, sel, {"kept_four_body": kept, "solvers": solvers, "four_body_gain_by_solver": per_solver})
        if sel["chosen"] and (kept or sel["chosen"]["cfg"].get("solver", base_solver) != base_solver):
            current = sel["chosen"]
        if not kept:
            notes.append("4b: no 4-body model beat the current model by more than the tie margin with any solver "
                         f"({', '.join(solvers)}); no 4-body terms")
        elif current["cfg"].get("solver", base_solver) != base_solver:
            notes.append(f"4b: the chosen model uses solver {current['cfg']['solver']}; later stages keep it")

    # ---- smoothing function: the cubic form shrinks many-body terms (Lindsey 2020); compare TERSOFF variants
    if "smoothing" in stages and (current["cfg"].get("order_3b") or current["cfg"].get("order_4b")):
        now = current["cfg"].get("fcuttyp", "CUBIC")
        variants = [_smoothing(v) for v in (getattr(args, "smoothings", None) or DEFAULTS["smoothings"])]
        cfgs = [{**current["cfg"], "fcuttyp": v} for v in variants if v != now]
        if cfgs:
            res = runner.fit(cfgs)
            sel = pick([current] + res)
            record("smoothing", res, sel)
            if sel["chosen"] and sel["chosen"] is not current:
                current = sel["chosen"]
                notes.append(f"smoothing: {current['cfg']['fcuttyp']} replaced {now} (tied or better; many-body signal "
                             f"{(current.get('signal') or {}).get('signal_3b')} vs {(sel['best'].get('signal') or {}).get('signal_3b')})")
        else:
            report_stages.append({"stage": "smoothing", "skipped": True, "reason": "no other smoothing to compare"})
    elif "smoothing" in stages:
        report_stages.append({"stage": "smoothing", "skipped": True, "reason": "no many-body terms"})

    # ---- cluster-type exclusions
    if "exclude" in stages and prefer == "richer":
        report_stages.append({"stage": "exclude", "skipped": True,
                              "reason": "prefer=richer keeps every cluster type (pruning belongs after active learning)"})
    elif "exclude" in stages and (current["cfg"].get("order_3b") or current["cfg"].get("order_4b")):
        coverage = current.get("cluster_coverage")
        if not coverage and current.get("point_dir") and (Path(current["point_dir"]) / "fm_setup.log").is_file():
            coverage = _hyper.cluster_coverage(Path(current["point_dir"]))  # fits cached before coverage was recorded
        coverage = coverage or {}
        type_rows = {}
        for body in ("3b", "4b"):
            if not current["cfg"].get(f"order_{body}"):
                continue
            for t, info in (coverage.get(body) or {}).items():
                type_rows[(body, t)] = {"body": body, "type": t, "instances": info.get("instances"),
                                        "min_distances": info.get("min_distances")}
        rounds = []
        entry = current  # every round is also judged against the model that entered the stage,
        # so losses that are each within the tie margin cannot add up across rounds
        for rnd in range(max(0, getattr(args, "exclude_rounds", 2) or 0)):
            c = current["cfg"]
            excluded = {("3b", " ".join(e)) for e in c.get("exclude_3b") or []} | {("4b", " ".join(e)) for e in c.get("exclude_4b") or []}
            cands, keys = [], []
            for (body, t) in type_rows:
                if (body, t) in excluded:
                    continue
                key = f"exclude_{body}"
                cands.append({**c, key: sorted((c.get(key) or []) + [t.split()])})
                keys.append((body, t))
            if not cands:
                break
            res = runner.fit(cands)
            pool = res + [current] + ([entry] if entry is not current else [])
            sel = pick(pool)
            base_score = current["score"]
            for (body, t), r in zip(keys, res):
                if r.get("status") == "done":
                    s = _hyper.score(r, objective, energy_weight)
                    type_rows[(body, t)].setdefault("score_change_when_removed", {})[f"round{rnd + 1}"] = round(s - base_score, 4)
                    type_rows[(body, t)].setdefault("coefficients_saved", current["n_params"] - r["n_params"])
            rounds.append({"round": rnd + 1, "reason": sel["reason"],
                           "table": [_row(r) for r in sorted(res, key=lambda r: r.get("score", 9e9))]})
            if sel["chosen"] is None or sel["chosen"]["key"] in (current["key"], entry["key"]):
                break
            current = sel["chosen"]
        excluded_final = {"3b": current["cfg"].get("exclude_3b") or [], "4b": current["cfg"].get("exclude_4b") or []}
        for (body, t), row in type_rows.items():
            row["excluded"] = t.split() in excluded_final[body]
            if row.get("instances") == 0:
                row["note"] = "absent from the training data: unconstrained if kept"
        if current is not entry and entry.get("train_relative_force_error") and current.get("train_relative_force_error"):
            rise = current["train_relative_force_error"] / entry["train_relative_force_error"] - 1
            if rise > 0.10:
                notes.append(f"exclude: training error rose {rise:.0%} with the exclusions "
                             f"({entry['train_relative_force_error']:.3f} -> {current['train_relative_force_error']:.3f}) while the "
                             "holdout could not resolve a difference; with a small holdout this is a real but unmeasurable "
                             "loss -- consider keeping the types, or re-judge with more holdout data")
        if not type_rows:
            reason = "no cluster-type data for the current model; exclusions not tested"
            notes.append("exclude: " + reason)
        elif any(excluded_final.values()):
            reason = f"excluded {sum(len(v) for v in excluded_final.values())} cluster type(s)"
        else:
            reason = "every cluster type is needed (removing any one hurts beyond the tie margin)"
        report_stages.append({"stage": "exclude", "n_points": sum(len(r["table"]) for r in rounds),
                              "reason": reason,
                              "cluster_types": list(type_rows.values()), "rounds": rounds})

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
        # per pair: one pair's lambda at a time, 0.9x and 1.1x of its current value
        if len(current["cfg"]["morse_lambda"]) > 1:
            cfgs = [{**current["cfg"], "lambda_scale_pairs": {**(current["cfg"].get("lambda_scale_pairs") or {}), p: f}}
                    for p in current["cfg"]["morse_lambda"] for f in (0.9, 1.1)]
            res = runner.fit(cfgs)
            sel = pick([current] + res)
            record("lambda_pairs", res, sel)
            if sel["chosen"] and sel["chosen"] is not current:
                current = sel["chosen"]
                notes.append(f"lambda_pairs: per-pair lambda scales {current['cfg']['lambda_scale_pairs']} beat the global "
                             "scale beyond the tie margin")

    # ---- refine 2-body order with many-body terms in place
    if "refine" in stages and (current["cfg"].get("order_3b") or current["cfg"].get("order_4b")):
        o = current["cfg"]["order_2b"]
        cfgs = [{**current["cfg"], "order_2b": v} for v in sorted({o - 2, o, o + 2}) if v >= 4]
        # cutoff midpoints between the chosen value and its grid neighbours (orders fixed)
        c = current["cfg"]
        for key, values, upper in (("s_maxim_2b", grid["s_maxim_2b"], None),
                                   ("s_maxim_3b", grid["s_maxim_3b"], c["s_maxim_2b"])):
            if not c.get(key) or key == "s_maxim_3b" and c.get("s_maxim_3b_pairs"):
                continue
            vals = sorted(set(values))
            i = vals.index(c[key]) if c[key] in vals else None
            for j in ((i - 1, i + 1) if i is not None else ()):
                if 0 <= j < len(vals):
                    mid = round(0.5 * (vals[i] + vals[j]), 2)
                    if upper is None or mid <= upper:
                        cfgs.append({**c, key: mid})
        res = runner.fit(cfgs)
        sel = pick(res)
        record("refine", res, sel)
        if sel["chosen"]:
            current = sel["chosen"]
            if current["cfg"]["order_2b"] < min(grid["orders_2b"]):
                notes.append(f"refine moved order_2b below the 2b grid ({current['cfg']['order_2b']}); lower orders may be worth trying")
            elif current["cfg"]["order_2b"] > max(grid["orders_2b"]):
                notes.append(f"refine moved order_2b above the 2b grid ({current['cfg']['order_2b']}); higher orders may be worth trying")

    # ---- regularization strength: alpha was fixed everywhere; on Cu-Zr energy error moved 0.78-1.12 kcal/mol/atom
    lasso_like = ("lassolars", "lasso", "nlasso", "blocklasso", "ridge", "nridge")
    solver_now = current["cfg"].get("solver", runner.base["algorithm"])
    if "alpha" in stages and solver_now in lasso_like:
        alphas = getattr(args, "alphas", None) or DEFAULTS["alphas"]
        cfgs = [{**current["cfg"], "alpha": float(a)} for a in alphas]
        res = runner.fit(cfgs, share_amat=True)  # one design matrix, several solves
        sel = pick(res + [current])
        record("alpha", res, sel)
        if sel["chosen"]:
            current = sel["chosen"]
        a_now = current["cfg"].get("alpha", runner.base["alpha"])
        notes += _edge_notes("alpha", {"cfg": {**current["cfg"], "alpha": a_now}}, "alpha", sorted(set(alphas) | {runner.base["alpha"]}))
    elif "alpha" in stages:
        report_stages.append({"stage": "alpha", "skipped": True,
                              "reason": f"solver {solver_now} chooses its own regularization (or has none)"})

    # ---- stress weight: published values (100-250) assume large cells; measure it on this data
    if "stress" in stages and fitstrs.lower() != "false":
        weights_try = getattr(args, "stress_weights", None) or DEFAULTS["stress_weights"]
        cfgs = [{**current["cfg"], "stress_weight": float(w)} for w in weights_try]
        res = runner.fit(cfgs, share_amat=True)
        base = current
        base.setdefault("score", _hyper.score(base, objective, energy_weight))
        tied = [base]
        for r in res:
            if r.get("status") != "done" or r.get("holdout_rmse_pressure_gpa") is None:
                continue
            r["score"] = _hyper.score(r, objective, energy_weight)
            se = (_hyper.paired_se(r["holdout_per_frame_force"], base["holdout_per_frame_force"])
                  if r.get("holdout_per_frame_force") and base.get("holdout_per_frame_force") else 0.0)
            r["tie_margin"] = round(max(tol * base["score"], se), 5)
            if r["score"] - base["score"] <= r["tie_margin"]:
                tied.append(r)
        best = min(tied, key=lambda r: (r.get("holdout_rmse_pressure_gpa") if r.get("holdout_rmse_pressure_gpa") is not None
                                        else float("inf")))
        if best is not base:
            reason = (f"stress weight {best['cfg']['stress_weight']} lowered the holdout pressure error to "
                      f"{best['holdout_rmse_pressure_gpa']:.2f} GPa (from {base.get('holdout_rmse_pressure_gpa') or float('nan'):.2f}) "
                      "with force/energy score statistically tied")
            current = best
        else:
            reason = "no stress weight improved the pressure error without a force/energy cost beyond the tie margin"
        report_stages.append({"stage": "stress", "n_points": len(res), "reason": reason,
                              "table": [_row(r) for r in sorted([base] + res, key=lambda r: r["cfg"].get("stress_weight") or 0)]})
    elif "stress" in stages:
        report_stages.append({"stage": "stress", "skipped": True, "reason": "not fitting stresses (fitstrs false)"})

    if inert:
        seen = sorted({(b, c) for b, c, _ in inert})
        notes.append("many-body terms with columns below min_signal of the 2-body scale at "
                     + ", ".join(f"{b} cutoff {c}" for b, c in seen)
                     + ": ChIMES' cubic smoothing multiplies one (1 - r/r_c)^3 factor per cluster distance, so these cutoffs "
                       "barely exceed the neighbour distances and the terms need very large coefficients. "
                       + ("They were excluded." if exclude_inert else "Treat any gain from them with suspicion."))

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
        "morse_lambda": _hyper.scaled_lambdas(c),
        "special_maxim_3b": c.get("s_maxim_3b") if c.get("order_3b") else None,
        "special_maxim_3b_pairs": c.get("s_maxim_3b_pairs") if c.get("order_3b") else None,
        "special_maxim_4b": c.get("s_maxim_4b") if c.get("order_4b") else None,
        "exclude_3b": c.get("exclude_3b") or None,
        "exclude_4b": c.get("exclude_4b") or None,
        "nlayers": final["nlayers"],
        "fcuttyp": c.get("fcuttyp", "CUBIC"),
        "masses": runner.base["masses"],
        "fitener": c["fitener"],
        "fitstrs": c.get("fitstrs", "false"),
        "stress_weight": c.get("stress_weight"),
        "algorithm": c.get("solver", runner.base["algorithm"]),
        "alpha": c.get("alpha", final.get("solver_alpha", runner.base["alpha"])),
        "weights": c.get("weights_preset", "uniform"),
    }
    atomic.write_json((best_dir / "hyper_choice.json"), choice, indent=1)

    gap = final["holdout_relative_force_error"] - final["train_relative_force_error"]
    if final["train_relative_force_error"] and gap > 0.5 * final["train_relative_force_error"]:
        notes.append(f"holdout error exceeds training error by {gap:.3f} (relative): possible overfitting or train/holdout mismatch")
    if final["holdout_relative_force_error"] > 0.3:
        notes.append(f"final relative force error {final['holdout_relative_force_error']:.2f} is high: the data (coverage, size, "
                     "consistency) is more likely the limit than these hyperparameters")
    if cv_folds > 1 and final.get("cv_fragile_frames"):
        fr = final["cv_fragile_frames"]
        notes.append(f"cross-validation: {len(fr)} training frame(s) are mispredicted (relative error > 1) when held out "
                     f"(indices {fr[:8]}{', ...' if len(fr) > 8 else ''}); median per-frame error {final.get('cv_median_frame_error'):.3f} "
                     f"vs pooled {final['holdout_relative_force_error']:.3f}. The model extrapolates there: those frames are "
                     "singular in the data (nothing similar remains once they are removed). Active learning should add "
                     "configurations near them; a model with smaller many-body coefficients is more robust")
    if cv_folds > 1 and final.get("ext_holdout_relative_force_error") is not None and \
            final["holdout_relative_force_error"] > 1.5 * final["ext_holdout_relative_force_error"]:
        notes.append(f"cross-validation ({final['holdout_relative_force_error']:.3f}) is much worse than the external holdout "
                     f"({final['ext_holdout_relative_force_error']:.3f}): high variance, the fit depends strongly on which "
                     "frames are present. Prefer the candidate with the smaller gap, TERSOFF smoothing, or fewer many-body terms")
    if final.get("holdout_frames_below_inner_cutoff"):
        notes.append(f"{final['holdout_frames_below_inner_cutoff']} holdout frame(s) have contacts inside the inner cutoff "
                     "(penalty region) and dominate the holdout error; re-split with data-curate/dataset-select, which "
                     "keep each pair's closest contact in training")
    if current["cfg"].get("fcuttyp", "CUBIC") == "CUBIC" and any(s in stages for s in ("3b", "4b")):
        notes.append("many-body terms were fitted with CUBIC smoothing, which shrinks 3-/4-body contributions; published "
                     "many-body ChIMES models use TERSOFF 0.5-0.75 (Lindsey et al. 2020). A 'no gain' from 3-/4-body terms "
                     "may reflect the smoothing; consider re-running those stages with --smoothing 'TERSOFF 0.5'")
    if analysis["n_frames"] < 200:
        notes.append(f"only {analysis['n_frames']} training frames and a small holdout: differences of a few percent between "
                     "points are within noise, which is why the tolerance favours smaller models")

    prof = _hyper.profiles(runner.all, final["cfg"])
    report = {
        "final": _row(final),
        "hyperparameters": choice,
        "profiles": prof,
        "cross_validation": {"folds": cv_folds, "n_frames": final.get("cv_n_frames"),
                             "ext_holdout_relative_force_error": final.get("ext_holdout_relative_force_error"),
                             "ext_holdout_rmse_energy_per_atom": final.get("ext_holdout_rmse_energy_per_atom"),
                             "ext_holdout_by_composition": final.get("ext_holdout_by_composition")} if cv_folds > 1 else None,
        "data": {"n_train": n_train, "n_holdout": len(xyzf_io.read_xyzf(holdout)), "elements": elements,
                 "n_force_equations": analysis.get("n_force_equations"), "n_energy_equations": analysis.get("n_energy_equations")},
        "best_dir": str(best_dir),
        "stages": report_stages,
        "notes": notes + runner.memory_notes + analysis.get("notes", []),
        "settings": {"prefer": prefer, "smoothing": smoothing, "weights_preset": weights_preset, "objective": objective, "energy_weight": energy_weight, "tolerance": tol, "min_gain": min_gain, "min_signal": min_signal,
                     "max_param_ratio": max_ratio, "grid": grid, "stages": stages, "four_body": four_body,
                     "train_xyzf": str(train), "holdout_xyzf": str(holdout)},
        "n_fits": len(runner.all),
    }
    path = out / "hyper_report.json"
    atomic.write_json(path, report, indent=1, default=str)
    (out / "HYPER_REPORT.md").write_text(render_markdown(report))
    return {"hyper_report": str(path), "params": str(best_dir / "params.txt"), "fm_setup_in": str(best_dir / "fm_setup.in"),
            "hyper_choice": str(best_dir / "hyper_choice.json"), "final": report["final"], "hyperparameters": choice,
            "stage_summary": [{k: s[k] for k in ("stage", "n_points", "skipped", "reason", "kept_three_body", "kept_four_body", "four_body_gain_by_solver") if k in s} for s in report_stages],
            "notes": report["notes"], "n_fits": report["n_fits"]}
