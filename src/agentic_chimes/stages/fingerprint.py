"""Cluster-graph fingerprints and Mahalanobis comparisons of datasets
(Laubach, Lordi, Lindsey, JCIM 66, 182, 2026); see io/fingerprint.py.

Uses:

- **Coverage**: is a candidate set (MD frames, a new database pull)
  distinguishable from the training set? That is D^2 between the two sets
  against a chi-squared critical value.
- **Novelty / active-learning selection**: which candidate frames are
  outliers relative to the training set? That is D_j^2 per frame. Novel
  frames are written to `novel.xyzf`, ready for `qe-relabel`.
- **Stopping active learning**: once ChIMES-MD frames at a state point are
  statistically indistinguishable from the DFT frames there, active learning
  has converged for that state point.

By default the fingerprint is type-agnostic, as the shipped tool computes
it: for alloys it compares structure, not chemical order. `--structure-weight`
(the paper's alpha) switches to the element-aware hybrid metric: 1 is
structure only, 0 composition only (element descriptor: the mass column of
params.txt, or `--descriptor number` for atomic numbers). The paper's
finding: for metallic alloys composition resolves configurations structure
treats as degenerate, so a small structural weight (0.25) is the data-
efficient choice there, while for molecular systems alpha hardly matters.
`--structure-weights` adds a sweep so a dataset can be diagnosed either way.
Cutoffs, Morse lambdas and the default body orders come from a model's
params.txt.
"""

from __future__ import annotations

import json
from ..io.pool import process_pool
from pathlib import Path

import numpy as np

from ..io import fingerprint as fp
from ..io import xyzf as xyzf_io
from ..io import atomic
from ..io import fs

NAME = "fingerprint"
SUPPORTS_DRY_RUN = True
SUMMARY = "Cluster-graph fingerprints: dataset coverage (D^2), per-frame novelty (D_j^2), novel frames for active learning."
SCHEMA = {
    "type": "object",
    "required": ["params", "reference_xyzf"],
    "properties": {
        "params": {"type": "string", "description": "params.txt whose cutoffs and Morse lambdas define the clusters."},
        "reference_xyzf": {"type": "string", "description": "Reference set, e.g. the training data."},
        "candidates_xyzf": {"type": ["string", "null"], "description": "Set to compare: MD frames, a new pool, the md-check harvest."},
        "orders": {"type": ["array", "null"], "items": {"type": "integer"}, "description": "Body orders in the fingerprint. Default: the model's (2, plus 3/4 if it has them)."},
        "max_frames": {"type": ["integer", "null"], "default": 500, "description": "Cap per set (evenly spaced)."},
        "max_clusters": {"type": "integer", "default": 5000, "description": "Clusters per order per frame used for the dissimilarity histogram (random subset above this)."},
        "alpha": {"type": "number", "default": 0.1, "description": "Significance level of the chi-squared tests (the paper used 0.1)."},
        "structure_weight": {"type": ["number", "null"], "description": "The paper's alpha in [0, 1]: weight of structure vs composition. Default: the type-agnostic fingerprint."},
        "structure_weights": {"type": ["array", "null"], "items": {"type": "number"}, "description": "Extra alphas to report D2 and novelty for (a diagnosis sweep)."},
        "descriptor": {"type": "string", "default": "mass", "enum": ["mass", "number"], "description": "Element descriptor for the composition term."},
        "workers": {"type": "integer", "default": 1},
        "machine": {"type": ["string", "null"], "description": "Run as one Slurm job (all frames in parallel on one node)."},
        "queue": {"type": "string", "default": "debug"},
        "walltime_hours": {"type": "number", "default": 1.0},
    },
}


def add_arguments(parser) -> None:
    parser.add_argument("--params", default=None)
    parser.add_argument("--reference-xyzf", dest="reference_xyzf", default=None)
    parser.add_argument("--candidates-xyzf", dest="candidates_xyzf", default=None)
    parser.add_argument("--orders", type=lambda s: [int(x) for x in s.split(",") if x], default=None)
    parser.add_argument("--max-frames", dest="max_frames", type=int, default=500)
    parser.add_argument("--max-clusters", dest="max_clusters", type=int, default=5000)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--structure-weight", dest="structure_weight", type=float, default=None)
    parser.add_argument("--structure-weights", dest="structure_weights", type=lambda s: [float(x) for x in s.split(",") if x], default=None)
    parser.add_argument("--descriptor", default="mass", choices=["mass", "number"])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--machine", default=None)
    parser.add_argument("--queue", default="debug")
    parser.add_argument("--walltime-hours", dest="walltime_hours", type=float, default=1.0)


def model_orders(params_path) -> list:
    for ln in Path(params_path).read_text().splitlines():
        if ln.startswith("PAIRTYP:"):
            t = ln.split()
            o3 = int(t[3]) if len(t) > 3 else 0
            o4 = int(t[4]) if len(t) > 4 and t[4].lstrip("-").isdigit() else 0
            return [2] + ([3] if o3 > 0 else []) + ([4] if o4 > 0 else [])
    return [2, 3]


def element_descriptor(model, kind: str = "mass") -> dict:
    """{element: value} for the composition term: params.txt masses (the fork's default) or atomic numbers."""
    if kind == "number":
        from ase.data import atomic_numbers

        return {e: float(atomic_numbers[e]) for e in model.elements}
    return dict(model.descriptor)


def _one(task):
    """(fingerprints, cluster counts) for one frame. `weights` is a list of the
    paper's alpha values (None = type-agnostic); clusters are enumerated once
    and histogrammed per weight. Returns one fingerprint per weight."""
    frame, params, orders, max_clusters, *rest = task
    weights = rest[0] if rest else [None]
    descriptor_kind = rest[1] if len(rest) > 1 else "mass"
    model = fp.ModelCutoffs(params)
    cl = fp.typed_clusters(frame, model, orders)
    desc = element_descriptor(model, descriptor_kind)
    fps = []
    for w in weights:
        fps.append(np.concatenate([fp.histogram(cl[o][0], o, max_clusters=max_clusters, types=cl[o][1], alpha=w,
                                                descriptor=desc)[0] for o in orders]))
    counts = {o: int(len(cl[o][0])) for o in orders}
    return (fps[0] if len(weights) == 1 else fps), counts


def _subsample(frames, cap):
    if cap and len(frames) > cap:
        idx = np.linspace(0, len(frames) - 1, cap).round().astype(int)
        return [frames[i] for i in idx], idx.tolist()
    return frames, list(range(len(frames)))


def _fingerprints(frames, params, orders, max_clusters, workers, weights=(None,), descriptor="mass"):
    """({weight: array (n_frames, n_bins)}, counts per frame)."""
    weights = list(weights)
    tasks = [(f, params, orders, max_clusters, weights, descriptor) for f in frames]
    if workers > 1 and len(tasks) > 1:
        with process_pool(workers) as pool:
            res = list(pool.map(_one, tasks))
    else:
        res = [_one(t) for t in tasks]
    if len(weights) == 1:
        return {weights[0]: np.array([r[0] for r in res])}, [r[1] for r in res]
    return {w: np.array([r[0][k] for r in res]) for k, w in enumerate(weights)}, [r[1] for r in res]


def _submit(args, out: Path) -> dict:
    import sys

    from .. import hpc, machines

    payload = {k: v for k, v in vars(args).items()
               if not k.startswith("_") and k not in ("machine", "json_in", "json_out", "describe", "force", "dry_run",
                                                      "output_dir", "stage", "queue", "walltime_hours")}
    for k in ("params", "reference_xyzf", "candidates_xyzf"):
        if payload.get(k):
            payload[k] = str(Path(payload[k]).resolve())
    profile = machines.load_profile(args.machine)
    payload["workers"] = cores = profile.default_ntasks_per_node or 112
    inp = out / "fingerprint_input.json"
    atomic.write_json(inp, payload, indent=1, default=str)
    run_dir = out / "run"
    cmd = (f"{sys.executable} -m agentic_chimes.cli fingerprint --json-in {inp} --output-dir {run_dir} --force "
           "> fingerprint.out 2>&1")
    handle = hpc.submit_job(profile, job_name="fingerprint", commands=[f"cd {out.resolve()}", cmd], work_dir=out, nodes=1,
                            ntasks_per_node=cores, walltime_hours=getattr(args, "walltime_hours", 1.0) or 1.0,
                            queue=getattr(args, "queue", "debug") or "debug", dry_run=bool(getattr(args, "dry_run", False)),
                            expect=[run_dir / "fingerprint.json"])
    return {"submitted": not handle.dry_run, "dry_run": handle.dry_run, "job_id": handle.job_id,
            "job_file": str(handle.job_file), "results_when_done": str(run_dir / "fingerprint.json")}


def run(args) -> dict:
    if not getattr(args, "params", None) or not getattr(args, "reference_xyzf", None):
        raise ValueError("fingerprint needs --params and --reference-xyzf")
    if getattr(args, "machine", None):
        out = Path(getattr(args, "output_dir", None) or ".").resolve()
        fs.ensure_dir(out)
        return _submit(args, out)
    params = str(Path(args.params).resolve())
    orders = getattr(args, "orders", None) or model_orders(params)
    cap = getattr(args, "max_frames", 500)
    max_clusters = getattr(args, "max_clusters", 5000) or 5000
    alpha = getattr(args, "alpha", 0.1) or 0.1
    primary = getattr(args, "structure_weight", None)
    sweep = [w for w in (getattr(args, "structure_weights", None) or []) if w != primary]
    if sweep and primary is not None and None not in sweep:
        sweep.append(None)          # the type-agnostic baseline is always in the table
    weights = [primary] + sweep
    descriptor = getattr(args, "descriptor", "mass") or "mass"
    model = fp.ModelCutoffs(params)
    if any(w is not None for w in weights) and len(set(fp_desc := element_descriptor(model, descriptor).values())) < 2:
        raise ValueError(f"--structure-weight needs at least two element descriptor values; {descriptor} gives {sorted(fp_desc)}")
    from .quests_stage import available_cpus

    workers = max(1, min(getattr(args, "workers", 1) or 1, available_cpus()))
    out = Path(getattr(args, "output_dir", None) or ".")
    fs.ensure_dir(out)

    ref_frames, ref_idx = _subsample(xyzf_io.read_xyzf(args.reference_xyzf), cap)
    fref_all, cref = _fingerprints(ref_frames, params, orders, max_clusters, workers, weights, descriptor)
    fref = fref_all[primary]
    result = {"orders": orders, "n_reference": len(ref_frames), "alpha": alpha, "structure_weight": primary,
              "descriptor": descriptor if primary is not None or sweep else None,
              "element_descriptor": element_descriptor(model, descriptor) if primary is not None or sweep else None,
              "clusters_per_frame_reference": {o: float(np.mean([c[o] for c in cref])) for o in orders}}
    arrays = {"reference": fref, "reference_index": np.array(ref_idx)}
    _label = lambda w: "type-agnostic" if w is None else f"{w:g}"
    for w in sweep:
        arrays[f"reference_w{_label(w)}"] = fref_all[w]
    notes = []
    empty = [o for o in orders if all(c[o] < 2 for c in cref)]
    if empty:
        notes.append(f"orders {empty} have <2 clusters in every reference frame (cells too small or cutoffs too short): "
                     "those histogram blocks carry no information")

    if getattr(args, "candidates_xyzf", None):
        cand_frames, cand_idx = _subsample(xyzf_io.read_xyzf(args.candidates_xyzf), cap)
        fcand_all, ccand = _fingerprints(cand_frames, params, orders, max_clusters, workers, weights, descriptor)
        fcand = fcand_all[primary]
        arrays.update({"candidates": fcand, "candidates_index": np.array(cand_idx)})
        result["n_candidates"] = len(cand_frames)
        if sweep:
            # the paper's diagnosis: does composition (low alpha) or structure (high alpha) separate the sets?
            table = {}
            for w in weights:
                r_, c_ = fref_all[w], fcand_all[w]
                nov_w = fp.novelty(r_, c_, alpha)
                row = {"fraction_novel": nov_w["fraction_novel"], "Dj2_median": float(np.median(nov_w["Dj2"]))}
                if len(r_) >= 2 and len(c_) >= 2:
                    sets_w = fp.mahalanobis_sets(r_, c_, alpha)
                    row.update({"D2": sets_w["D2"], "critical": sets_w["critical"], "distinguishable": sets_w["distinguishable"]})
                table[_label(w)] = row
                if w != primary:
                    arrays[f"candidates_w{_label(w)}"] = c_
            result["by_structure_weight"] = table
            from ..io import plots

            order_keys = sorted((k for k in table if k != "type-agnostic"), key=float) + (["type-agnostic"] if "type-agnostic" in table else [])
            ratios = {f"α = {k}" if k != "type-agnostic" else k: table[k]["D2"] / table[k]["critical"]
                      for k in order_keys if table[k].get("D2") is not None}
            if ratios:
                png = plots.bars(ratios, out / "fingerprint_structure_weight.png", ylabel="D² / critical",
                                 title="Set separation vs structure weight (0 = composition, 1 = structure)")
                if png:
                    result["plot"] = png
            # compare D2 relative to its critical value: the composition-only histograms have few distinct values,
            # so their covariance rank (the chi-squared dof) is much lower and raw D2 values are not comparable
            d2s = {k: v["D2"] / v["critical"] for k, v in table.items() if v.get("D2") is not None and k != "type-agnostic"}
            for k, v in table.items():
                if v.get("D2") is not None:
                    v["D2_over_critical"] = round(v["D2"] / v["critical"], 2)
            if len(d2s) >= 2:
                lo, hi = min(d2s, key=float), max(d2s, key=float)
                if d2s[lo] > 1.5 * d2s[hi]:
                    notes.append(f"composition separates the sets more than structure (D2/critical {d2s[lo]:.1f} at alpha={lo} vs "
                                 f"{d2s[hi]:.1f} at alpha={hi}): chemical order differs; the paper's metallic-alloy regime, "
                                 "where a small structural weight selects data most efficiently")
                elif d2s[hi] > 1.5 * d2s[lo]:
                    notes.append(f"structure separates the sets more than composition (D2/critical {d2s[hi]:.1f} at alpha={hi} vs "
                                 f"{d2s[lo]:.1f} at alpha={lo}): geometry already encodes the chemistry (the paper's "
                                 "molecular regime); the type-agnostic fingerprint suffices")
                else:
                    notes.append(f"structure and composition separate the sets about equally (D2/critical {d2s[hi]:.1f} at "
                                 f"alpha={hi}, {d2s[lo]:.1f} at alpha={lo})")
        if len(ref_frames) >= 2 and len(cand_frames) >= 2:
            result["sets"] = fp.mahalanobis_sets(fref, fcand, alpha)
        nov = fp.novelty(fref, fcand, alpha)
        result["novelty"] = {k: v for k, v in nov.items() if k not in ("Dj2", "novel")}
        result["novelty"]["Dj2_quantiles"] = [float(np.quantile(nov["Dj2"], q)) for q in (0.1, 0.5, 0.9)]
        novel = [cand_frames[i] for i, flag in enumerate(nov["novel"]) if flag]
        if novel:
            path = out / "novel.xyzf"
            xyzf_io.write_xyzf(novel, path)
            result["novel_xyzf"] = str(path)
            result["novel_candidate_indices"] = [cand_idx[i] for i, flag in enumerate(nov["novel"]) if flag]
        arrays["candidates_Dj2"] = np.array(nov["Dj2"])
        if result.get("sets"):
            notes.append("candidate set is statistically distinguishable from the reference (D2 > critical): it samples "
                         "configurations the reference does not" if result["sets"]["distinguishable"] else
                         "candidate set is statistically indistinguishable from the reference at alpha="
                         f"{alpha}: for MD-vs-DFT frames at one state point, active learning has converged there")
        if result["novelty"]["dof"] < 3:
            notes.append("covariance rank < 3: too few or too similar reference frames for a meaningful test")
    np.savez(out / "fingerprints.npz", **arrays)
    result.update({"fingerprints": str(out / "fingerprints.npz"), "notes": notes})
    atomic.write_json((out / "fingerprint.json"), result, indent=1, default=str)
    return result
