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

The fingerprint is type-agnostic, as in the paper and the shipped tool; for
alloys it compares structure, not chemical order. Cutoffs, Morse lambdas and
the default body orders come from a model's params.txt.
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


def _one(task):
    frame, params, orders, max_clusters = task
    model = fp.ModelCutoffs(params)
    cl = fp.clusters(frame, model, orders)
    parts, counts = [], {}
    for o in orders:
        h, n = fp.histogram(cl[o], o, max_clusters=max_clusters)
        parts.append(h)
        counts[o] = int(len(cl[o]))
    return np.concatenate(parts), counts


def _subsample(frames, cap):
    if cap and len(frames) > cap:
        idx = np.linspace(0, len(frames) - 1, cap).round().astype(int)
        return [frames[i] for i in idx], idx.tolist()
    return frames, list(range(len(frames)))


def _fingerprints(frames, params, orders, max_clusters, workers):
    tasks = [(f, params, orders, max_clusters) for f in frames]
    if workers > 1 and len(tasks) > 1:
        with process_pool(workers) as pool:
            res = list(pool.map(_one, tasks))
    else:
        res = [_one(t) for t in tasks]
    return np.array([r[0] for r in res]), [r[1] for r in res]


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
    from .quests_stage import available_cpus

    workers = max(1, min(getattr(args, "workers", 1) or 1, available_cpus()))
    out = Path(getattr(args, "output_dir", None) or ".")
    fs.ensure_dir(out)

    ref_frames, ref_idx = _subsample(xyzf_io.read_xyzf(args.reference_xyzf), cap)
    fref, cref = _fingerprints(ref_frames, params, orders, max_clusters, workers)
    result = {"orders": orders, "n_reference": len(ref_frames), "alpha": alpha,
              "clusters_per_frame_reference": {o: float(np.mean([c[o] for c in cref])) for o in orders}}
    arrays = {"reference": fref, "reference_index": np.array(ref_idx)}
    notes = []
    empty = [o for o in orders if all(c[o] < 2 for c in cref)]
    if empty:
        notes.append(f"orders {empty} have <2 clusters in every reference frame (cells too small or cutoffs too short): "
                     "those histogram blocks carry no information")

    if getattr(args, "candidates_xyzf", None):
        cand_frames, cand_idx = _subsample(xyzf_io.read_xyzf(args.candidates_xyzf), cap)
        fcand, ccand = _fingerprints(cand_frames, params, orders, max_clusters, workers)
        arrays.update({"candidates": fcand, "candidates_index": np.array(cand_idx)})
        result["n_candidates"] = len(cand_frames)
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
