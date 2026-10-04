"""Assemble one active-learning batch to label, from candidate frames.

Published ChIMES active learning labels frames that are (a) close contacts
the model has not seen and (b) otherwise representative of the MD it will
run (Lindsey et al. 2020, 2025). Two more signals are available here:
structural novelty against the training set, and model uncertainty. This
stage combines them into one batch within a labeling budget:

1. **Close contacts** (`md-check` reports them; recomputed here from the
   model's inner cutoffs when candidates come from elsewhere): frames with
   a pair inside, or within `close_margin` of, an inner cutoff. Up to
   `min_close` of them are taken first; they are what makes MD stable.
2. **Novelty**: QUESTS dH (per environment) and the ChIMES cluster-graph
   fingerprint D_j^2 (per frame), when available.
3. **Uncertainty**: bootstrap-committee force spread (`committee` stage),
   when `--fm-setup-in` is given.

Each candidate gets the mean of its normalized ranks over the available
signals (0 = least interesting, 1 = most). Near-duplicate candidates
(dataset_select.correlated_groups) are collapsed to their best member, and
frames identical to training frames are dropped. The best `--budget`
frames become `batch.xyzf`, with `batch.json` recording every score, so the
choice is inspectable. Label the batch with `qe-relabel` using the base
set's settings, then `al-merge` it.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..io import atomic, fs
from ..io import xyzf as xyzf_io

NAME = "al-batch"
SUMMARY = "One active-learning batch: close contacts + novelty (QUESTS, fingerprint) + committee uncertainty, deduplicated, within a budget."
SCHEMA = {
    "type": "object",
    "required": ["candidates_xyzf", "train_xyzf", "params"],
    "properties": {
        "candidates_xyzf": {"type": "string", "description": "Candidate frames (e.g. md-check harvest.xyzf)."},
        "train_xyzf": {"type": "string", "description": "Current training set (novelty reference; duplicates are dropped)."},
        "params": {"type": "string", "description": "Current model (inner cutoffs for close contacts; fingerprint clusters)."},
        "fm_setup_in": {"type": ["string", "null"], "description": "Current basis: enables committee uncertainty (adds a few fits)."},
        "budget": {"type": "integer", "default": 40},
        "min_close": {"type": "integer", "default": 10, "description": "Close-contact frames taken first (at most)."},
        "close_margin": {"type": "number", "default": 0.1},
        "signals": {"type": "array", "items": {"type": "string"}, "default": ["quests", "fingerprint", "committee"], "description": "Which signals to combine (missing tools are skipped with a note)."},
        "n_models": {"type": "integer", "default": 4, "description": "Committee size."},
        "algorithm": {"type": "string", "default": "lassolars"},
        "alpha": {"type": "number", "default": 1e-5},
        "seed": {"type": "integer", "default": 0},
        "workers": {"type": "integer", "default": 0, "description": "Parallel fingerprint workers (default: all cores)."},
        "max_fingerprint_frames": {"type": "integer", "default": 40, "description": "Training frames used as the fingerprint reference (cluster enumeration is the slow part)."},
        "structure_weight": {"type": ["number", "null"], "description": "Element-aware fingerprint (the paper's alpha; 0.25 suits alloys). Default: type-agnostic."},
    },
}


def add_arguments(parser) -> None:
    parser.add_argument("--candidates-xyzf", dest="candidates_xyzf", default=None)
    parser.add_argument("--train-xyzf", dest="train_xyzf", default=None)
    parser.add_argument("--params", default=None)
    parser.add_argument("--fm-setup-in", dest="fm_setup_in", default=None)
    parser.add_argument("--budget", type=int, default=40)
    parser.add_argument("--min-close", dest="min_close", type=int, default=10)
    parser.add_argument("--close-margin", dest="close_margin", type=float, default=0.1)
    parser.add_argument("--signals", type=lambda s: [x for x in s.split(",") if x], default=None)
    parser.add_argument("--n-models", dest="n_models", type=int, default=4)
    parser.add_argument("--algorithm", default="lassolars")
    parser.add_argument("--alpha", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-fingerprint-frames", dest="max_fingerprint_frames", type=int, default=40)
    parser.add_argument("--structure-weight", dest="structure_weight", type=float, default=None)
    parser.add_argument("--workers", type=int, default=0)


def _ranks(values) -> np.ndarray:
    """Normalized ranks in [0, 1] (1 = largest), NaN-safe."""
    v = np.asarray(values, float)
    out = np.zeros(len(v))
    ok = np.isfinite(v)
    if ok.sum() > 1:
        order = np.argsort(v[ok])
        r = np.empty(ok.sum())
        r[order] = np.arange(ok.sum())
        out[ok] = r / (ok.sum() - 1)
    return out


def run(args) -> dict:
    from ._compose import ns
    from .data_curate import _dedupe_key
    from .dataset_select import correlated_groups
    from .evaluate import _closest, check_frames_against_models
    from .md_check import pair_inner_cutoffs

    for req in ("candidates_xyzf", "train_xyzf", "params"):
        if not getattr(args, req, None):
            raise ValueError(f"al-batch needs --{req.replace('_', '-')}")
    out = fs.ensure_dir(Path(getattr(args, "output_dir", None) or ".").resolve())
    cands = xyzf_io.read_xyzf(args.candidates_xyzf)
    train = xyzf_io.read_xyzf(args.train_xyzf)
    params = str(Path(args.params).resolve())
    check_frames_against_models(cands, [params])
    notes = []

    # drop exact duplicates of training frames
    seen = {_dedupe_key(f) for f in train}
    keep = [i for i, f in enumerate(cands) if _dedupe_key(f) not in seen]
    if len(keep) < len(cands):
        notes.append(f"{len(cands) - len(keep)} candidate(s) identical to training frames dropped")
    n = len(keep)
    if n == 0:
        raise ValueError("no candidates left after removing duplicates of training frames")
    sub = [cands[i] for i in keep]

    # close contacts
    inner = pair_inner_cutoffs(params)
    margin = float(getattr(args, "close_margin", 0.1) or 0.1)
    closest = [_closest(f, rmax=max(inner.values()) + margin + 0.5) for f in sub]
    below = np.array([any(r < inner.get(k, 0.0) for k, r in c.items()) for c in closest])
    close = np.array([any(r < inner.get(k, 0.0) + margin for k, r in c.items()) for c in closest])
    gap = np.array([min((r - inner.get(k, 0.0) for k, r in c.items()), default=np.inf) for c in closest])

    signals = list(getattr(args, "signals", None) or ["quests", "fingerprint", "committee"])
    scores = {}
    if "quests" in signals:
        try:
            from quests.entropy import DEFAULT_BANDWIDTH, DEFAULT_BATCH, delta_entropy

            from .quests_stage import descriptors, frame_scores

            species = sorted({s for f in train for s in f.symbols})
            X, _ = descriptors(train, species=species)
            Y, owner = descriptors(sub, species=species)
            dh = delta_entropy(Y, X, h=DEFAULT_BANDWIDTH, batch_size=DEFAULT_BATCH)
            scores["quests_dH_max"] = frame_scores(dh, owner, n, "max")
        except ImportError:
            notes.append("quests not installed: QUESTS novelty skipped (pip install quests)")
    if "fingerprint" in signals:
        from ..io import fingerprint as fp

        model = fp.ModelCutoffs(params)
        orders = [2] + ([3] if "TRIPLETTYPE" in Path(params).read_text() else [])
        # cluster enumeration is the expensive part (tiny cells replicate to hundreds of atoms; ~20 s per frame with
        # 3-body clusters at 7 A): cap the reference and fingerprint frames in parallel
        cap = int(getattr(args, "max_fingerprint_frames", 40) or 40)
        ref_idx = np.linspace(0, len(train) - 1, min(cap, len(train))).round().astype(int)
        try:
            from ..io.pool import process_pool
            from functools import partial

            from .fingerprint import _one

            from .quests_stage import available_cpus

            workers = max(1, min(int(getattr(args, "workers", 0) or available_cpus()), 32))
            sw = [getattr(args, "structure_weight", None)]
            tasks = [(train[i], params, orders, 1500, sw) for i in ref_idx] + [(f, params, orders, 1500, sw) for f in sub]
            with process_pool(min(workers, len(tasks))) as pool:
                fps = [r[0] for r in pool.map(_one, tasks)]
            F_ref, F_c = np.array(fps[:len(ref_idx)]), np.array(fps[len(ref_idx):])
            nov = fp.novelty(F_ref, F_c)
            scores["fingerprint_Dj2"] = np.asarray(nov["Dj2"])
            if nov["dof"] < 3:
                notes.append(f"fingerprint reference covariance has rank {nov['dof']} ({len(ref_idx)} frames): D_j^2 ranks "
                             "candidates but carries little information; give al-batch a larger training set")
            if sw[0] is not None:
                notes.append(f"fingerprint novelty uses the element-aware metric (structure weight {sw[0]:g})")
                if len(set(model.descriptor.values())) < 2:
                    notes.append("single element descriptor value: the composition term is zero and the metric is structure only")
            if len(ref_idx) < len(train):
                notes.append(f"fingerprint reference subsampled to {len(ref_idx)} of {len(train)} training frames "
                             "(--max-fingerprint-frames); run the fingerprint stage with --machine for the full set")
        except ValueError as exc:
            notes.append(f"fingerprint novelty skipped: {exc}")
    if "committee" in signals and getattr(args, "fm_setup_in", None):
        from . import committee

        cdir = out / "committee"
        xyzf_io.write_xyzf(sub, out / "candidates.dedup.xyzf")
        c = committee.run(ns(fm_setup_in=args.fm_setup_in, algorithm=args.algorithm, alpha=args.alpha,
                             n_models=int(getattr(args, "n_models", 4) or 4), seed=int(getattr(args, "seed", 0) or 0),
                             candidates_xyzf=str(out / "candidates.dedup.xyzf"), n_select=n, output_dir=str(cdir)))
        spread = np.full(n, np.nan)
        for s in c["selected"]:
            spread[s["frame"]] = s["force_spread"]
        scores["committee_force_spread"] = spread
    elif "committee" in signals:
        notes.append("no --fm-setup-in: committee uncertainty skipped")

    combined = np.mean([_ranks(v) for v in scores.values()], axis=0) if scores else np.zeros(n)
    combined = np.where(close, np.maximum(combined, 0.5), combined)  # close contacts never rank below the median

    # collapse near-duplicate candidates to their best member
    groups = correlated_groups(sub)
    best_in_group = {}
    for i, g in enumerate(groups):
        if g not in best_in_group or combined[i] > combined[best_in_group[g]]:
            best_in_group[g] = i
    reps = set(best_in_group.values())
    n_collapsed = n - len(reps)

    budget = int(getattr(args, "budget", 40) or 40)
    min_close = int(getattr(args, "min_close", 10) or 0)
    order = sorted(reps, key=lambda i: -combined[i])
    chosen = [i for i in sorted(reps, key=lambda i: gap[i]) if close[i]][:min_close]
    for i in order:
        if len(chosen) >= budget:
            break
        if i not in chosen:
            chosen.append(i)
    chosen = chosen[:budget]
    xyzf_io.write_xyzf([sub[i] for i in chosen], out / "batch.xyzf")
    rows = [{"candidate": keep[i], "combined": round(float(combined[i]), 4), "close_contact": bool(close[i]),
             "below_inner_cutoff": bool(below[i]), "gap_to_inner_cutoff": round(float(gap[i]), 3) if np.isfinite(gap[i]) else None,
             **{k: (round(float(v[i]), 4) if np.isfinite(v[i]) else None) for k, v in scores.items()},
             "selected": i in set(chosen)} for i in range(n)]
    result = {"n_candidates": len(cands), "n_after_dedup": n, "n_near_duplicate_collapsed": n_collapsed,
              "signals_used": sorted(scores), "n_close_contact": int(close.sum()), "n_below_inner_cutoff": int(below.sum()),
              "budget": budget, "n_selected": len(chosen), "n_selected_close_contact": int(sum(close[i] for i in chosen)),
              "batch_xyzf": str(out / "batch.xyzf"), "candidates": sorted(rows, key=lambda r: -r["combined"]), "notes": notes}
    atomic.write_json(out / "batch.json", result, indent=1)
    return {k: v for k, v in result.items() if k != "candidates"} | {"batch_json": str(out / "batch.json")}
