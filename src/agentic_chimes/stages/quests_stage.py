"""QUESTS: information-theoretic coverage, novelty and selection.

QUESTS (Quick Uncertainty and Entropy via STructural Similarity; Schwalbe-Koda
et al., 2024, `pip install quests`) describes every atomic environment by its
sorted neighbor distances and the distances between those neighbors, then
treats a dataset as a kernel density in that space:

- **entropy H** of a dataset: how much structural information it holds
  (nats). Saturation of H with dataset size means more of the same data adds
  nothing;
- **diversity**: an estimate of the number of distinct environments;
- **differential entropy dH(y | X)** of an environment y against a dataset
  X: how far y lies from everything in X. dH <= 0 means y is at least as well
  covered as X's own environments; large dH means novel.

This stage complements the ChIMES cluster-graph `fingerprint` (which compares
whole configurations through the model's own clusters): QUESTS is per
environment, model-free, and gives a selection criterion. It is used for:

1. **coverage of a dataset** (`--reference-xyzf` alone): H, diversity, dH of
   the reference against itself, and an entropy curve H(n) over nested
   subsets (saturation check);
2. **novelty of candidates** (`--candidates-xyzf`): per-atom dH against the
   reference, per-frame max/mean, a data-driven threshold (the reference's own
   99th percentile of self-dH), `novel.xyzf`;
3. **selection** (`--select n`): greedy entropy-maximizing choice of n
   candidate frames (each step adds the frame whose environments are least
   covered by reference + already selected), written to `selected.xyzf`.

Descriptor: `single` (species-agnostic, QUESTS' published form) or `multi`
(per-species blocks, marked experimental upstream). Bandwidth and k/cutoff
follow QUESTS' defaults.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..io import atomic, fs
from ..io import xyzf as xyzf_io

NAME = "quests"
SUMMARY = "QUESTS entropy/diversity of a dataset, per-environment novelty (dH) of candidates, entropy-maximizing selection."
SCHEMA = {
    "type": "object",
    "required": ["reference_xyzf"],
    "properties": {
        "reference_xyzf": {"type": "string"},
        "candidates_xyzf": {"type": ["string", "null"]},
        "descriptor": {"type": "string", "enum": ["single", "multi"], "default": "single"},
        "k": {"type": "integer", "default": 32, "description": "Neighbors per environment (QUESTS default 32)."},
        "cutoff": {"type": "number", "default": 5.0, "description": "Å, weight-function cutoff (QUESTS default 5)."},
        "bandwidth": {"type": ["number", "null"], "description": "Kernel bandwidth; default QUESTS' (0.015)."},
        "max_frames": {"type": ["integer", "null"], "default": 2000, "description": "Cap per set (evenly spaced)."},
        "select": {"type": ["integer", "null"], "description": "Pick this many candidate frames by greedy entropy gain."},
        "dh_threshold": {"type": ["number", "null"], "description": "dH above which an environment is novel. Default: the reference's own 99th-percentile self-dH."},
        "rank_by": {"type": "string", "enum": ["max", "mean"], "default": "max", "description": "Frame novelty = max (most novel environment) or mean dH over its atoms."},
        "entropy_curve": {"type": "boolean", "default": True, "description": "H over nested subsets of the reference (12.5 % … 100 %)."},
        "plot": {"type": "boolean", "default": True},
    },
}


def add_arguments(parser) -> None:
    parser.add_argument("--reference-xyzf", dest="reference_xyzf", default=None)
    parser.add_argument("--candidates-xyzf", dest="candidates_xyzf", default=None)
    parser.add_argument("--descriptor", choices=["single", "multi"], default="single")
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--bandwidth", type=float, default=None)
    parser.add_argument("--max-frames", dest="max_frames", type=int, default=2000)
    parser.add_argument("--select", type=int, default=None)
    parser.add_argument("--dh-threshold", dest="dh_threshold", type=float, default=None)
    parser.add_argument("--rank-by", dest="rank_by", choices=["max", "mean"], default="max")
    parser.add_argument("--no-entropy-curve", dest="entropy_curve", action="store_false", default=True)
    parser.add_argument("--no-plot", dest="plot", action="store_false", default=True)


def available_cpus() -> int:
    """CPUs this process may use (the Slurm/cgroup affinity mask), not the node's core count."""
    import os

    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def _quests():
    try:
        import numba
        from quests import descriptor, entropy
    except ImportError as exc:
        raise ImportError("QUESTS is not installed: pip install quests (or pip install -e '.[quests]')") from exc
    # numba defaults to the node's core count; inside a 4-core allocation that is 112 threads on 4 cores
    try:
        numba.set_num_threads(min(available_cpus(), numba.config.NUMBA_NUM_THREADS))
    except (ValueError, AttributeError):
        pass
    return descriptor, entropy


def _subsample(frames, cap):
    if cap and len(frames) > cap:
        idx = np.linspace(0, len(frames) - 1, cap).round().astype(int)
        return [frames[i] for i in idx], idx.tolist()
    return frames, list(range(len(frames)))


def descriptors(frames, *, kind="single", k=32, cutoff=5.0, species=None):
    """(X, frame index per row) for a list of xyzf frames."""
    from ..data_sources import convert

    d, _ = _quests()
    atoms = [convert.frame_to_atoms(f) for f in frames]
    if kind == "multi":
        X = d.get_descriptors_multicomponent(atoms, k=k, cutoff=cutoff, species=species)
    else:
        X = d.get_descriptors(atoms, k=k, cutoff=cutoff)
    owner = np.concatenate([np.full(f.natoms, i) for i, f in enumerate(frames)])
    return X, owner


def frame_scores(dh, owner, n_frames, how="max"):
    out = np.zeros(n_frames)
    for i in range(n_frames):
        v = dh[owner == i]
        out[i] = (v.max() if how == "max" else v.mean()) if len(v) else np.nan
    return out


def greedy_select(X_ref, X_cand, owner, n_frames, n_select, h, batch) -> list:
    """Greedy entropy-maximizing frames: at each step add the candidate frame
    with the largest mean dH of its environments against reference + selected."""
    _, e = _quests()
    ref = X_ref
    chosen = []
    remaining = set(range(n_frames))
    for _ in range(min(n_select, n_frames)):
        dh = e.delta_entropy(X_cand, ref, h=h, batch_size=batch)
        scores = frame_scores(dh, owner, n_frames, "mean")
        for i in chosen:
            scores[i] = -np.inf
        best = int(np.nanargmax(scores))
        if not np.isfinite(scores[best]):
            break
        chosen.append(best)
        remaining.discard(best)
        ref = np.vstack([ref, X_cand[owner == best]])
    return chosen


def run(args) -> dict:
    _, e = _quests()
    from quests.entropy import DEFAULT_BANDWIDTH, DEFAULT_BATCH

    if not getattr(args, "reference_xyzf", None):
        raise ValueError("quests needs --reference-xyzf")
    h = float(getattr(args, "bandwidth", None) or DEFAULT_BANDWIDTH)
    k, cutoff = int(getattr(args, "k", 32) or 32), float(getattr(args, "cutoff", 5.0) or 5.0)
    kind = getattr(args, "descriptor", "single") or "single"
    cap = getattr(args, "max_frames", 2000)
    out = fs.ensure_dir(Path(getattr(args, "output_dir", None) or ".").resolve())

    ref_frames, _ = _subsample(xyzf_io.read_xyzf(args.reference_xyzf), cap)
    species = sorted({s for f in ref_frames for s in f.symbols})
    X, owner_ref = descriptors(ref_frames, kind=kind, k=k, cutoff=cutoff, species=species)
    H, div, dH_self = e.get_all_metrics(X, h=h, batch_size=DEFAULT_BATCH)
    thr = getattr(args, "dh_threshold", None)
    thr = float(thr) if thr is not None else float(np.percentile(dH_self, 99))
    result = {"descriptor": kind, "k": k, "cutoff": cutoff, "bandwidth": h, "n_reference_frames": len(ref_frames),
              "n_reference_environments": int(len(X)), "entropy": round(float(H), 4), "diversity": round(float(div), 4),
              "self_dH_quantiles": {q: round(float(np.percentile(dH_self, q)), 3) for q in (50, 90, 99)},
              "dh_threshold": round(thr, 3), "notes": []}
    if getattr(args, "dh_threshold", None) is None and (len(ref_frames) < 10 or thr <= 0):
        result["notes"].append(f"reference of {len(ref_frames)} frame(s): its 99th-percentile self-dH ({thr:.3f}) is not a "
                               "reliable novelty threshold" + (" and is <= 0, so every candidate environment counts as novel"
                                                                if thr <= 0 else "") + "; pass --dh-threshold (dH > 0 means "
                               "an environment the reference lacks) or use a larger reference")

    if getattr(args, "entropy_curve", True) and len(ref_frames) >= 8:
        from .learning_curve import nested_subsets

        curve = []
        for frac, idx in nested_subsets(ref_frames, [0.125, 0.25, 0.5, 0.75, 1.0], 0):
            Xs = X[np.isin(owner_ref, idx)]
            curve.append({"fraction": frac, "n_frames": len(idx), "entropy": round(float(e.entropy(Xs, h=h, batch_size=DEFAULT_BATCH)), 4)})
        result["entropy_curve"] = curve
        if len(curve) >= 3:
            gain = curve[-1]["entropy"] - curve[-2]["entropy"]
            result["entropy_saturated"] = bool(abs(gain) < 0.02)
            result["notes"].append(("entropy has saturated: the last 25 % of frames added "
                                    f"{gain:+.3f} nats; more of the same kind of data adds little information")
                                   if result["entropy_saturated"] else
                                   f"entropy still rising ({gain:+.3f} nats for the last 25 % of frames): the data is not saturated")

    arrays = {"reference_dH": dH_self}
    cand = getattr(args, "candidates_xyzf", None)
    if cand:
        cand_frames, cand_idx = _subsample(xyzf_io.read_xyzf(cand), cap)
        Y, owner = descriptors(cand_frames, kind=kind, k=k, cutoff=cutoff, species=species)
        dH = e.delta_entropy(Y, X, h=h, batch_size=DEFAULT_BATCH)
        how = getattr(args, "rank_by", "max") or "max"
        fscore = frame_scores(dH, owner, len(cand_frames), how)
        fmean = frame_scores(dH, owner, len(cand_frames), "mean")
        novel = [i for i in range(len(cand_frames)) if fscore[i] > thr]
        H_union = float(e.entropy(np.vstack([X, Y]), h=h, batch_size=DEFAULT_BATCH))
        result.update({
            "n_candidate_frames": len(cand_frames), "n_candidate_environments": int(len(Y)),
            "candidate_dH_quantiles": {q: round(float(np.percentile(dH, q)), 3) for q in (50, 90, 99)},
            "fraction_novel_environments": round(float(np.mean(dH > thr)), 4),
            "fraction_novel_frames": round(len(novel) / len(cand_frames), 4) if cand_frames else None,
            "entropy_gain_if_all_added": round(H_union - float(H), 4),
            "frames": [{"frame": cand_idx[i], "dH_max": round(float(fscore[i] if how == "max" else dH[owner == i].max()), 3),
                        "dH_mean": round(float(fmean[i]), 3), "novel": i in set(novel)} for i in range(len(cand_frames))],
        })
        arrays["candidate_dH"] = dH
        if novel:
            xyzf_io.write_xyzf([cand_frames[i] for i in novel], out / "novel.xyzf")
            result["novel_xyzf"] = str(out / "novel.xyzf")
        n_sel = getattr(args, "select", None)
        if n_sel:
            chosen = greedy_select(X, Y, owner, len(cand_frames), int(n_sel), h, DEFAULT_BATCH)
            xyzf_io.write_xyzf([cand_frames[i] for i in chosen], out / "selected.xyzf")
            result.update({"selected_xyzf": str(out / "selected.xyzf"), "selected_candidate_indices": [cand_idx[i] for i in chosen]})
        result["notes"].append(
            f"{len(novel)} of {len(cand_frames)} candidate frames hold environments above the novelty threshold "
            f"(dH > {thr:.2f}); adding all candidates would raise the reference entropy by {H_union - float(H):+.3f} nats")
        if getattr(args, "plot", True):
            from ..io import plots

            png = plots.histograms({"reference (self)": dH_self, "candidates": dH}, out / "quests_dH.png",
                                   xlabel="dH (nats)", title="QUESTS differential entropy vs the reference set")
            if png:
                result["plot"] = png
    np.savez(out / "quests.npz", **arrays)
    atomic.write_json(out / "quests.json", result, indent=1)
    return result
