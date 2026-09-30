"""Turn one or more fetched pools into a base training set for ChIMES:
merge (only when provenance shows a single level of theory), filter out
unphysical and outlier frames (every removal gets a recorded reason),
measure what ChIMES fitting depends on, optionally subsample to a target
size with farthest-point sampling, split train/holdout, and write
`data_manifest.json` -- the handoff the model-building phase reads.

What gets measured, and why it matters for ChIMES:
  - per element-pair minimum distance and short-range sampling: the inner
    cutoff `s_minim` sits just below the smallest sampled distance, and a
    pair with little short-range data gets a poorly constrained repulsive
    wall;
  - per element-pair frame counts: a pair absent from the data has an
    unconstrained basis;
  - the thinnest periodic cell width: the outer cutoff must stay below half
    of it (times 2*N_LAYERS+1), so small DFT cells limit `s_maxim`;
  - per-atom energy residual after fitting one reference energy per
    element, which makes energies of different compositions comparable for
    outlier detection.

Input pools are `.xyzf` files; a `provenance.json` in the same directory
(written by `data-fetch`) supplies the level of theory and frame ids.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np

from ..data_sources import convert
from ..io import xyzf as xyzf_io
from . import dataset_select
from ._compose import ns

NAME = "data-curate"
SUMMARY = "Filter, analyze, subsample and split fetched pools into a ChIMES base dataset + data_manifest.json."
SCHEMA = {
    "type": "object",
    "required": ["frames", "elements"],
    "properties": {
        "frames": {"type": "array", "items": {"type": "string"}, "description": ".xyzf pools; provenance.json beside each is used when present."},
        "elements": {"type": "array", "items": {"type": "string"}},
        "min_distance_ang": {"type": "number", "default": 0.5, "description": "Drop frames with any interatomic distance below this (unphysical)."},
        "max_force_ev_ang": {"type": "number", "default": 50.0, "description": "Drop frames with any force component norm above this."},
        "energy_outlier_mad": {"type": ["number", "null"], "default": 8.0, "description": "Drop frames whose per-atom energy residual is more than this many scaled MADs from the median (and at least energy_outlier_floor away); null disables."},
        "energy_outlier_floor": {"type": "number", "default": 1.0, "description": "eV/atom. A residual must also be at least this far from the median to count: broken labels are off by ~eV/atom, while a tight dataset's MAD can be small enough to flag legitimate strained frames."},
        "dedupe": {"type": "boolean", "default": True},
        "natoms_min": {"type": ["integer", "null"]},
        "natoms_max": {"type": ["integer", "null"]},
        "require_orthorhombic": {"type": "boolean", "default": False},
        "max_volume_ratio": {"type": ["number", "null"], "default": 3.0, "description": "Drop frames whose volume per atom exceeds this multiple of the pool median: clusters/molecules in vacuum boxes, which general databases include but a bulk model should not fit. null keeps them."},
        "coverage_cutoff_ang": {"type": "number", "default": 6.0, "description": "Pair-distance analysis radius."},
        "target_size": {"type": ["integer", "null"], "description": "Farthest-point subsample to this many frames before splitting."},
        "holdout_fraction": {"type": ["number", "null"], "default": 0.2, "description": "Composition-stratified holdout; null = no split."},
        "split_by": {"type": "string", "enum": ["group", "frame"], "default": "group", "description": "Hold out whole groups of correlated frames (relaxation paths, closely spaced MD frames; see dataset-select) or independent frames. group avoids near-copies of training frames in the holdout, which overstates accuracy."},
        "allow_mixed_theory": {"type": "boolean", "default": False, "description": "Permit merging pools from different datasets or levels of theory. Energies from different DFT setups are not comparable; only use after checking settings match."},
        "seed": {"type": "integer", "default": 42},
    },
}


def _list_arg(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def add_arguments(parser) -> None:
    parser.add_argument("--frames", type=_list_arg, default=None, help="Comma-separated .xyzf pools.")
    parser.add_argument("--elements", type=_list_arg, default=None)
    parser.add_argument("--min-distance-ang", dest="min_distance_ang", type=float, default=0.5)
    parser.add_argument("--max-force-ev-ang", dest="max_force_ev_ang", type=float, default=50.0)
    parser.add_argument("--energy-outlier-mad", dest="energy_outlier_mad", type=float, default=8.0)
    parser.add_argument("--no-energy-outliers", dest="energy_outlier_mad", action="store_const", const=None)
    parser.add_argument("--energy-outlier-floor", dest="energy_outlier_floor", type=float, default=1.0)
    parser.add_argument("--no-dedupe", dest="dedupe", action="store_false", default=True)
    parser.add_argument("--natoms-min", dest="natoms_min", type=int, default=None)
    parser.add_argument("--natoms-max", dest="natoms_max", type=int, default=None)
    parser.add_argument("--require-orthorhombic", dest="require_orthorhombic", action="store_true", default=False)
    parser.add_argument("--max-volume-ratio", dest="max_volume_ratio", type=float, default=3.0)
    parser.add_argument("--keep-vacuum", dest="max_volume_ratio", action="store_const", const=None)
    parser.add_argument("--coverage-cutoff-ang", dest="coverage_cutoff_ang", type=float, default=6.0)
    parser.add_argument("--target-size", dest="target_size", type=int, default=None)
    parser.add_argument("--holdout-fraction", dest="holdout_fraction", type=float, default=0.2)
    parser.add_argument("--split-by", dest="split_by", choices=["group", "frame"], default="group")
    parser.add_argument("--no-holdout", dest="holdout_fraction", action="store_const", const=None)
    parser.add_argument("--allow-mixed-theory", dest="allow_mixed_theory", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)


def pair_key(a: str, b: str) -> str:
    return "-".join(sorted((a, b)))


def _load_pools(paths):
    frames, origins, provenances = [], [], []
    for p in paths:
        p = Path(p)
        prov_path = p.parent / "provenance.json"
        prov = json.loads(prov_path.read_text()) if prov_path.is_file() else None
        pool = xyzf_io.read_xyzf(p)
        ids = (prov or {}).get("frame_ids") or [f"{p.name}#{i}" for i in range(len(pool))]
        if len(ids) != len(pool):
            ids = [f"{p.name}#{i}" for i in range(len(pool))]
        frames.extend(pool)
        origins.extend({"pool": str(p), "index": i, "frame_id": ids[i]} for i in range(len(pool)))
        provenances.append({"pool": str(p), "provenance": str(prov_path) if prov else None,
                            "level_of_theory": (prov or {}).get("level_of_theory"),
                            "label_policy": (prov or {}).get("label_policy", "source"),
                            "source": (prov or {}).get("source"),
                            "license": ((prov or {}).get("source_metadata") or {}).get("license"),
                            "doi": ((prov or {}).get("source_metadata") or {}).get("doi")})
    return frames, origins, provenances


def _theory_key(prov) -> str:
    lot = prov.get("level_of_theory") or {}
    methods = sorted(k for k in (lot.get("methods") or {}) if k)
    software = sorted(k for k in (lot.get("software") or {}) if k)
    if not methods:
        return "unspecified"
    return " | ".join(methods) + (f" ({', '.join(software)})" if software else "")


def check_theory(provenances, allow_mixed: bool):
    keys = {_theory_key(p) for p in provenances}
    policies = {p["label_policy"] for p in provenances}
    if len(policies) > 1:
        raise ValueError(f"cannot merge labeled and structure-only pools: {policies}")
    labeled = policies == {"source"}
    if labeled and len(keys) > 1 and not allow_mixed:
        raise ValueError(
            f"pools come from different levels of theory {sorted(keys)}. Energies from different DFT "
            "setups have different references and must not share one ChIMES fit. Curate them "
            "separately, relabel the structures with one method (label_policy=relabel + qe-relabel), "
            "or pass allow_mixed_theory to override."
        )
    sources = {p.get("source") for p in provenances}
    if labeled and len(sources) > 1 and not allow_mixed:
        raise ValueError(
            f"pools come from different datasets {sorted(map(str, sources))}. Even with the same functional "
            "label, datasets differ in DFT settings (code, pseudopotentials, cutoffs, smearing), which shifts "
            "absolute energies. Fit one source, relabel structures with one method, or pass "
            "allow_mixed_theory if you have checked the settings match (and consider fitener=false)."
        )
    if labeled and "unspecified" in keys and len(provenances) > 1 and not allow_mixed:
        raise ValueError("a pool has no recorded level of theory, so merging it cannot be checked; "
                         "fetch it with --level-of-theory or pass allow_mixed_theory")
    return sorted(keys), labeled


def _geometry(frame, cutoff):
    """(pair -> distances array, min distance or None, thinnest cell width, volume per atom)."""
    atoms = convert.frame_to_atoms(frame)
    i, j, d = convert.unique_pairs(atoms, cutoff)
    sym = np.asarray(frame.symbols)
    pairs = defaultdict(list)
    for a, b, dist in zip(sym[i], sym[j], d):
        pairs[pair_key(a, b)].append(dist)
    cell = atoms.cell[:]
    vol = abs(np.linalg.det(cell))
    widths = [vol / np.linalg.norm(np.cross(cell[(k + 1) % 3], cell[(k + 2) % 3])) for k in range(3)]
    return ({k: np.asarray(v) for k, v in pairs.items()}, (float(d.min()) if len(d) else None),
            float(min(widths)), vol / len(atoms))


def nlayers_required(s_maxim: float, thinnest_width: float) -> int:
    """Smallest N_LAYERS with s_maxim <= (2*N_LAYERS+1) * width / 2 -- the
    periodic-image condition chimes_lsq enforces (ClassDefs.C IS_RCUT_SAFE),
    written with the perpendicular width, which is the conservative bound
    for tilted cells."""
    import math

    return max(0, math.ceil((2 * s_maxim / thinnest_width - 1) / 2))


def reference_energies(frames, elements):
    """Least-squares per-element reference energies (eV/atom) and each frame's
    per-atom residual (eV/atom) -- removes the composition trend so frames of
    different stoichiometry can be compared."""
    idx = {e: k for k, e in enumerate(elements)}
    A = np.zeros((len(frames), len(elements)))
    y = np.zeros(len(frames))
    for r, f in enumerate(frames):
        for s in f.symbols:
            A[r, idx[s]] += 1
        y[r] = convert.energy_ev(f)
    mu, *_ = np.linalg.lstsq(A, y, rcond=None)
    natoms = A.sum(axis=1)
    return {e: float(mu[idx[e]]) for e in elements}, (y - A @ mu) / natoms


def robust_outliers(values, k, floor=0.0):
    med = np.median(values)
    mad = 1.4826 * np.median(np.abs(values - med))
    return np.abs(values - med) > max(k * mad, floor)


def _dedupe_key(frame, decimals=3):
    box = np.round(np.asarray(frame.box, dtype=float), decimals).tobytes()
    pos = np.round(np.asarray(frame.positions, dtype=float), decimals).tobytes()
    return (tuple(frame.symbols), box, pos)


def run(args) -> dict:
    paths = getattr(args, "frames", None)
    target = getattr(args, "elements", None)
    if not paths or not target:
        raise ValueError("data-curate requires --frames and --elements")
    target = sorted(target)
    seed = getattr(args, "seed", 42)

    frames, origins, provenances = _load_pools(paths)
    theory_keys, labeled = check_theory(provenances, getattr(args, "allow_mixed_theory", False))

    removed = []
    keep = np.ones(len(frames), dtype=bool)

    def drop(k, reason):
        if keep[k]:
            keep[k] = False
            removed.append({**origins[k], "reason": reason})

    tset = set(target)
    for k, f in enumerate(frames):
        if not set(f.symbols) <= tset:
            drop(k, "elements_outside_target")
        elif getattr(args, "natoms_min", None) and f.natoms < args.natoms_min:
            drop(k, "too_few_atoms")
        elif getattr(args, "natoms_max", None) and f.natoms > args.natoms_max:
            drop(k, "too_many_atoms")
        elif getattr(args, "require_orthorhombic", False) and f.non_ortho:
            drop(k, "non_orthorhombic")

    if getattr(args, "dedupe", True):
        seen = {}
        for k in np.flatnonzero(keep):
            key = _dedupe_key(frames[k])
            if key in seen:
                drop(k, f"duplicate_of:{origins[seen[key]]['frame_id']}")
            else:
                seen[key] = k

    cutoff = max(getattr(args, "coverage_cutoff_ang", 6.0), getattr(args, "min_distance_ang", 0.5))
    geom = {}
    for k in np.flatnonzero(keep):
        geom[k] = _geometry(frames[k], cutoff)
        dmin = geom[k][1]
        if dmin is None:
            drop(k, f"isolated_atom (no neighbour within {cutoff} A)")
        elif dmin < getattr(args, "min_distance_ang", 0.5):
            drop(k, f"min_distance {dmin:.3f} A")

    vratio = getattr(args, "max_volume_ratio", 3.0)
    if vratio is not None:
        vpa = {k: geom[k][3] for k in np.flatnonzero(keep)}
        if vpa:
            median = float(np.median(list(vpa.values())))
            for k, v in vpa.items():
                if v > vratio * median:
                    drop(k, f"low_density {v:.1f} A^3/atom vs median {median:.1f} (vacuum/cluster)")

    max_f = getattr(args, "max_force_ev_ang", 50.0)
    force_norms = {}
    if labeled:
        for k in np.flatnonzero(keep):
            fmax = float(np.linalg.norm(convert.forces_ev_ang(frames[k]), axis=1).max())
            force_norms[k] = fmax
            if max_f is not None and fmax > max_f:
                drop(k, f"max_force {fmax:.1f} eV/A")

    ref_mu, residual_stats = None, None
    energy_idx = [k for k in np.flatnonzero(keep) if frames[k].energy is not None] if labeled else []
    if energy_idx:
        ref_mu, resid = reference_energies([frames[k] for k in energy_idx], target)
        k_mad = getattr(args, "energy_outlier_mad", 8.0)
        if k_mad is not None:
            for k, bad, r in zip(energy_idx, robust_outliers(resid, k_mad, getattr(args, "energy_outlier_floor", 1.0)), resid):
                if bad:
                    drop(k, f"energy_outlier residual {r:+.3f} eV/atom")
        kept_resid = resid[[keep[k] for k in energy_idx]]
        if len(kept_resid):
            residual_stats = {q: float(np.percentile(kept_resid, p)) for q, p in (("p01", 1), ("p50", 50), ("p99", 99))}
            residual_stats["std"] = float(kept_resid.std())

    kept = list(np.flatnonzero(keep))
    if not kept:
        raise ValueError(f"every frame was filtered out: {dict(Counter(r['reason'].split()[0] for r in removed))}")

    out = Path(getattr(args, "output_dir", None) or ".")
    out.mkdir(parents=True, exist_ok=True)

    target_size = getattr(args, "target_size", None)
    selection_note = None
    if target_size and target_size < len(kept):
        filtered = out / "filtered.xyzf"
        xyzf_io.write_xyzf([frames[k] for k in kept], filtered)
        sel = dataset_select.run(ns(frames=str(filtered), method="fps", n_select=target_size, holdout_fraction=None,
                                    seed=seed, descriptor="composition_energy" if energy_idx else "composition",
                                    output_dir=str(out / "_fps")))
        kept = [kept[i] for i in sel["selected_indices"]]
        selection_note = f"farthest-point sampled {target_size} of {sel['n_pool']} filtered frames"

    curated_frames = [frames[k] for k in kept]
    curated = out / "curated.xyzf"
    xyzf_io.write_xyzf(curated_frames, curated)

    pairs = {}
    all_pairs = [pair_key(a, b) for a, b in combinations_with_replacement(target, 2)]
    for pk in all_pairs:
        dists = [geom[k][0][pk] for k in kept if pk in geom[k][0]]
        n_frames = len(dists)
        d = np.concatenate(dists) if dists else np.array([])
        pairs[pk] = {
            "n_frames": n_frames,
            "n_distances": int(d.size),
            "min_distance": float(d.min()) if d.size else None,
            "p01_distance": float(np.percentile(d, 1)) if d.size else None,
            "n_within_1.2x_min": int((d < 1.2 * d.min()).sum()) if d.size else 0,
        }
    widths = [geom[k][2] for k in kept]

    warnings = []
    for pk, s in pairs.items():
        if s["n_frames"] == 0:
            warnings.append(f"pair {pk} never occurs within {cutoff} A: its ChIMES terms are unconstrained")
        elif s["n_within_1.2x_min"] < 20:
            warnings.append(f"pair {pk}: little short-range data ({s['n_within_1.2x_min']} distances near its minimum); the repulsive wall is poorly sampled")
    need8 = nlayers_required(8.0, min(widths))
    if need8 > 0:
        warnings.append(f"thinnest cell width {min(widths):.2f} A: an 8 A outer cutoff needs N_LAYERS >= {need8} in fm_setup.in (see fit_hints.nlayers_required)")
    n_non_ortho = sum(1 for f in curated_frames if f.non_ortho)
    if n_non_ortho:
        warnings.append(f"{n_non_ortho} non-orthorhombic frames: chimes_lsq accepts them, but auto-build's cutoff derivation and lammps-run data files are orthorhombic-only")
    if not labeled:
        warnings.append("structure-only set: label with qe-relabel before fitting")

    split = None
    holdout_fraction = getattr(args, "holdout_fraction", 0.2)
    if holdout_fraction:
        s = dataset_select.run(ns(frames=str(curated), method="stratified_holdout", n_select=None,
                                  holdout_fraction=holdout_fraction, seed=seed, descriptor="composition",
                                  split_by=getattr(args, "split_by", "group") or "group", group_rmsd=0.3,
                                  group_cell_tol=0.03, output_dir=str(out / "_split")))
        train_path, holdout_path = out / "train.xyzf", out / "holdout.xyzf"
        xyzf_io.write_xyzf([curated_frames[i] for i in s["selected_indices"]], train_path)
        xyzf_io.write_xyzf([curated_frames[i] for i in s["holdout_indices"]], holdout_path)
        split = {"train_xyzf": str(train_path), "holdout_xyzf": str(holdout_path),
                 "n_train": s["n_selected"], "n_holdout": s["n_holdout"], "split_by": s.get("split_by"),
                 "n_groups": (s.get("groups") or {}).get("n_groups")}
        warnings.extend((s.get("groups") or {}).get("notes", []))

    removal_counts = dict(Counter(r["reason"].split(":")[0].split()[0] for r in removed))
    fvals = [force_norms[k] for k in kept if k in force_norms]
    report = {
        "n_input": len(frames),
        "n_removed": len(removed),
        "removed_by_reason": removal_counts,
        "removed": removed,
        "selection": selection_note,
        "kept_frame_ids": [origins[k]["frame_id"] for k in kept],
        "reference_energies_ev_per_atom": ref_mu,
    }
    report_path = out / "curation_report.json"
    report_path.write_text(json.dumps(report, indent=1))

    summary = {
        "n_frames": len(kept),
        "compositions": dict(Counter("-".join(sorted(set(f.symbols))) for f in curated_frames).most_common()),
        "natoms_range": [min(f.natoms for f in curated_frames), max(f.natoms for f in curated_frames)],
        "n_non_orthorhombic": n_non_ortho,
        "thinnest_cell_width_ang": float(min(widths)),
        "max_force_ev_ang": {"p50": float(np.percentile(fvals, 50)), "p99": float(np.percentile(fvals, 99)), "max": float(max(fvals))} if fvals else None,
        "energy_residual_ev_per_atom": residual_stats,
    }
    manifest = {
        "schema_version": 1,
        "elements": target,
        "labeled": labeled,
        "level_of_theory": theory_keys,
        "units": {"energy": "kcal/mol", "forces": "hartree/bohr", "positions": "angstrom"},
        "curated_xyzf": str(curated),
        **(split or {}),
        "summary": summary,
        "pairs": pairs,
        "fit_hints": {
            "s_minim_upper_bound": {pk: s["min_distance"] for pk, s in pairs.items() if s["min_distance"] is not None},
            "nlayers_required": {f"{r:.1f}": nlayers_required(r, min(widths)) for r in (4.0, 6.0, 8.0)},
            "fitener": bool(energy_idx),
            "fitstrs": False,
        },
        "warnings": warnings,
        "sources": provenances,
        "curation_report": str(report_path),
    }
    manifest_path = out / "data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1))

    return {
        "data_manifest": str(manifest_path),
        "curated_xyzf": str(curated),
        **(split or {}),
        "n_input": len(frames),
        "n_frames": len(kept),
        "removed_by_reason": removal_counts,
        "level_of_theory": theory_keys,
        "summary": summary,
        "pairs": pairs,
        "warnings": warnings,
    }
