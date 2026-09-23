"""FPS / random / stratified-holdout sampling over a `.xyzf` frame pool,
formalizing prior hand-rolled `make_holdout_split.py`-style scripts.

Descriptor for distance-based selection (`fps`) is deliberately simple and
self-contained -- composition fractions (over the elements observed in the
pool) optionally concatenated with per-atom energy (z-score standardized so
neither dimension dominates) -- not the paper's full cluster-graph
structural fingerprint (that needs a LAMMPS fingerprint build and is out of
scope for a general-purpose CLI utility; see
docs/concepts/qm_driver_plugins.md's committee_spread note for where
heavier descriptors could plug in later).

`stratified_holdout` bins frames by composition class (the set of elements
present, e.g. "C-only" vs "C+H mixed") and splits proportionally within
each class -- the same "mixed/pure" stratification pattern used in prior
HEA/binary-alloy studies with this toolchain, generalized off any
element-set composition, not just binary systems.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from ..io import xyzf as xyzf_io

NAME = "dataset-select"
SUMMARY = "FPS / random / stratified-holdout sampling over a frame pool."
SCHEMA = {
    "type": "object",
    "required": ["frames", "method"],
    "properties": {
        "frames": {"type": "string", "description": "Path to a .xyzf frame pool."},
        "method": {"type": "string", "enum": ["fps", "random", "stratified_holdout"]},
        "n_select": {"type": ["integer", "null"], "description": "Size of the selected/train set; alternative to holdout_fraction."},
        "holdout_fraction": {"type": ["number", "null"], "description": "Fraction of the pool held out; alternative to n_select."},
        "seed": {"type": "integer", "default": 42},
        "descriptor": {"type": "string", "enum": ["composition", "energy", "composition_energy"], "default": "composition", "description": "fps only."},
    },
}


def add_arguments(parser) -> None:
    parser.add_argument("--frames", default=None)
    parser.add_argument("--method", choices=["fps", "random", "stratified_holdout"], default=None)
    parser.add_argument("--n-select", dest="n_select", type=int, default=None)
    parser.add_argument("--holdout-fraction", dest="holdout_fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--descriptor", choices=["composition", "energy", "composition_energy"], default="composition")


def _resolve_n_select(n_pool: int, n_select, holdout_fraction) -> int:
    if n_select is not None:
        if not (0 < n_select <= n_pool):
            raise ValueError(f"n_select={n_select} out of range for a pool of {n_pool} frames")
        return n_select
    if holdout_fraction is not None:
        if not (0.0 < holdout_fraction < 1.0):
            raise ValueError(f"holdout_fraction={holdout_fraction} must be in (0, 1)")
        return max(1, round(n_pool * (1.0 - holdout_fraction)))
    raise ValueError("dataset-select requires one of n_select or holdout_fraction")


def _composition_vector(frame, elements) -> list:
    counts = {el: 0 for el in elements}
    for sym in frame.symbols:
        counts[sym] += 1
    n = len(frame.symbols)
    return [counts[el] / n for el in elements]


def _standardize(columns: list) -> list:
    """columns: list of per-dimension value lists (same length each) ->
    z-score standardized in place (returns new lists); a zero-variance
    dimension is left as all-zeros rather than dividing by zero."""
    out = []
    for col in columns:
        n = len(col)
        mean = sum(col) / n
        var = sum((x - mean) ** 2 for x in col) / n
        std = var**0.5
        out.append([0.0] * n if std == 0 else [(x - mean) / std for x in col])
    return out


def _descriptor_matrix(frames, mode: str):
    elements = sorted({sym for fr in frames for sym in fr.symbols})
    # one column per element (composition fractions), already in [0, 1]
    comp_cols = [list(col) for col in zip(*(_composition_vector(fr, elements) for fr in frames))]

    if mode == "composition":
        columns = comp_cols
    else:
        missing = [i for i, fr in enumerate(frames) if fr.energy is None]
        if missing:
            raise ValueError(
                f"descriptor={mode!r} requires every frame to have an energy, but "
                f"{len(missing)}/{len(frames)} frames don't (e.g. frame index {missing[0]})"
            )
        energy_col = [fr.energy / max(len(fr.symbols), 1) for fr in frames]
        energy_std = _standardize([energy_col])
        columns = energy_std if mode == "energy" else _standardize(comp_cols) + energy_std

    # transpose columns -> per-frame vectors
    return [list(row) for row in zip(*columns)] if columns else [[] for _ in frames]


def _farthest_point_sample(vectors, n_select: int, seed: int) -> list:
    n = len(vectors)
    if n_select >= n:
        return list(range(n))

    rng = random.Random(seed)
    start = rng.randrange(n)
    selected = [start]
    in_selected = {start}

    def sqdist(a, b):
        return sum((x - y) ** 2 for x, y in zip(a, b))

    min_dist = [sqdist(vectors[i], vectors[start]) for i in range(n)]

    for _ in range(n_select - 1):
        next_idx = max((i for i in range(n) if i not in in_selected), key=lambda i: min_dist[i])
        selected.append(next_idx)
        in_selected.add(next_idx)
        for i in range(n):
            if i in in_selected:
                continue
            d = sqdist(vectors[i], vectors[next_idx])
            if d < min_dist[i]:
                min_dist[i] = d

    return selected


def _composition_class(frame) -> tuple:
    return tuple(sorted({sym for sym in frame.symbols}))


def _stratified_split(frames, n_select: int, seed: int) -> tuple:
    rng = random.Random(seed)
    by_class: dict = {}
    for i, fr in enumerate(frames):
        by_class.setdefault(_composition_class(fr), []).append(i)

    n_pool = len(frames)
    selected: list = []
    for cls, idxs in by_class.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        k = round(len(idxs) * n_select / n_pool)
        selected.extend(idxs[:k])

    selected_set = set(selected)
    # rounding can drift the total slightly off n_select; correct by
    # moving frames between selected/holdout, preferring to keep every
    # class represented in both splits where possible
    if len(selected_set) < n_select:
        remaining = [i for i in range(n_pool) if i not in selected_set]
        rng.shuffle(remaining)
        selected_set.update(remaining[: n_select - len(selected_set)])
    elif len(selected_set) > n_select:
        extra = list(selected_set)
        rng.shuffle(extra)
        selected_set = set(extra[:n_select])

    return sorted(selected_set), [i for i in range(n_pool) if i not in selected_set]


def run(args) -> dict:
    if not getattr(args, "frames", None):
        raise ValueError("dataset-select requires --frames (or 'frames' in --json-in)")
    method = getattr(args, "method", None)
    if method not in ("fps", "random", "stratified_holdout"):
        raise ValueError("dataset-select requires --method {fps,random,stratified_holdout}")

    frames = xyzf_io.read_xyzf(args.frames)
    n_pool = len(frames)
    if n_pool == 0:
        raise ValueError(f"no frames found in {args.frames}")

    n_select = _resolve_n_select(n_pool, getattr(args, "n_select", None), getattr(args, "holdout_fraction", None))
    seed = getattr(args, "seed", 42) or 42

    if method == "fps":
        descriptor = getattr(args, "descriptor", "composition") or "composition"
        vectors = _descriptor_matrix(frames, descriptor)
        selected_indices = sorted(_farthest_point_sample(vectors, n_select, seed))
        holdout_indices = [i for i in range(n_pool) if i not in set(selected_indices)]
    elif method == "random":
        rng = random.Random(seed)
        selected_indices = sorted(rng.sample(range(n_pool), n_select))
        holdout_indices = [i for i in range(n_pool) if i not in set(selected_indices)]
    else:  # stratified_holdout
        selected_indices, holdout_indices = _stratified_split(frames, n_select, seed)

    out_dir = Path(getattr(args, "output_dir", None) or ".")
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_xyzf = out_dir / "selected.xyzf"
    holdout_xyzf = out_dir / "holdout.xyzf"
    xyzf_io.write_xyzf([frames[i] for i in selected_indices], selected_xyzf)
    xyzf_io.write_xyzf([frames[i] for i in holdout_indices], holdout_xyzf)

    indices_path = out_dir / "indices.json"
    indices_path.write_text(json.dumps({"selected_indices": selected_indices, "holdout_indices": holdout_indices}, indent=2))

    return {
        "method": method,
        "n_pool": n_pool,
        "n_selected": len(selected_indices),
        "n_holdout": len(holdout_indices),
        "selected_indices": selected_indices,
        "holdout_indices": holdout_indices,
        "selected_xyzf": str(selected_xyzf),
        "holdout_xyzf": str(holdout_xyzf),
        "indices_json": str(indices_path),
    }
