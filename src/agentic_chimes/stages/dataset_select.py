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

By default (`split_by: group`) it holds out whole *groups* of correlated
frames rather than single frames. Consecutive steps of one relaxation or
closely spaced MD frames are near-copies of each other; splitting them
across train and holdout leaks the answer into the test and overstates
accuracy. Two frames are linked when they have the same atoms in the same
order, cells within `group_cell_tol` and a minimum-image RMS displacement
below `group_rmsd` Å; groups are the connected chains of links (checked
within a window of nearby frames, since trajectories are stored in order).
A composition class that is a single group cannot be split by group; its
last frames (a contiguous block) are held out instead, and a note says so.

Either way, the frame holding each element pair's closest contact always
stays in the selected (training) set. The inner cutoff is set just below the
closest *training* distance, so a holdout frame closer than that sits inside
the repulsive penalty. On Cu-Zr that single frame pushed the holdout relative
force error from ~0.3 to ~1.4.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

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
        "split_by": {"type": "string", "enum": ["group", "frame"], "default": "group", "description": "stratified_holdout only. group = hold out whole groups of correlated frames (relaxation paths, closely spaced MD frames) so the holdout is not a near-copy of training data; frame = independent frames (the old behavior)."},
        "group_rmsd": {"type": "number", "default": 0.3, "description": "Å. Frames with identical atom order, matching cells and a minimum-image RMS displacement below this are linked into one group."},
        "group_cell_tol": {"type": "number", "default": 0.03, "description": "Relative tolerance on cell vectors for linking frames."},
    },
}


def add_arguments(parser) -> None:
    parser.add_argument("--frames", default=None)
    parser.add_argument("--method", choices=["fps", "random", "stratified_holdout"], default=None)
    parser.add_argument("--n-select", dest="n_select", type=int, default=None)
    parser.add_argument("--holdout-fraction", dest="holdout_fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--descriptor", choices=["composition", "energy", "composition_energy"], default="composition")
    parser.add_argument("--split-by", dest="split_by", choices=["group", "frame"], default="group")
    parser.add_argument("--group-rmsd", dest="group_rmsd", type=float, default=0.3)
    parser.add_argument("--group-cell-tol", dest="group_cell_tol", type=float, default=0.03)


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

    # closest contacts stay in training (see the module docstring): swap each one in for another frame
    for k in sorted(closest_contact_frames(frames)):
        if k not in selected_set:
            swap = [i for i in sorted(selected_set) if _composition_class(frames[i]) == _composition_class(frames[k])]
            swap = swap or sorted(selected_set)
            selected_set.discard(rng.choice(swap))
            selected_set.add(k)

    return sorted(selected_set), [i for i in range(n_pool) if i not in selected_set]


def _cell(frame) -> np.ndarray:
    box = np.asarray(frame.box, dtype=float)
    return box if box.ndim == 2 else np.diag(box)


def correlated_groups(frames, *, rmsd: float = 0.3, cell_tol: float = 0.03, window: int = 100) -> list:
    """Group id per frame: connected chains of near-identical frames."""
    parent = list(range(len(frames)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    buckets: dict = {}
    for i, fr in enumerate(frames):
        buckets.setdefault(tuple(fr.symbols), []).append(i)
    for idxs in buckets.values():
        pos = {i: np.asarray(frames[i].positions, dtype=float) for i in idxs}
        cells = {i: _cell(frames[i]) for i in idxs}
        for a, i in enumerate(idxs):
            ci = cells[i]
            inv = np.linalg.inv(ci)
            scale = np.linalg.norm(ci, axis=1)
            for j in idxs[a + 1: a + 1 + window]:
                if find(i) == find(j):
                    continue
                if np.any(np.linalg.norm(cells[j] - ci, axis=1) > cell_tol * scale):
                    continue
                frac = (pos[j] - pos[i]) @ inv
                frac -= np.round(frac)
                d = frac @ ci
                if np.sqrt(np.mean(np.sum(d * d, axis=1))) < rmsd:
                    parent[find(j)] = find(i)
    roots: dict = {}
    return [roots.setdefault(find(i), len(roots)) for i in range(len(frames))]


def closest_contact_frames(frames, rmax: float = 4.0) -> set:
    """Indices of the frames holding each element pair's smallest distance."""
    from ase import Atoms
    from ase.neighborlist import neighbor_list

    best = {}
    for k, f in enumerate(frames):
        cell = np.asarray(f.box if f.non_ortho else np.diag(f.box), dtype=float)
        at = Atoms(symbols=f.symbols, positions=f.positions, cell=cell, pbc=True)
        i, j, d = neighbor_list("ijd", at, rmax)
        sym = np.array(f.symbols)
        for a, b, r in zip(sym[i], sym[j], d):
            key = tuple(sorted((a, b)))
            if r < best.get(key, (np.inf, None))[0]:
                best[key] = (float(r), k)
    return {k for _, k in best.values()}


def _group_split(frames, n_select: int, seed: int, *, rmsd: float, cell_tol: float) -> tuple:
    """Stratified by composition class; whole correlated groups go to the holdout."""
    rng = random.Random(seed)
    groups = correlated_groups(frames, rmsd=rmsd, cell_tol=cell_tol)
    protected_groups = {groups[k] for k in closest_contact_frames(frames)}
    n_pool = len(frames)
    n_hold_target = n_pool - n_select
    by_class: dict = {}
    for i, fr in enumerate(frames):
        by_class.setdefault(_composition_class(fr), {}).setdefault(groups[i], []).append(i)

    holdout, notes = set(), []
    for cls, members in by_class.items():
        n_cls = sum(len(v) for v in members.values())
        target = round(n_cls * n_hold_target / n_pool)
        if target == 0:
            continue
        gids = [g for g in members if g not in protected_groups]
        rng.shuffle(gids)
        taken = 0
        if len(gids) > 1:
            for g in gids:
                size = len(members[g])
                if taken + size <= target + max(1, target // 4):
                    holdout.update(members[g])
                    taken += size
                if taken >= target:
                    break
        if taken < target // 2 and members:
            # one (or a few very large) groups: hold out a contiguous tail block of the largest one
            big = max(members, key=lambda g: len(members[g]))
            protected = closest_contact_frames([frames[i] for i in members[big]])
            block = [i for n_, i in enumerate(members[big]) if i not in holdout and n_ not in protected][-(target - taken):]
            holdout.update(block)
            taken += len(block)
            notes.append(f"class {'-'.join(cls)}: frames are one correlated group (e.g. a single trajectory); held out its last "
                         f"{len(block)} frames as a contiguous block, which is less correlated with training than random frames")
    selected = [i for i in range(n_pool) if i not in holdout]
    info = {"n_groups": len(set(groups)), "largest_group": max(groups.count(g) for g in set(groups)), "notes": notes,
            "kept_for_closest_contacts": len(protected_groups)}
    return selected, sorted(holdout), info


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
        split_by = getattr(args, "split_by", "group") or "group"
        if split_by == "group":
            selected_indices, holdout_indices, group_info = _group_split(
                frames, n_select, seed, rmsd=getattr(args, "group_rmsd", 0.3) or 0.3,
                cell_tol=getattr(args, "group_cell_tol", 0.03) or 0.03)
        else:
            selected_indices, holdout_indices = _stratified_split(frames, n_select, seed)

    out_dir = Path(getattr(args, "output_dir", None) or ".")
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_xyzf = out_dir / "selected.xyzf"
    holdout_xyzf = out_dir / "holdout.xyzf"
    xyzf_io.write_xyzf([frames[i] for i in selected_indices], selected_xyzf)
    xyzf_io.write_xyzf([frames[i] for i in holdout_indices], holdout_xyzf)

    indices_path = out_dir / "indices.json"
    indices_path.write_text(json.dumps({"selected_indices": selected_indices, "holdout_indices": holdout_indices}, indent=2))

    extra = {}
    if method == "stratified_holdout":
        extra = {"split_by": split_by, **({"groups": group_info} if split_by == "group" else {})}
    return {
        **extra,
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
