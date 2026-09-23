"""Per-element-pair minimum distance and radial distribution "shape"
computation over a pool of `.xyzf` frames, PBC-aware (minimum-image
convention), orthorhombic boxes only (mirrors io/lammps_data.py's
existing scope limit -- raises a clear error for non-ortho frames).

This is the data-driven half of ChIMES' own documented cutoff/lambda
guidance (see docs/concepts/cutoffs_and_lambdas.md): S_MINIM tracks the
minimum observed pair distance, MORSE_LAMBDA tracks the first RDF peak,
and S_MAXIM tracks RDF minima (shell boundaries) -- all directly
computable from the training data instead of guessed.

`RDF.g` is a *shape* signal for locating peaks/minima (raw pair-count
histogram divided by r^2 to cancel the r^2 shell-volume growth that would
otherwise dominate a raw count histogram and hide the real bonding/shell
structure), not a physically normalized g(r) -- absolute scale doesn't
matter for finding *where* the peaks/minima are, which is all any caller
here needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def _pair_key(a: str, b: str) -> tuple:
    return tuple(sorted((a, b)))


def _frame_pair_distances(frame, elements: set) -> dict:
    """{(el_a, el_b): np.ndarray of minimum-image distances} for one frame."""
    if frame.non_ortho:
        raise ValueError("io.rdf only supports orthorhombic frames (frame.non_ortho=False)")

    positions = np.asarray(frame.positions, dtype=float)
    box = np.asarray(frame.box, dtype=float)
    symbols = list(frame.symbols)
    n = len(symbols)

    diff = positions[:, None, :] - positions[None, :, :]
    diff -= box * np.round(diff / box)
    dist = np.sqrt((diff**2).sum(axis=-1))

    iu, ju = np.triu_indices(n, k=1)
    out: dict = {}
    for i, j in zip(iu, ju):
        key = _pair_key(symbols[i], symbols[j])
        out.setdefault(key, []).append(dist[i, j])
    return {k: np.asarray(v) for k, v in out.items()}


def pair_min_distance(frames: list, elements: list) -> dict:
    """{(el_a, el_b): minimum observed distance} across all frames, for
    every unordered pair among `elements` that actually co-occurs."""
    element_set = set(elements)
    mins: dict = {}
    for frame in frames:
        per_pair = _frame_pair_distances(frame, element_set)
        for key, arr in per_pair.items():
            m = float(arr.min())
            if key not in mins or m < mins[key]:
                mins[key] = m
    return mins


def min_box_dimension(frames: list) -> float:
    """Smallest box edge length across all frames (used for the S_MAXIM
    safety bound: chimes_lsq requires S_MAXIM <= this * NLAYERS / 2)."""
    dims = []
    for frame in frames:
        if frame.non_ortho:
            raise ValueError("io.rdf only supports orthorhombic frames (frame.non_ortho=False)")
        dims.extend(frame.box)
    return float(min(dims))


@dataclass
class RDF:
    r: list
    g: list

    def _smoothed(self, window: int = 5) -> list:
        g = np.asarray(self.g, dtype=float)
        if window <= 1 or len(g) < window:
            return g.tolist()
        kernel = np.ones(window) / window
        return np.convolve(g, kernel, mode="same").tolist()

    def first_peak(self, window: int = 5) -> Optional[float]:
        g = self._smoothed(window)
        for i in range(1, len(g) - 1):
            if g[i] > g[i - 1] and g[i] >= g[i + 1] and g[i] > 0:
                return self.r[i]
        return None

    def first_minimum_after_peak(self, window: int = 5) -> Optional[float]:
        g = self._smoothed(window)
        peak_idx = None
        for i in range(1, len(g) - 1):
            if g[i] > g[i - 1] and g[i] >= g[i + 1] and g[i] > 0:
                peak_idx = i
                break
        if peak_idx is None:
            return None
        for i in range(peak_idx + 1, len(g) - 1):
            if g[i] < g[i - 1] and g[i] <= g[i + 1]:
                return self.r[i]
        return None

    def second_minimum(self, window: int = 5) -> Optional[float]:
        g = self._smoothed(window)
        state = "seeking_peak1"
        for i in range(1, len(g) - 1):
            rising = g[i] > g[i - 1]
            falling = g[i] < g[i - 1]
            is_local_max = g[i] > g[i - 1] and g[i] >= g[i + 1] and g[i] > 0
            is_local_min = g[i] < g[i - 1] and g[i] <= g[i + 1]
            if state == "seeking_peak1" and is_local_max:
                state = "seeking_min1"
            elif state == "seeking_min1" and is_local_min:
                state = "seeking_peak2"
            elif state == "seeking_peak2" and is_local_max:
                state = "seeking_min2"
            elif state == "seeking_min2" and is_local_min:
                return self.r[i]
        return None


def pair_rdf(frames: list, elements: list, *, bin_width: float = 0.05, r_max: Optional[float] = None) -> dict:
    """{(el_a, el_b): RDF} over all frames. r_max defaults to
    min_box_dimension(frames)/2 (the same bound chimes_lsq itself enforces
    for S_MAXIM), so every bin reflects a value that's actually usable."""
    if r_max is None:
        r_max = min_box_dimension(frames) / 2.0

    n_bins = max(int(round(r_max / bin_width)), 1)
    edges = np.linspace(0.0, r_max, n_bins + 1)
    centers = ((edges[:-1] + edges[1:]) / 2.0).tolist()

    element_set = set(elements)
    counts: dict = {}
    for frame in frames:
        per_pair = _frame_pair_distances(frame, element_set)
        for key, arr in per_pair.items():
            arr = arr[arr < r_max]
            hist, _ = np.histogram(arr, bins=edges)
            if key not in counts:
                counts[key] = np.zeros(n_bins)
            counts[key] += hist

    result = {}
    r_arr = np.asarray(centers)
    safe_r2 = np.where(r_arr > 0, r_arr**2, 1.0)
    for key, hist in counts.items():
        shape = hist / safe_r2
        result[key] = RDF(r=centers, g=shape.tolist())
    return result
