"""ChIMES cluster-graph fingerprints (Laubach, Lordi, Lindsey, JCIM 66, 182,
2026), computed natively rather than through the FINGERPRINT build of
LAMMPS plus the MPI `histogram` tool in chimes_calculator
(`chimesFF/src/FP`). That route writes cluster files per frame and rank,
which lustre's file-count quota punishes.

For one configuration:

1. Every unique periodic 2-, 3- and 4-body cluster whose edges are all
   inside the model's outer cutoffs (per pair, per cluster type, as chimesFF
   reads them from params.txt) is a complete graph.
2. Each edge length is Morse-transformed with the pair's inner/outer cutoff
   and lambda, then sorted within the cluster.
3. The dissimilarity between two clusters of the same body order is the
   Euclidean distance between their sorted edge vectors.
4. Each body order gives a histogram of all pairwise dissimilarities,
   normalized to sum to 1: 100 bins on [0, d_max] with d_max = 3,
   sqrt(12) + 1 and sqrt(24) + 1, as in the shipped tool. Concatenated,
   they are the fingerprint.

This matches the shipped tool's output, including its (currently inactive)
composition term: `gen_flat_hists` builds its type vectors with a size and
then push_back()s onto them, so the composition weight it adds is always 0.
The fingerprint is therefore type-agnostic, as the paper describes. Verified
against `etc/lmp/tests/example-fingerprint/expected_output`.

Datasets are compared with the Mahalanobis distance of the paper:

- D^2 between two sets of per-frame fingerprints uses the pooled
  covariance;
- D_j^2 of one frame against a reference set uses that set's covariance;
- chi-squared critical values use the covariance rank as the degrees of
  freedom (pseudo-inverse: 300 bins, and usually fewer frames than that).
"""

from __future__ import annotations

import math
from itertools import combinations
from pathlib import Path

import numpy as np

NBINS = 100
DMAX = {2: 3.0, 3: math.sqrt(12.0) + 1.0, 4: math.sqrt(24.0) + 1.0}


# ------------------------------------------------------------------ model geometry

class ModelCutoffs:
    """Per-pair lambdas and 2/3/4-body inner/outer cutoffs from a params.txt."""

    def __init__(self, params_path):
        lines = Path(params_path).read_text().splitlines()
        self.elements, self.pair = [], {}
        i = next(k for k, ln in enumerate(lines) if "TYPEIDX" in ln)
        for ln in lines[i + 1:]:
            t = ln.split()
            if len(t) < 2 or not t[0].isdigit():
                break
            self.elements.append(t[1])
        i = next(k for k, ln in enumerate(lines) if "PAIRIDX" in ln)
        for ln in lines[i + 1:]:
            t = ln.split()
            if len(t) < 5 or not t[0].isdigit():
                break
            lam = float(t[6]) if len(t) > 6 else float(t[-1])
            for a, b in ((t[1], t[2]), (t[2], t[1])):
                self.pair[(a, b)] = (float(t[3]), float(t[4]), lam)
        # special many-body cutoffs: {(order, bound): {"ALL": v} or {cluster_key: [(pair_name, v), ...]}}
        self.special = {}
        k = 0
        while k < len(lines):
            ln = lines[k]
            if ln.startswith("SPECIAL"):
                t = ln.replace(":", " ").split()
                order, bound, mode = int(t[1][0]), t[2].upper(), t[3].upper()
                if mode == "ALL":
                    self.special[(order, bound)] = {"ALL": float(t[4])}
                else:
                    table = {}
                    npairs = order * (order - 1) // 2
                    for row in lines[k + 1: k + 1 + int(t[4])]:
                        r = row.split()
                        names = r[1:1 + npairs]
                        vals = [float(x) for x in r[1 + npairs:1 + 2 * npairs]]
                        table[tuple(sorted(names))] = list(zip(names, vals))
                    self.special[(order, bound)] = table
                    k += int(t[4])
            k += 1

    def pair_name(self, a, b):
        ia, ib = self.elements.index(a), self.elements.index(b)
        return a + b if ia <= ib else b + a

    def max_outer(self, order):
        vals = [v[1] for v in self.pair.values()]
        spec = self.special.get((order, "S_MAXIM"))
        if order > 2 and spec:
            vals = [spec["ALL"]] if "ALL" in spec else [v for rows in spec.values() for _, v in rows]
        return max(vals)

    def edge_params(self, order, symbols_pairs):
        """[(rin, rout, lambda)] for a cluster's edges, given its [(a, b)] pairs."""
        names = [self.pair_name(a, b) for a, b in symbols_pairs]
        out = []
        for bound_idx, bound in ((0, "S_MINIM"), (1, "S_MAXIM")):
            vals = [self.pair[(a, b)][bound_idx] for a, b in symbols_pairs]
            spec = self.special.get((order, bound)) if order > 2 else None
            if spec:
                if "ALL" in spec:
                    vals = [spec["ALL"]] * len(vals)
                else:
                    rows = spec.get(tuple(sorted(names)))
                    if rows:
                        pool = list(rows)
                        for e, n in enumerate(names):
                            m = next(q for q, (pn, _) in enumerate(pool) if pn == n)
                            vals[e] = pool.pop(m)[1]
            out.append(vals)
        lams = [self.pair[(a, b)][2] for a, b in symbols_pairs]
        return list(zip(out[0], out[1], lams))


def morse(rin, rout, lam, r):
    """chimes_calculator FP `transform`: exp(-r/lambda) mapped to [-1, 1] over [rin, rout]."""
    x_min, x_max = math.exp(-rin / lam), math.exp(-rout / lam)
    return (math.exp(-r / lam) - 0.5 * (x_max + x_min)) / (-0.5 * (x_max - x_min))


# ------------------------------------------------------------------ clusters

def _atoms(frame):
    from ase import Atoms

    cell = np.asarray(frame.box if frame.non_ortho else np.diag(frame.box), dtype=float)
    return Atoms(symbols=frame.symbols, positions=frame.positions, cell=cell, pbc=True)


def clusters(frame, model: ModelCutoffs, orders=(2, 3, 4)) -> dict:
    """{order: array (n_clusters, n_edges) of sorted transformed edge lengths}
    for every unique periodic cluster with all edges inside the outer cutoffs.

    Vectorized: each atom anchors the clusters it belongs to (neighbor lists
    with image shifts), edges are filtered with per-type cutoffs, and the
    `order` copies of each periodic cluster (one per anchor) collapse to one
    through canonical (atom, image shift) keys. Exact for any cell size."""
    from ase.neighborlist import neighbor_list

    at = _atoms(frame)
    types = np.array([model.elements.index(s) for s in at.get_chemical_symbols()])
    rmax = max(model.max_outer(o) for o in orders)
    ii, jj, SS, DD = neighbor_list("ijSD", at, rmax)
    n = len(at)
    starts = np.searchsorted(ii, np.arange(n + 1))
    out = {}
    for order in orders:
        rcut = model.max_outer(order)
        npair = order * (order - 1) // 2
        pairs = list(combinations(range(order), 2))
        table = {}
        edge_rows, key_rows = [], []
        for i in range(n):
            sl = slice(starts[i], starts[i + 1])
            d_i = np.linalg.norm(DD[sl], axis=1)
            keep = d_i < rcut
            js, ss, vs = jj[sl][keep], SS[sl][keep], DD[sl][keep]
            m = len(js)
            if m < order - 1:
                continue
            if order == 2:
                combos = np.arange(m)[:, None]
            else:
                dm = np.linalg.norm(vs[:, None, :] - vs[None, :, :], axis=-1)
                adj = dm < rcut
                if order == 3:
                    combos = np.argwhere(np.triu(adj, 1))
                else:
                    trip = []
                    for a in range(m):
                        for b in np.nonzero(adj[a, a + 1:])[0] + a + 1:
                            cs = np.nonzero(adj[a, b + 1:] & adj[b, b + 1:])[0] + b + 1
                            if len(cs):
                                trip.append(np.column_stack([np.full(len(cs), a), np.full(len(cs), b), cs]))
                    combos = np.vstack(trip) if trip else np.empty((0, 3), dtype=int)
            if not len(combos):
                continue
            # member positions (anchor at the origin), atom ids, image shifts, types
            P = np.concatenate([np.zeros((len(combos), 1, 3)), vs[combos]], axis=1)
            A = np.concatenate([np.full((len(combos), 1), i), js[combos]], axis=1)
            S = np.concatenate([np.zeros((len(combos), 1, 3), dtype=int), ss[combos]], axis=1)
            T = types[A]
            D = np.stack([np.linalg.norm(P[:, a] - P[:, b], axis=1) for a, b in pairs], axis=1)
            E = np.empty_like(D)
            ok = np.zeros(len(combos), dtype=bool)
            for tt in {tuple(r) for r in T.tolist()}:
                sel = np.all(T == np.array(tt), axis=1)
                if tt not in table:
                    syms = [model.elements[t] for t in tt]
                    table[tt] = np.array(model.edge_params(order, [(syms[a], syms[b]) for a, b in pairs]))
                prm = table[tt]                                   # (npair, 3): rin, rout, lambda
                d = D[sel]
                inside = np.all(d < prm[:, 1], axis=1)
                x_min, x_max = np.exp(-prm[:, 0] / prm[:, 2]), np.exp(-prm[:, 1] / prm[:, 2])
                E[sel] = (np.exp(-d / prm[:, 2]) - 0.5 * (x_max + x_min)) / (-0.5 * (x_max - x_min))
                ok[sel] = inside
            if not ok.any():
                continue
            A, S, E = A[ok], S[ok], E[ok]
            # distinct members only (a small cell can bring the same image twice)
            key = A * 1_000_000 + (S[..., 0] + 50) * 10_000 + (S[..., 1] + 50) * 100 + (S[..., 2] + 50)
            distinct = np.array([len(set(r)) == order for r in key.tolist()]) if order > 2 else A[:, 0] * 0 == 0
            if order == 2:
                distinct = ~((A[:, 0] == A[:, 1]) & np.all(S[:, 1] == 0, axis=1))
            A, S, E = A[distinct], S[distinct], E[distinct]
            # canonical form: shift so the lowest-(atom, shift) member sits in the home cell, then sort members
            k0 = A * 1_000_000 + (S[..., 0] + 50) * 10_000 + (S[..., 1] + 50) * 100 + (S[..., 2] + 50)
            anchor = np.argmin(k0, axis=1)
            S = S - S[np.arange(len(S)), anchor][:, None, :]
            kc = np.sort(A * 1_000_000 + (S[..., 0] + 50) * 10_000 + (S[..., 1] + 50) * 100 + (S[..., 2] + 50), axis=1)
            key_rows.append(kc)
            edge_rows.append(np.sort(E, axis=1))
        if key_rows:
            keys = np.vstack(key_rows)
            edges = np.vstack(edge_rows)
            _, first = np.unique(keys, axis=0, return_index=True)
            out[order] = edges[np.sort(first)]
        else:
            out[order] = np.empty((0, npair))
    return out


def histogram(edges: np.ndarray, order: int, *, max_clusters: int | None = 20000, seed: int = 0) -> tuple:
    """(normalized histogram over NBINS bins, n_clusters used) of all pairwise
    dissimilarities within one body order. Above max_clusters, a uniform
    random subset is used (the histogram is an estimate)."""
    n = len(edges)
    if max_clusters and n > max_clusters:
        edges = edges[np.random.default_rng(seed).choice(n, max_clusters, replace=False)]
        n = max_clusters
    dmax = DMAX[order]
    binw = dmax / NBINS
    hist = np.zeros(NBINS, dtype=np.int64)
    for start in range(0, n - 1, 128):
        a = edges[start:start + 128]
        d = np.sqrt(((a[:, None, :] - edges[None, :, :]) ** 2).sum(-1))
        rows, cols = np.nonzero(np.arange(start, start + len(a))[:, None] < np.arange(n)[None, :])
        d = d[rows, cols]
        b = np.floor(d / binw).astype(int)
        b[d == dmax] -= 1
        hist += np.bincount(np.clip(b, 0, NBINS - 1), minlength=NBINS)
    total = hist.sum()
    return (hist / total if total else hist.astype(float)), n


def fingerprint(frame, model: ModelCutoffs, orders=(2, 3, 4), max_clusters=20000) -> np.ndarray:
    cl = clusters(frame, model, orders)
    return np.concatenate([histogram(cl[o], o, max_clusters=max_clusters)[0] for o in orders])


# ------------------------------------------------------------------ statistics

def _chi2_critical(k: int, alpha: float) -> float:
    try:
        from scipy.stats import chi2

        return float(chi2.ppf(1.0 - alpha, k))
    except ImportError:  # Wilson-Hilferty approximation
        z = {0.1: 1.2816, 0.05: 1.6449, 0.01: 2.3263}.get(alpha, 1.6449)
        return float(k * (1 - 2 / (9 * k) + z * math.sqrt(2 / (9 * k))) ** 3)


def mahalanobis_sets(fx: np.ndarray, fy: np.ndarray, alpha: float = 0.1) -> dict:
    """D^2 between the mean fingerprints of two sets (pooled covariance),
    compared with chi-squared as in the paper. This measures how far apart
    the two distributions sit in units of their spread (an effect size), not
    whether their means differ significantly: two sets whose frames overlap
    heavily are "indistinguishable" even with a detectable mean shift."""
    nx, ny = len(fx), len(fy)
    if nx < 2 or ny < 2:
        raise ValueError("each set needs at least 2 frames for a covariance")
    s = ((nx - 1) * np.cov(fx, rowvar=False) + (ny - 1) * np.cov(fy, rowvar=False)) / (nx + ny - 2)
    diff = fx.mean(0) - fy.mean(0)
    s_inv = np.linalg.pinv(s, rcond=1e-10)
    k = int(np.linalg.matrix_rank(s, tol=1e-10 * np.abs(s).max())) if s.any() else 0
    d2 = float(diff @ s_inv @ diff)
    crit = _chi2_critical(max(k, 1), alpha)
    return {"D2": d2, "dof": k, "critical": crit, "alpha": alpha, "distinguishable": d2 > crit}


def novelty(reference: np.ndarray, frames: np.ndarray, alpha: float = 0.1) -> dict:
    """D_j^2 of each frame against the reference set's mean and covariance."""
    if len(reference) < 2:
        raise ValueError("the reference set needs at least 2 frames")
    s = np.cov(reference, rowvar=False)
    s_inv = np.linalg.pinv(s, rcond=1e-10)
    k = int(np.linalg.matrix_rank(s, tol=1e-10 * np.abs(s).max())) if s.any() else 0
    diff = frames - reference.mean(0)
    dj2 = np.einsum("ij,jk,ik->i", diff, s_inv, diff)
    crit = _chi2_critical(max(k, 1), alpha)
    return {"Dj2": dj2.tolist(), "dof": k, "critical": crit, "alpha": alpha,
            "novel": (dj2 > crit).tolist(), "fraction_novel": float(np.mean(dj2 > crit))}
