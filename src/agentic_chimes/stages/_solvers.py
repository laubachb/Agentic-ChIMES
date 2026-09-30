"""Column-normalized least-squares solvers for ChIMES design matrices.

chimes_lsq.py's local solvers act on raw columns. ChIMES columns span many
orders of magnitude: each cluster distance contributes a (1 - r/r_c)^3
smoothing factor, so on real Cu-Zr data 3-body columns were ~1e-3 and
4-body ~1e-7 of the 2-body scale. An L1 penalty or SVD truncation sized for
the 2-body columns then silently removes the many-body terms (4-body fits
matched 3-body fits to 1e-8). Scaling every column to unit norm before
solving, then unscaling the solution, puts every term on equal footing.

The solution is handed back to chimes_lsq.py through its own
`--read_output` path (x.txt, Ax.txt), so params.txt is written by the
reference code, not reimplemented here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

NORMALIZED = {"nsvd", "nridge", "nlasso", "nridgecv", "blocklasso"}
DEFAULT_ALPHAS = tuple(10.0 ** k for k in range(-10, 1))


def solve_normalized(A, b, method: str, *, alpha: float, eps: float, weights=None):
    """Return x for A x ~= b with column-normalized regularization.

    nsvd:   truncated SVD, singular values below eps * max dropped
    nridge: ridge, penalty alpha * ||w||^2 on normalized coefficients
    nlasso: coordinate-descent lasso (sklearn objective
            1/(2n)||b - Aw||^2 + alpha ||w||_1) on normalized coefficients
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        A, b = A * w[:, None], b * w
    scale = np.linalg.norm(A, axis=0)
    scale[scale == 0] = 1.0
    As = A / scale

    if method == "nsvd":
        U, s, Vt = np.linalg.svd(As, full_matrices=False)
        keep = s > eps * s.max()
        coef = Vt[keep].T @ ((U[:, keep].T @ b) / s[keep])
    elif method == "nridge":
        n = As.shape[1]
        coef = np.linalg.solve(As.T @ As + alpha * np.eye(n), As.T @ b)
    elif method == "nlasso":
        from sklearn.linear_model import Lasso

        reg = Lasso(alpha=alpha, fit_intercept=False, max_iter=200000, tol=1e-8)
        reg.fit(As, b)
        coef = reg.coef_
    else:
        raise ValueError(f"unknown normalized method {method!r}; known: {sorted(NORMALIZED)}")
    return coef / scale


def frame_groups(b_labeled: Path, natoms: Path):
    """Row -> training-frame index, from amat-build's own b-labeled.txt and
    natoms.txt (natoms of the row's frame, one line per row). A frame is
    3*N force rows followed by its stress ("s_..") and energy ("+1") rows, if any."""
    labels = [ln.split()[0] for ln in Path(b_labeled).read_text().splitlines()]
    nat = [int(float(x)) for x in Path(natoms).read_text().split()]
    groups, i, g = [], 0, 0
    while i < len(labels):
        n_force = 3 * nat[i]
        j = i + n_force
        while j < len(labels) and (labels[j] == "+1" or "s_" in labels[j]):
            j += 1
        groups.extend([g] * (j - i))
        i, g = j, g + 1
    return np.asarray(groups[: len(labels)])


def ridge_cv(A, b, groups, *, alphas=DEFAULT_ALPHAS, folds: int = 5, weights=None, seed: int = 0, score_rows=None):
    """Column-normalized ridge with alpha chosen by K-fold cross-validation
    over whole training frames (rows of one frame are correlated; splitting
    them would reward overfitting). One SVD per fold covers every alpha.
    `score_rows` (bool mask) limits the CV error to those rows -- pass the
    force rows: energy rows are hundreds of kcal/mol, repeated per frame,
    and otherwise choose alpha for energies alone.
    Returns (x, chosen_alpha, {alpha: cv_rmse})."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        A, b = A * w[:, None], b * w
    scale = np.linalg.norm(A, axis=0)
    scale[scale == 0] = 1.0
    As = A / scale
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(rng.permutation(uniq), np.arange(len(uniq)) % max(2, min(folds, len(uniq)))))
    fold = np.array([fold_of[g] for g in groups])
    sq = np.zeros(len(alphas))
    for k in np.unique(fold):
        tr, te = fold != k, fold == k
        U, s, Vt = np.linalg.svd(As[tr], full_matrices=False)
        Utb = U.T @ b[tr]
        for ai, a in enumerate(alphas):
            coef = Vt.T @ (s / (s**2 + a) * Utb)
            m = te if score_rows is None else te & score_rows
            sq[ai] += np.sum((b[m] - As[m] @ coef) ** 2)
    n_scored = len(b) if score_rows is None else int(np.sum(score_rows))
    cv = np.sqrt(sq / n_scored)
    best = int(np.argmin(cv))
    U, s, Vt = np.linalg.svd(As, full_matrices=False)
    coef = Vt.T @ (s / (s**2 + alphas[best]) * (U.T @ b))
    return coef / scale, float(alphas[best]), {float(a): float(c) for a, c in zip(alphas, cv)}


def block_counts(fm_setup_log: Path) -> tuple:
    """(n_2b, n_3b, n_4b) coefficient counts from chimes_lsq's log; energy
    offset columns (one per element when energies are fit) come after them."""
    import re

    log = Path(fm_setup_log).read_text(errors="replace")

    def count(pattern):
        m = re.search(pattern, log)
        return int(m.group(1)) if m else 0

    return (count(r"two-body non-coulomb parameters is:\s*(\d+)"),
            count(r"three-body Chebyshev parameters is:\s*(\d+)"),
            count(r"four-body\s+Chebyshev parameters is:\s*(\d+)"))


def block_lasso(A, b, blocks, *, alpha: float, weights=None):
    """LassoLars (exactly chimes_lsq.py's `lassolars`) after rescaling each
    body-order block so its median column norm matches the 2-body block.

    Raw lassolars penalizes a 4-body coefficient as if its column had 2-body
    scale, although it is ~1e-7 as large, so many-body terms are switched off
    regardless of their value. Per-column normalization (nlasso) overcorrects
    and lets tiny, noisy columns in. Block scaling equalizes body orders and
    keeps the within-block relative scales, so suppression inside a block
    is unchanged. Identical to lassolars for a 2-body-only basis.
    `blocks` = (n_2b, n_3b, n_4b); remaining columns (energy offsets) keep scale 1.
    """
    from sklearn.linear_model import LassoLars

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        A, b = A * w[:, None], b * w
    n2, n3, n4 = blocks
    norms = np.linalg.norm(A, axis=0)
    ref = np.median(norms[:n2]) if n2 else 1.0
    scale = np.ones(A.shape[1])
    for lo, hi in ((n2, n2 + n3), (n2 + n3, n2 + n3 + n4)):
        if hi > lo:
            med = np.median(norms[lo:hi])
            if med > 0:
                scale[lo:hi] = ref / med
    reg = LassoLars(alpha=alpha, fit_intercept=False, fit_path=False, max_iter=100000)
    reg.fit(A * scale, b)
    return np.ravel(reg.coef_) * scale
