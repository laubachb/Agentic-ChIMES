"""Normalized solvers must recover coefficients of tiny-scale columns that
raw-scale regularization discards (ChIMES 4-body columns are ~1e-7 of the
2-body scale)."""

import numpy as np
import pytest

from agentic_chimes.stages._solvers import solve_normalized


@pytest.fixture
def disparate():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(400, 6))
    A[:, 4:] *= 1e-7                       # "4-body" columns
    x_true = np.array([1.0, -2.0, 0.5, 3.0, 4e6, -2e6])  # tiny columns need huge coefficients
    return A, A @ x_true, x_true


@pytest.mark.parametrize("method,kw", [("nsvd", {"alpha": 0, "eps": 1e-10}), ("nridge", {"alpha": 1e-10, "eps": 0}),
                                       ("nlasso", {"alpha": 1e-7, "eps": 0})])
def test_recovers_small_scale_columns(disparate, method, kw):
    A, b, x_true = disparate
    x = solve_normalized(A, b, method, **kw)
    assert np.allclose(A @ x, b, atol=1e-5)
    assert np.allclose(x[4:], x_true[4:], rtol=1e-2)


def test_raw_ridge_would_lose_them(disparate):
    A, b, x_true = disparate
    raw = np.linalg.solve(A.T @ A + 1e-6 * np.eye(6), A.T @ b)
    assert abs(raw[4]) < 1e-3 * abs(x_true[4])  # why normalization is needed


def test_row_weights_applied(disparate):
    A, b, _ = disparate
    w = np.ones(len(b)); w[:200] = 0.0
    b2 = b.copy(); b2[:200] += 100.0          # corrupt only zero-weight rows
    x = solve_normalized(A, b2, "nsvd", alpha=0, eps=1e-10, weights=w)
    assert np.allclose(A[200:] @ x, b[200:], atol=1e-5)


def test_frame_groups_from_amat_layout(tmp_path):
    # two frames: 2 atoms (6 force rows) + 3 energy rows, then 1 atom (3 rows) + 3 energy rows
    labels = ["Cu"] * 6 + ["+1"] * 3 + ["Zr"] * 3 + ["+1"] * 3
    (tmp_path / "b-labeled.txt").write_text("\n".join(f"{l} 0.0" for l in labels))
    (tmp_path / "natoms.txt").write_text("\n".join(["2"] * 9 + ["1"] * 6))
    from agentic_chimes.stages._solvers import frame_groups
    assert frame_groups(tmp_path / "b-labeled.txt", tmp_path / "natoms.txt").tolist() == [0] * 9 + [1] * 6


def test_ridge_cv_prefers_regularization_on_noisy_overparameterized_data():
    from agentic_chimes.stages._solvers import ridge_cv
    rng = np.random.default_rng(1)
    A = rng.normal(size=(120, 80))
    b = A[:, :3] @ np.array([1.0, -1.0, 0.5]) + rng.normal(scale=1.0, size=120)
    groups = np.repeat(np.arange(40), 3)
    x, alpha, curve = ridge_cv(A, b, groups)
    assert alpha > 1e-6                          # not the unregularized end
    assert curve[alpha] <= min(curve.values()) + 1e-12


def test_block_lasso_equals_lassolars_without_many_body_and_rescues_small_block():
    from sklearn.linear_model import LassoLars

    from agentic_chimes.stages._solvers import block_lasso
    rng = np.random.default_rng(2)
    A2 = rng.normal(size=(300, 5)) * 40.0
    A4 = rng.normal(size=(300, 3)) * 4e-6
    x2, x4 = np.array([1.0, -0.5, 0.3, 0.0, 2.0]), np.array([3e6, -2e6, 1e6])
    b = A2 @ x2 + A4 @ x4
    ref = LassoLars(alpha=1e-3, fit_intercept=False, fit_path=False).fit(A2, A2 @ x2).coef_
    assert np.allclose(block_lasso(A2, A2 @ x2, (5, 0, 0), alpha=1e-3), np.ravel(ref))
    A = np.hstack([A2, A4])
    raw = np.ravel(LassoLars(alpha=1e-3, fit_intercept=False, fit_path=False).fit(A, b).coef_)
    blk = block_lasso(A, b, (5, 0, 3), alpha=1e-3)
    assert np.allclose(raw[5:], 0)                        # raw lasso switches the small block off
    assert np.linalg.norm(b - A @ blk) < 0.05 * np.linalg.norm(b - A @ raw)
