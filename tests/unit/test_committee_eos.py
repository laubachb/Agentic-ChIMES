"""Committee bootstrap weights/spread and the Birch-Murnaghan fit."""

import numpy as np
import pytest

from agentic_chimes.stages import committee, eos_check


def test_bootstrap_weights_are_sqrt_counts_per_frame():
    groups = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    w = committee.bootstrap_weights(groups, np.random.default_rng(3))
    for g in range(3):
        assert len(set(w[groups == g])) == 1                      # one weight per frame
    assert np.isclose((w[[0, 3, 5]] ** 2).sum(), 3)               # counts sum to the number of frames


def test_spread_measures_disagreement():
    same = [[(1.0, np.ones((2, 3)))], [(1.0, np.ones((2, 3)))]]
    diff = [[(1.0, np.ones((2, 3)))], [(3.0, 3 * np.ones((2, 3)))]]
    assert committee.spread(same) == ([0.0], [0.0])
    f, e = committee.spread(diff)
    assert f[0] == pytest.approx(1.0) and e[0] == pytest.approx(0.5)


def test_birch_murnaghan_recovers_parameters():
    v0, e0, b0, b0p = 16.0, -100.0, 0.5, 4.5
    v = np.linspace(14.5, 17.5, 11)
    x = (v0 / v) ** (2 / 3)
    e = e0 + 9 * v0 * b0 / 16 * ((x - 1) ** 3 * b0p + (x - 1) ** 2 * (6 - 4 * x))
    p, rms = eos_check.birch_murnaghan(v, e)
    assert p == pytest.approx([v0, e0, b0, b0p], rel=1e-5) and rms < 1e-9
