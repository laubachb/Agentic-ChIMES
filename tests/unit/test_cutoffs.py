"""stages/_cutoffs.py tested against a simple cubic lattice with known
shell structure (real physics, not just synthetic numbers), and a
deliberately small periodic box to exercise the box-safety cap."""

import pytest

from agentic_chimes.io.xyzf import Frame
from agentic_chimes.stages import _cutoffs

A = 3.0
N = 4


def _sc_lattice_frame(box_scale=1.0):
    positions = [[i * A, j * A, k * A] for i in range(N) for j in range(N) for k in range(N)]
    box = [N * A * box_scale, N * A * box_scale, N * A * box_scale]
    return Frame(symbols=["Ar"] * len(positions), positions=positions, forces=[[0, 0, 0]] * len(positions), box=box)


def test_s_minim_is_min_distance_minus_delta():
    result = _cutoffs.derive_pair_params([_sc_lattice_frame()], ["Ar"], s_minim_delta=0.02)
    assert result["pairs"]["Ar-Ar"]["s_minim"] == pytest.approx(A - 0.02, abs=1e-6)


def test_s_minim_delta_is_configurable():
    result = _cutoffs.derive_pair_params([_sc_lattice_frame()], ["Ar"], s_minim_delta=0.002)
    assert result["pairs"]["Ar-Ar"]["s_minim"] == pytest.approx(A - 0.002, abs=1e-6)


def test_morse_lambda_near_first_shell():
    result = _cutoffs.derive_pair_params([_sc_lattice_frame()], ["Ar"])
    assert result["pairs"]["Ar-Ar"]["morse_lambda"] == pytest.approx(A, abs=0.3)


def test_s_maxim_ordering_and_within_safety_bound():
    result = _cutoffs.derive_pair_params([_sc_lattice_frame()], ["Ar"])
    p = result["pairs"]["Ar-Ar"]
    bound = result["box_safety_bound"]
    assert p["s_maxim_3b"] <= bound
    assert p["s_maxim_2b"] <= bound
    # 3-body (1st shell boundary) should not exceed 2-body (2nd shell boundary)
    assert p["s_maxim_3b"] <= p["s_maxim_2b"] + 1e-6
    assert not p["s_maxim_3b_capped"]


def test_safety_cap_engages_on_small_box():
    # box edge = 3.0 -> safety bound = 1.5, far below the documented 8 A default
    small = Frame(
        symbols=["Ar"] * 2, positions=[[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]],
        forces=[[0, 0, 0]] * 2, box=[3.0, 3.0, 3.0],
    )
    result = _cutoffs.derive_pair_params([small], ["Ar"])
    p = result["pairs"]["Ar-Ar"]
    assert result["box_safety_bound"] == pytest.approx(1.5)
    assert p["s_maxim_2b"] == pytest.approx(1.5)
    assert p["s_maxim_2b_capped"] is True
    assert "box-safety bound" in p["s_maxim_2b_cap_reason"]


def test_nlayers_scales_safety_bound():
    result_1 = _cutoffs.derive_pair_params([_sc_lattice_frame()], ["Ar"], nlayers=1)
    result_2 = _cutoffs.derive_pair_params([_sc_lattice_frame()], ["Ar"], nlayers=2)
    assert result_2["box_safety_bound"] == pytest.approx(result_1["box_safety_bound"] * 2)


def test_output_keys_are_json_safe_strings():
    import json

    result = _cutoffs.derive_pair_params([_sc_lattice_frame()], ["Ar"])
    json.dumps(result)  # must not raise
    assert "Ar-Ar" in result["pairs"]
