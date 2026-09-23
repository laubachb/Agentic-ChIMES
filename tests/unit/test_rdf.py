"""io/rdf.py tested against a simple cubic (SC) lattice -- a real
physically-motivated system with exactly known shell distances (1st
shell = a, 2nd = a*sqrt(2), 3rd = a*sqrt(3)), not just arbitrary
hand-placed numbers."""

import math

import pytest

from agentic_chimes.io import rdf
from agentic_chimes.io.xyzf import Frame

A = 3.0
N = 4


def _sc_lattice_frame():
    positions = [[i * A, j * A, k * A] for i in range(N) for j in range(N) for k in range(N)]
    box = [N * A, N * A, N * A]
    return Frame(symbols=["Ar"] * len(positions), positions=positions, forces=[[0, 0, 0]] * len(positions), box=box)


def test_pair_min_distance_matches_lattice_constant():
    mins = rdf.pair_min_distance([_sc_lattice_frame()], ["Ar"])
    assert mins[("Ar", "Ar")] == pytest.approx(A, abs=1e-9)


def test_min_box_dimension():
    assert rdf.min_box_dimension([_sc_lattice_frame()]) == pytest.approx(N * A)


def test_rdf_shape_peaks_and_minima_at_expected_shells():
    rdfs = rdf.pair_rdf([_sc_lattice_frame()], ["Ar"], bin_width=0.05)
    r = rdfs[("Ar", "Ar")]

    shell1, shell2, shell3 = A, A * math.sqrt(2), A * math.sqrt(3)

    peak = r.first_peak()
    assert peak == pytest.approx(shell1, abs=0.3)

    min1 = r.first_minimum_after_peak()
    assert shell1 < min1 < shell2

    min2 = r.second_minimum()
    assert shell2 - 0.3 < min2 < shell3


def test_pbc_minimum_image_wraps_correctly():
    # two atoms placed near opposite faces of the box are actually close
    # together through the periodic boundary -- the minimum-image distance
    # must reflect that wrap, not the raw (unwrapped) separation.
    box = [10.0, 10.0, 10.0]
    frame = Frame(symbols=["C", "C"], positions=[[0.2, 5.0, 5.0], [9.8, 5.0, 5.0]], forces=[[0, 0, 0]] * 2, box=box)
    mins = rdf.pair_min_distance([frame], ["C"])
    # true wrapped separation is 0.4 (0.2 -> 0 -> -0.2, i.e. 10 - 9.6), NOT 9.6
    assert mins[("C", "C")] == pytest.approx(0.4, abs=1e-9)


def test_non_orthorhombic_frame_rejected():
    frame = Frame(
        symbols=["C", "C"], positions=[[0, 0, 0], [1, 0, 0]], forces=[[0, 0, 0]] * 2,
        box=[[10, 0, 0], [0, 10, 0], [0, 0, 10]], non_ortho=True,
    )
    with pytest.raises(ValueError, match="orthorhombic"):
        rdf.pair_min_distance([frame], ["C"])
    with pytest.raises(ValueError, match="orthorhombic"):
        rdf.min_box_dimension([frame])


def test_no_common_pairs_returns_empty():
    frame = Frame(symbols=["C"], positions=[[0, 0, 0]], forces=[[0, 0, 0]], box=[10, 10, 10])
    assert rdf.pair_min_distance([frame], ["C"]) == {}
