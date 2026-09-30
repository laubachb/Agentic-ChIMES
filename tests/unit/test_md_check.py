"""md-check analysis helpers (no LAMMPS needed)."""

import numpy as np
import pytest

from agentic_chimes.stages import md_check

DUMP_TRI = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS xy xz yz pp pp pp
0.0 12.0 2.0
0.0 10.0 0.0
0.0 10.0 0.0
ITEM: ATOMS id type x y z fx fy fz
1 1 0.0 0.0 0.0 0 0 0
2 2 1.0 2.0 3.0 0 0 0
"""


def test_parse_dump_triclinic_bounds(tmp_path):
    p = tmp_path / "dump.out"
    p.write_text(DUMP_TRI)
    (syms, cell, pos), = md_check.parse_dump(p, ["Cu", "Zr"])
    assert syms == ["Cu", "Zr"]
    # LAMMPS reports the bounding box: xhi_bound = xhi + xy for xy > 0
    assert np.allclose(cell, [[10.0, 0, 0], [2.0, 10.0, 0], [0, 0, 10.0]])
    assert np.allclose(pos[1], [1.0, 2.0, 3.0])


def test_pair_inner_cutoffs_reads_both_orders(tmp_path):
    p = tmp_path / "params.txt"
    p.write_text("junk\n# PAIRIDX #\t# ATM_TY1 #\n\t0 Cu Cu 2.235 8 MORSE 2.51\n\t1 Cu Zr 2.263 8 MORSE 2.79\n\nrest\n")
    inner = md_check.pair_inner_cutoffs(p)
    assert inner["Cu-Cu"] == 2.235 and inner["Zr-Cu"] == inner["Cu-Zr"] == 2.263


def test_rdf_of_a_crystal_peaks_at_the_neighbor_distance():
    from ase.build import bulk

    at = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((4, 4, 4))
    rdf = md_check.partial_rdf([at], rmax=4.0)
    r, g = rdf["Cu-Cu"]
    assert r[np.argmax(g)] == pytest.approx(3.6 / np.sqrt(2), abs=0.05)
    assert md_check.rdf_distance(rdf, rdf) == {"Cu-Cu": 0.0}
    cp = md_check.closest_pairs(at)
    assert cp["Cu-Cu"] == pytest.approx(3.6 / np.sqrt(2), abs=1e-6)
