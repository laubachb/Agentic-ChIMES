"""converters/qe2xyzf.py tested against a hand-crafted, realistic pw.x
stdout fixture (no live QE run needed) with independently hand-computed
expected values -- QE reports Rydberg energies and Ry/bohr forces; the
target .xyzf convention (matching vasp2xyzf.py's contract) is kcal/mol
energies and hartree/bohr forces.
"""

import pytest

from agentic_chimes.converters import qe2xyzf
from agentic_chimes.io.xyzf import Frame

# Deliberately lists atom 2 before atom 1 in the forces block, to verify
# the parser re-sorts by QE's atom index rather than trusting file order.
FAKE_PWX_OUTPUT = """
     Program PWSCF v.7.2 starts on 1Jan2026

     bravais-lattice index     =            0
     lattice parameter (alat)  =       1.0000  a.u.

     iteration #  1     ecut=    60.00 Ry     beta=0.70
     total cpu time spent up to now is        1.2 secs

     total energy              =     -10.00000001 Ry
     estimated scf accuracy    <       0.00000010 Ry

     iteration #  2     ecut=    60.00 Ry     beta=0.70
     total cpu time spent up to now is        2.1 secs

     convergence has been achieved in   2 iterations

!    total energy              =     -10.00000000 Ry

     total all-electron energy =      -100.00000000 Ry

     The total energy is the sum of the following terms:
     one-electron contribution =      -20.00000000 Ry

     Forces acting on atoms (cartesian axes, Ry/au):

     atom    2 type  2   force =    -0.10000000   -0.20000000   -0.30000000
     atom    1 type  1   force =     0.10000000    0.20000000    0.30000000

     Total force =     0.529150   Total SCF correction =     0.000012

          total   stress  (Ry/bohr**3)                   (kbar)     P=       0.00
   0.00000000   0.00000000   0.00000000            0.00      0.00      0.00

     Writing all to output data dir ./out/qe_relabel.save/

     init_run     :      0.10s CPU      0.10s WALL (       1 calls)

     PWSCF        :      2.30s CPU      2.50s WALL

   This run was terminated on:  12: 0: 0   1Jan2026
=------------------------------------------------------------------------=
   JOB DONE.
=------------------------------------------------------------------------=
"""

NOT_CONVERGED_OUTPUT = """
     Program PWSCF v.7.2 starts on 1Jan2026
     convergence NOT achieved after  100 iterations: stopping
"""


def test_parse_pwx_output_energy_takes_last_bang_line():
    # the fixture has two "total energy" lines (an un-prefixed intermediate
    # one and the final "!"-prefixed converged one) -- only the "!" line
    # should match, and its value (not the intermediate one) is used.
    result = qe2xyzf.parse_pwx_output(FAKE_PWX_OUTPUT)
    assert result.energy_ry == pytest.approx(-10.00000000)
    assert result.converged is True


def test_parse_pwx_output_forces_sorted_by_atom_index():
    result = qe2xyzf.parse_pwx_output(FAKE_PWX_OUTPUT)
    assert result.forces_ry_bohr == [
        [0.1, 0.2, 0.3],  # atom 1, even though it appears second in the text
        [-0.1, -0.2, -0.3],  # atom 2
    ]


def test_parse_pwx_output_not_converged():
    result = qe2xyzf.parse_pwx_output(NOT_CONVERGED_OUTPUT)
    assert result.converged is False


def test_frame_from_qe_output_unit_conversion():
    base = Frame(symbols=["C", "H"], positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], forces=[[0, 0, 0]] * 2, box=[10.0, 10.0, 10.0])

    frame, converged = qe2xyzf.frame_from_qe_output(base, FAKE_PWX_OUTPUT)

    assert converged is True
    # hand-computed: -10 Ry * 13.605693009 eV/Ry * 23.0605 kcal-mol/eV
    expected_energy_kcal_mol = -10.0 * 13.605693009 * 23.0605
    assert frame.energy == pytest.approx(expected_energy_kcal_mol, rel=1e-9)

    # hand-computed: Ry/bohr -> hartree/bohr is exactly *0.5 (1 Hartree = 2 Ry)
    assert frame.forces[0] == pytest.approx([0.05, 0.1, 0.15])
    assert frame.forces[1] == pytest.approx([-0.05, -0.1, -0.15])

    # geometry/box carried over unchanged from the base (pre-DFT) frame
    assert frame.symbols == ["C", "H"]
    assert frame.positions == [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    assert frame.box == [10.0, 10.0, 10.0]


def test_frame_from_qe_output_missing_energy_raises():
    base = Frame(symbols=["C"], positions=[[0, 0, 0]], forces=[[0, 0, 0]], box=[10, 10, 10])
    with pytest.raises(ValueError, match="total energy"):
        qe2xyzf.frame_from_qe_output(base, "no energy line here")


def test_frame_from_qe_output_atom_count_mismatch_raises():
    base = Frame(symbols=["C", "H", "O"], positions=[[0, 0, 0]] * 3, forces=[[0, 0, 0]] * 3, box=[10, 10, 10])
    with pytest.raises(ValueError, match="force rows"):
        qe2xyzf.frame_from_qe_output(base, FAKE_PWX_OUTPUT)  # fixture only has 2 force rows
