"""Stress conventions end to end: ChIMES xyzf stores GPa with the pressure
sign (positive = compressed)."""

import numpy as np
import pytest

from agentic_chimes.converters import qe2xyzf
from agentic_chimes.data_sources import convert
from agentic_chimes.io import xyzf as xyzf_io

QE_STRESS = """
     Computing stress (Cartesian axis) and pressure

          total   stress  (Ry/bohr**3)                   (kbar)     P=      123.40
   0.00083888   0.00000100   0.00000200          123.40        0.15        0.29
   0.00000100   0.00083888   0.00000300            0.15      123.40        0.44
   0.00000200   0.00000300   0.00083888            0.29        0.44      123.40

"""


def test_qe_stress_kbar_to_gpa_pressure_sign():
    res = qe2xyzf.parse_pwx_output(QE_STRESS)
    assert res.stress_kbar[0] == [123.40, 0.15, 0.29]
    base = xyzf_io.Frame(symbols=["Cu"], positions=[[0, 0, 0]], forces=[[0, 0, 0]], box=[3, 3, 3])
    text = "!    total energy              =     -10.0 Ry\n" + \
        "     atom    1 type  1   force =     0.0 0.0 0.0\n" + QE_STRESS
    frame, _ = qe2xyzf.frame_from_qe_output(base, text)
    assert frame.stress == pytest.approx([12.34, 12.34, 12.34, 0.015, 0.029, 0.044])


def test_ase_compressed_crystal_gives_positive_chimes_pressure_in_any_orientation():
    from ase.build import bulk
    from ase.calculators.emt import EMT

    at = bulk("Cu", "fcc", a=3.4)            # compressed, primitive (triclinic) cell
    at.calc = EMT()
    sigma = at.get_stress(voigt=False)       # ASE: Cauchy sign, negative when compressed
    fr = convert.to_frame(at.get_chemical_symbols(), at.cell[:], at.positions, stress_ev_ang3=sigma)
    p = sum(fr.stress[:3]) / 3
    assert p > 0
    assert p == pytest.approx(-np.trace(sigma) / 3 * 160.21766208, rel=1e-9)   # trace survives the rotation
    fr2 = convert.to_frame(at.get_chemical_symbols(), at.cell[:], at.positions, stress_ev_ang3=-sigma,
                           stress_sign="pressure")
    assert fr2.stress == pytest.approx(fr.stress)


def test_stress_sign_check_detects_flipped_data():
    frames = []
    for v in np.linspace(10, 14, 8):
        L = v ** (1 / 3)
        p = 50.0 - 4.0 * v                   # pressure falls as volume rises
        frames.append(xyzf_io.Frame(symbols=["Cu"], positions=[[0, 0, 0]], forces=[[0, 0, 0]], box=[L, L, L],
                                    stress=[p, p, p, 0, 0, 0]))
    assert convert.stress_sign_check(frames)["verdict"] == "ok"
    for f in frames:
        f.stress = [-x for x in f.stress]
    assert convert.stress_sign_check(frames)["verdict"] == "flipped"
    assert convert.stress_sign_check(frames[:3])["verdict"] == "unverified"
