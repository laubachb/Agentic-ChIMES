"""Validates the ctypes wrapper loading in stages.evaluate against a real,
published force field from codes/chimes_calculator-LLfork's own
serial_interface/tests fixtures (force_fields/test_params.CHON.txt +
configurations/CHON.testfile_#000.xyz), cross-checked by hand against the
standalone `chimescalc` binary and expected_output/CHON.testfile.000.xyz.dat
(energy -7.8371469 kcal/mol; forces flattened per atom).

Skips (does not fail) if `chimes-agent setup --component chimes_calculator`
hasn't been run yet on this machine, since that's an environment
precondition, not a code defect.
"""

import pytest

from agentic_chimes import config
from agentic_chimes.stages import evaluate as ev

FIXTURES_DIR = config.CHIMES_CALCULATOR_ROOT / "serial_interface" / "tests"
CONFIG_XYZ = FIXTURES_DIR / "configurations" / "CHON.testfile_#000.xyz"
PARAMS_TXT = FIXTURES_DIR / "force_fields" / "test_params.CHON.txt"

EXPECTED_ENERGY = -7.8371469
EXPECTED_FORCE_ATOM0 = (-19.88263, 20.59586, 12.05260)


def _chimescalc_lib_available() -> bool:
    return config.resolve_component("chimescalc_lib", required=False) is not None


def _read_plain_xyz(path):
    """The chimes_calculator test-suite's own config format: natoms line,
    then a flattened 3x3 cell (9 numbers), then `<elem> x y z` rows -- NOT
    the same as ChIMES lsq's `.xyzf` training format (io.xyzf), which is why
    this reader lives here rather than in the shared io module."""
    lines = path.read_text().splitlines()
    natoms = int(lines[0].split()[0])
    box9 = [float(x) for x in lines[1].split()]
    cell = (box9[0:3], box9[3:6], box9[6:9])
    symbols, positions = [], []
    for row in lines[2 : 2 + natoms]:
        toks = row.split()
        symbols.append(toks[0])
        positions.append([float(toks[1]), float(toks[2]), float(toks[3])])
    return natoms, cell, symbols, positions


@pytest.mark.skipif(not _chimescalc_lib_available(), reason="chimes_calculator not built; run `chimes-agent setup --component chimes_calculator --machine <name>`")
def test_evaluate_matches_published_reference():
    assert CONFIG_XYZ.is_file() and PARAMS_TXT.is_file(), "vendored fixture missing"

    natoms, (cell_a, cell_b, cell_c), symbols, positions = _read_plain_xyz(CONFIG_XYZ)

    wrapper = ev._load_wrapper()
    ptr = wrapper.chimes_open_instance()
    wrapper.set_chimes_instance(ptr, small=False)
    wrapper.init_chimes_instance(ptr, str(PARAMS_TXT), 0)
    try:
        xcrd = [p[0] for p in positions]
        ycrd = [p[1] for p in positions]
        zcrd = [p[2] for p in positions]
        fx0, fy0, fz0, stress0 = [0.0] * natoms, [0.0] * natoms, [0.0] * natoms, [0.0] * 9
        fx, fy, fz, _stress, energy = wrapper.calculate_chimes_instance(
            ptr, natoms, xcrd, ycrd, zcrd, symbols, cell_a, cell_b, cell_c, 0.0, fx0, fy0, fz0, stress0
        )
    finally:
        wrapper.chimes_close_instance(ptr)

    assert energy == pytest.approx(EXPECTED_ENERGY, abs=1e-3)
    assert fx[0] == pytest.approx(EXPECTED_FORCE_ATOM0[0], abs=1e-3)
    assert fy[0] == pytest.approx(EXPECTED_FORCE_ATOM0[1], abs=1e-3)
    assert fz[0] == pytest.approx(EXPECTED_FORCE_ATOM0[2], abs=1e-3)


@pytest.mark.skipif(not _chimescalc_lib_available(), reason="chimes_calculator not built; run `chimes-agent setup --component chimes_calculator --machine <name>`")
def test_evaluate_stage_rmse_self_consistent():
    """A holdout .xyzf built from the model's own predictions must score
    ~zero RMSE against itself -- exercises stages.evaluate.run end to end
    (not just the wrapper) via a synthetic self-consistent .xyzf."""
    from agentic_chimes.io import xyzf as xyzf_io

    natoms, (cell_a, cell_b, cell_c), symbols, positions = _read_plain_xyz(CONFIG_XYZ)
    wrapper = ev._load_wrapper()
    ptr = wrapper.chimes_open_instance()
    wrapper.set_chimes_instance(ptr, small=False)
    wrapper.init_chimes_instance(ptr, str(PARAMS_TXT), 0)
    try:
        xcrd = [p[0] for p in positions]
        ycrd = [p[1] for p in positions]
        zcrd = [p[2] for p in positions]
        fx0, fy0, fz0, stress0 = [0.0] * natoms, [0.0] * natoms, [0.0] * natoms, [0.0] * 9
        fx, fy, fz, _stress, energy = wrapper.calculate_chimes_instance(
            ptr, natoms, xcrd, ycrd, zcrd, symbols, cell_a, cell_b, cell_c, 0.0, fx0, fy0, fz0, stress0
        )
    finally:
        wrapper.chimes_close_instance(ptr)

    frame = xyzf_io.Frame(
        symbols=symbols,
        positions=positions,
        forces=list(zip(fx, fy, fz)),
        box=[cell_a[0], cell_b[1], cell_c[2]],
        energy=energy,
    )

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        holdout_path = Path(td) / "self.xyzf"
        xyzf_io.write_xyzf([frame], holdout_path)

        class Args:
            params = [str(PARAMS_TXT)]
            holdout_xyzf = str(holdout_path)
            max_frames = None

        result = ev.run(Args())

    assert result["n_frames"] == 1
    assert result["results"][0]["rmse_force_kcal_mol_ang"] < 1e-6
    assert result["results"][0]["rmse_energy_kcal_mol"] < 1e-6
