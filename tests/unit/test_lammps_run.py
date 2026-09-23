"""Unit tests for lammps_run.py's pure parsing/writing logic (no LAMMPS
binary needed), plus a real end-to-end regression test (skipped cleanly if
`chimes-agent setup --component lammps` hasn't been run) that cross-checks
LAMMPS forces/energy against the same known-good published reference
test_evaluate.py validates the ctypes evaluator against -- a genuine
three-way agreement check (standalone binary / ctypes / LAMMPS)."""

from pathlib import Path

import pytest

from agentic_chimes import config
from agentic_chimes.io import lammps_data
from agentic_chimes.io import xyzf as xyzf_io
from agentic_chimes.stages import lammps_run

FIXTURES_DIR = config.CHIMES_CALCULATOR_ROOT / "serial_interface" / "tests"
CONFIG_XYZ = FIXTURES_DIR / "configurations" / "CHON.testfile_#000.xyz"
PARAMS_TXT = FIXTURES_DIR / "force_fields" / "test_params.CHON.txt"
CHON_MASSES = {"C": 12.011, "H": 1.0079, "O": 15.9994, "N": 14.007}

EXPECTED_ENERGY = -7.8371473
EXPECTED_FORCE_ATOM0 = (-19.8826, 20.5959, 12.0526)


def _lammps_bin_available() -> bool:
    return config.resolve_component("lammps_bin", required=False) is not None


def _read_plain_xyz(path):
    lines = path.read_text().splitlines()
    natoms = int(lines[0].split()[0])
    box9 = [float(x) for x in lines[1].split()]
    symbols, positions = [], []
    for row in lines[2 : 2 + natoms]:
        toks = row.split()
        symbols.append(toks[0])
        positions.append([float(toks[1]), float(toks[2]), float(toks[3])])
    return natoms, box9, symbols, positions


def _make_frame(tmp_path) -> Path:
    natoms, box9, symbols, positions = _read_plain_xyz(CONFIG_XYZ)
    frame = xyzf_io.Frame(symbols=symbols, positions=positions, forces=[[0.0, 0.0, 0.0]] * natoms, box=[box9[0], box9[4], box9[8]])
    path = tmp_path / "structure.xyzf"
    xyzf_io.write_xyzf([frame], path)
    return path


def test_write_lammps_data_structure(tmp_path):
    frame = xyzf_io.Frame(
        symbols=["C", "H", "C"],
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        forces=[[0, 0, 0]] * 3,
        box=[10.0, 10.0, 10.0],
    )
    out = tmp_path / "structure.data"
    lammps_data.write_lammps_data(frame, ["C", "H"], {"C": 12.011, "H": 1.008}, out)
    text = out.read_text()

    assert "3 atoms" in text
    assert "2 atom types" in text
    assert "0.0 10.0 xlo xhi" in text
    assert "1 12.011" in text  # C is type 1
    assert "2 1.008" in text  # H is type 2
    # atom 1 (C) -> type 1, atom 2 (H) -> type 2, atom 3 (C) -> type 1
    atoms_section = text.split("Atoms")[1]
    rows = [ln.split() for ln in atoms_section.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    assert rows[0][:2] == ["1", "1"]
    assert rows[1][:2] == ["2", "2"]
    assert rows[2][:2] == ["3", "1"]


def test_write_lammps_data_rejects_non_ortho(tmp_path):
    frame = xyzf_io.Frame(
        symbols=["C"], positions=[[0, 0, 0]], forces=[[0, 0, 0]],
        box=[[10, 0, 0], [0, 10, 0], [0, 0, 10]], non_ortho=True,
    )
    with pytest.raises(ValueError, match="orthorhombic"):
        lammps_data.write_lammps_data(frame, ["C"], {"C": 12.011}, tmp_path / "x.data")


def test_write_lammps_data_rejects_unknown_element(tmp_path):
    frame = xyzf_io.Frame(symbols=["C", "O"], positions=[[0, 0, 0], [1, 0, 0]], forces=[[0, 0, 0]] * 2, box=[10, 10, 10])
    with pytest.raises(ValueError, match="not in"):
        lammps_data.write_lammps_data(frame, ["C"], {"C": 12.011}, tmp_path / "x.data")


def test_render_input_single_point_contains_key_directives():
    text = lammps_run._render_input("single_point", "structure.data", "params.txt")
    assert "read_data structure.data" in text
    assert "pair_style chimesFF" in text
    assert "pair_coeff * * params.txt" in text
    assert "run 0" in text
    assert "dump_modify 1 sort id" in text


def test_render_input_md_contains_fix_and_run_nsteps():
    text = lammps_run._render_input("md", "structure.data", "params.txt", temperature=350.0, nsteps=500, timestep=0.5, md_seed=7)
    assert "fix 1 all nvt temp 350.0 350.0 100.0" in text
    assert "run 500" in text
    assert "timestep 0.5" in text
    assert "velocity all create 350.0 7" in text


def test_parse_dump_forces_sorted_by_id():
    dump_text = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type x y z fx fy fz
2 1 1.0 0.0 0.0 2.0 2.0 2.0
1 1 0.0 0.0 0.0 1.0 1.0 1.0
"""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "dump.out"
        p.write_text(dump_text)
        forces = lammps_run._parse_dump_forces(p)
    assert forces == [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]  # sorted by atom id, not file order


def test_parse_log_pe_single_point_format():
    log_text = "some preamble\n\n   Step         PotEng    \n         0  -7.8371473    \nLoop time of 1e-06 ...\n"
    assert lammps_run._parse_log_pe(log_text) == pytest.approx(-7.8371473)


def test_parse_log_pe_returns_none_for_multi_column_thermo():
    # md mode's 5-column thermo_style isn't parsed by this single-point-
    # specific helper -- must return None, not misparse a wrong column.
    log_text = "   Step          Temp          PotEng         TotEng         Press     \n         0   300.0   -7.8   -6.5   1.2\n"
    assert lammps_run._parse_log_pe(log_text) is None


@pytest.mark.skipif(not _lammps_bin_available(), reason="lammps not built; run `chimes-agent setup --component lammps`")
def test_lammps_single_point_matches_published_reference(tmp_path):
    structure = _make_frame(tmp_path)
    from types import SimpleNamespace

    args = SimpleNamespace(
        params=str(PARAMS_TXT),
        structure_xyzf=str(structure),
        frame_index=0,
        elements=["C", "H", "O", "N"],
        masses=CHON_MASSES,
        mode="single_point",
        temperature=300.0,
        nsteps=0,
        timestep=1.0,
        md_seed=1,
        lammps_bin=None,
        nprocs=1,
        output_dir=str(tmp_path / "run"),
    )
    result = lammps_run.run(args)

    assert result["energy_kcal_mol"] == pytest.approx(EXPECTED_ENERGY, abs=1e-3)
    f0 = result["forces_kcal_mol_ang"][0]
    assert f0[0] == pytest.approx(EXPECTED_FORCE_ATOM0[0], abs=1e-2)
    assert f0[1] == pytest.approx(EXPECTED_FORCE_ATOM0[1], abs=1e-2)
    assert f0[2] == pytest.approx(EXPECTED_FORCE_ATOM0[2], abs=1e-2)
