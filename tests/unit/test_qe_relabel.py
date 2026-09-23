"""qe-relabel tested end to end with --dry-run (no Slurm, no built pw.x
needed for submit) and a synthetic pw.out (no live QE needed for collect)."""

import json
from types import SimpleNamespace

import pytest

from agentic_chimes.io import xyzf as xyzf_io
from agentic_chimes.stages import qe_relabel

SYNTHETIC_PW_OUT = """
     Program PWSCF starts
     convergence has been achieved in   3 iterations
!    total energy              =      -5.00000000 Ry
     Forces acting on atoms (cartesian axes, Ry/au):
     atom    1 type  1   force =     0.01000000    0.02000000    0.03000000
     atom    2 type  2   force =    -0.01000000   -0.02000000   -0.03000000
     JOB DONE.
"""


@pytest.fixture
def structure_pool(tmp_path):
    frames = [
        xyzf_io.Frame(symbols=["C", "H"], positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], forces=[[0, 0, 0]] * 2, box=[10.0, 10.0, 10.0]),
        xyzf_io.Frame(symbols=["C", "H"], positions=[[0.0, 0.0, 0.0], [1.3, 0.0, 0.0]], forces=[[0, 0, 0]] * 2, box=[10.0, 10.0, 10.0]),
    ]
    path = tmp_path / "pool.xyzf"
    xyzf_io.write_xyzf(frames, path)
    return path


@pytest.fixture
def pseudos(tmp_path):
    c = tmp_path / "C.upf"
    h = tmp_path / "H.upf"
    c.touch()
    h.touch()
    return {"C": str(c), "H": str(h)}


def _submit_args(structure_pool, pseudos, out_dir, **overrides):
    defaults = dict(
        structure_xyzf=str(structure_pool),
        frame_indices=None,
        elements=["C", "H"],
        masses={"C": 12.011, "H": 1.008},
        pseudopotentials=pseudos,
        ecutwfc=60.0,
        ecutrho=None,
        kpoints=[1, 1, 1],
        smearing="gaussian",
        degauss=0.01,
        conv_thr=1e-8,
        machine="dane",
        queue="debug",
        walltime_hours=1.0,
        nodes=1,
        ntasks_per_node=None,
        collect=None,
        dry_run=True,
        output_dir=str(out_dir),
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_submit_dry_run_writes_pw_in_for_each_frame(tmp_path, structure_pool, pseudos):
    out_dir = tmp_path / "out"
    result = qe_relabel.run(_submit_args(structure_pool, pseudos, out_dir))

    assert result["n_frames"] == 2
    assert result["dry_run"] is True
    assert result["job_id"] is None

    pw_in = (out_dir / "frame_0000" / "pw.in").read_text()
    assert "calculation = 'scf'" in pw_in
    assert "nat = 2" in pw_in
    assert "ntyp = 2" in pw_in
    assert "ATOMIC_POSITIONS angstrom" in pw_in
    assert "C  0.0  0.0  0.0" in pw_in
    assert "K_POINTS automatic" in pw_in
    assert "1 1 1 0 0 0" in pw_in

    # pseudopotentials symlinked in
    assert (out_dir / "frame_0000" / "C.upf").is_symlink()
    assert (out_dir / "frame_0000" / "H.upf").is_symlink()


def test_submit_dry_run_job_script_has_dane_guardrail(tmp_path, structure_pool, pseudos):
    out_dir = tmp_path / "out"
    result = qe_relabel.run(_submit_args(structure_pool, pseudos, out_dir))
    script = (out_dir / "run.cmd").read_text()
    assert "--ntasks-per-node 112" in script
    assert "pw.x -in pw.in" in script
    # exactly one module load line (regression guard: qe_relabel must not
    # duplicate the module load hpc.submit_job's own renderer already adds)
    assert script.count("module load") == 1


def test_submit_requires_machine(tmp_path, structure_pool, pseudos):
    out_dir = tmp_path / "out"
    with pytest.raises(ValueError, match="machine"):
        qe_relabel.run(_submit_args(structure_pool, pseudos, out_dir, machine=None))


def test_collect_round_trip(tmp_path, structure_pool, pseudos):
    out_dir = tmp_path / "out"
    qe_relabel.run(_submit_args(structure_pool, pseudos, out_dir))

    (out_dir / "frame_0000" / "pw.out").write_text(SYNTHETIC_PW_OUT)
    # frame_0001/pw.out intentionally left missing

    collect_args = SimpleNamespace(collect=str(out_dir))
    result = qe_relabel.run(collect_args)

    assert result["n_total"] == 2
    assert result["n_converged"] == 1
    assert result["n_missing_or_failed"] == 1
    assert result["report"][1]["status"] == "missing_output"

    labeled = xyzf_io.read_xyzf(result["labeled_xyzf"])
    assert len(labeled) == 1
    assert labeled[0].energy == pytest.approx(-5.0 * 13.605693009 * 23.0605)
    assert labeled[0].forces[0] == pytest.approx([0.005, 0.01, 0.015])
    assert labeled[0].symbols == ["C", "H"]  # geometry carried over from the original structure


def test_collect_without_manifest_raises(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        qe_relabel.run(SimpleNamespace(collect=str(empty_dir)))
