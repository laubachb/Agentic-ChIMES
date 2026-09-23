"""auto-build tested three ways:
  - a real end-to-end run (labeled_xyzf -> split -> cutoffs -> sweep ->
    chosen), against the same bundled fixture the quickstart uses --
    skipped cleanly if chimes_lsq/chimescalc haven't been built yet.
  - the QE-submission --dry-run branch, which needs no built binaries at
    all (mirrors test_qe_relabel.py's dry-run coverage).
  - the `stabilize` ALC-0 staging step, against a fake al_driver fixture
    (mirrors test_al_run.py's pattern), so it needs no real al_driver
    study either.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic_chimes import config
from agentic_chimes.io import xyzf as xyzf_io
from agentic_chimes.stages import auto_build

FM_FIXTURE = config.CHIMES_LSQ_ROOT / "test_suite-lsq" / "test_4atoms.2" / "dump2.xyzf"


def _toolchain_available() -> bool:
    return (
        config.resolve_component("chimes_lsq_bin", required=False) is not None
        and config.resolve_component("chimescalc_lib", required=False) is not None
    )


def _base_args(**overrides):
    defaults = dict(
        unlabeled_xyzf=None,
        labeled_xyzf=str(FM_FIXTURE),
        elements=["C", "H"],
        masses={"C": 12.011, "H": 1.008},
        charges=None,
        pseudopotentials=None,
        ecutwfc=None,
        ecutrho=None,
        kpoints=[1, 1, 1],
        smearing="gaussian",
        degauss=0.01,
        conv_thr=1e-8,
        machine=None,
        queue="batch",
        walltime_hours=4.0,
        nodes=1,
        ntasks_per_node=None,
        holdout_xyzf=None,
        holdout_fraction=0.2,
        split_seed=42,
        s_minim_delta=0.02,
        s_maxim_2b_default=8.0,
        nlayers=1,
        order_grid={"2": [6, 8], "3": [2, 3], "4": [None]},
        algorithm="svd",
        alpha=1e-5,
        fitener="false",
        fitstrs="false",
        max_frames=None,
        stabilize=None,
        dry_run=False,
        output_dir=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.mark.skipif(not _toolchain_available(), reason="chimes_lsq/chimescalc not built; run `chimes-agent setup`")
def test_end_to_end_against_real_fixture(tmp_path):
    result = auto_build.run(_base_args(output_dir=str(tmp_path)))

    assert Path(result["params"]).is_file()
    assert "ENDFILE" in Path(result["params"]).read_text()
    assert result["chosen"]["status"] == "done"
    assert result["chosen"]["rmse_force_kcal_mol_ang"] > 0

    # cutoffs were actually data-driven, not left as defaults
    cutoffs = result["cutoffs"]["pairs"]
    assert set(cutoffs) == {"C-C", "C-H", "H-H"}
    for pair in cutoffs.values():
        assert pair["s_minim"] > 0
        assert pair["s_maxim_2b"] > pair["s_minim"]

    # split actually happened and both sets are non-trivial
    assert result["trace"]["split"]["n_selected"] > 0
    assert result["trace"]["split"]["n_holdout"] > 0

    # the sweep actually swept: 2 x 2 x 1 = 4 points
    assert result["trace"]["sweep"]["n_points"] == 4


@pytest.mark.skipif(not _toolchain_available(), reason="chimes_lsq/chimescalc not built; run `chimes-agent setup`")
def test_uses_provided_holdout_when_given(tmp_path):
    # build a small standalone holdout file up front so auto-build skips its own split
    frames = xyzf_io.read_xyzf(FM_FIXTURE)
    holdout_path = tmp_path / "my_holdout.xyzf"
    xyzf_io.write_xyzf(frames[:20], holdout_path)

    result = auto_build.run(_base_args(holdout_xyzf=str(holdout_path), output_dir=str(tmp_path / "run")))
    assert result["holdout_xyzf"] == str(holdout_path)
    assert "split" not in result["trace"]


def test_requires_elements_and_masses(tmp_path):
    with pytest.raises(ValueError, match="elements and masses"):
        auto_build.run(_base_args(elements=None, output_dir=str(tmp_path)))


def test_requires_some_data_source(tmp_path):
    with pytest.raises(ValueError, match="labeled_xyzf"):
        auto_build.run(_base_args(labeled_xyzf=None, unlabeled_xyzf=None, output_dir=str(tmp_path)))


def test_qe_submit_dry_run_needs_no_binaries(tmp_path):
    pseudos = {"C": str(tmp_path / "C.upf"), "H": str(tmp_path / "H.upf")}
    (tmp_path / "C.upf").touch()
    (tmp_path / "H.upf").touch()

    pool = tmp_path / "pool.xyzf"
    frames = [xyzf_io.Frame(symbols=["C", "H"], positions=[[0, 0, 0], [1.2, 0, 0]], forces=[[0, 0, 0]] * 2, box=[10, 10, 10])]
    xyzf_io.write_xyzf(frames, pool)

    result = auto_build.run(
        _base_args(
            labeled_xyzf=None,
            unlabeled_xyzf=str(pool),
            pseudopotentials=pseudos,
            ecutwfc=60.0,
            machine="dane",
            dry_run=True,
            output_dir=str(tmp_path / "run"),
        )
    )
    assert result["phase"] == "qe_submit_dry_run"
    assert result["trace"]["qe_submit"]["dry_run"] is True
    assert result["trace"]["qe_submit"]["job_id"] is None


@pytest.fixture
def fake_al_driver_root(tmp_path, monkeypatch):
    root = tmp_path / "fake_root"
    main_py_dir = root / "codes" / "al_driver-LLfork" / "src"
    main_py_dir.mkdir(parents=True)
    (main_py_dir / "main.py").write_text("import sys\nprint('fake driver, cycles=', sys.argv[1:])\n")

    import agentic_chimes.config as cfg

    monkeypatch.setattr(cfg, "AL_DRIVER_SRC", main_py_dir)
    return root


@pytest.mark.skipif(not _toolchain_available(), reason="chimes_lsq/chimescalc not built; run `chimes-agent setup`")
def test_stabilize_stages_alc0_and_launches_via_auto_build(tmp_path, fake_al_driver_root):
    """Runs the real auto-build pipeline end to end (small grid) with a
    `stabilize` block set, exercising auto_build.py's own Phase 5 code
    path directly -- not a hand-copied replica of it."""
    al_study = tmp_path / "al_study"
    al_study.mkdir()
    (al_study / "config.py").write_text("# fake config\n")
    alc0_dir = tmp_path / "alc0"

    result = auto_build.run(
        _base_args(
            output_dir=str(tmp_path / "run"),
            stabilize={"alc0_dir": str(alc0_dir), "al_run_work_dir": str(al_study), "cycles": [0]},
        )
    )

    assert (alc0_dir / "fm_setup.in").is_file()
    assert "TRJFILE" in (alc0_dir / "fm_setup.in").read_text() or (alc0_dir / "fm_setup.in").stat().st_size > 0
    assert any(alc0_dir.glob("*.xyzf"))
    assert result["stabilize"]["status"] == "launched"
    assert Path(result["stabilize"]["log"]).is_file()
