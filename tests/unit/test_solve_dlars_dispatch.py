"""solve.py's dlars/dlasso dispatch validation and the
_ensure_canonical_link helper, with stages._dlars_hpc.run_dlars_hpc itself
mocked out (its own control-flow logic is covered by
tests/unit/test_dlars_hpc.py) -- this file only checks that solve.py
requires the right inputs and wires them through correctly."""

from types import SimpleNamespace

import pytest

from agentic_chimes.stages import solve


def _args(**overrides):
    defaults = dict(
        A=None, b=None, header=None, map=None, dim=None,
        algorithm="dlars", alpha=1e-5, eps=1e-5, weights=None, folds=4,
        normalize=True, split_files=False,
        machine=None, queue="batch", walltime_hours=2.0, nodes=1, ntasks_per_node=None,
        poll_interval_s=60, output_dir=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_dlars_requires_machine(tmp_path):
    a_txt = tmp_path / "A.txt"
    a_txt.write_text("1 2\n")
    (tmp_path / "b.txt").write_text("1\n")
    (tmp_path / "params.header").write_text("")
    (tmp_path / "ff_groups.map").write_text("")
    (tmp_path / "dim.txt").write_text("1 2\n")

    with pytest.raises(ValueError, match="requires --machine"):
        solve.run(_args(A=str(a_txt), b=str(tmp_path / "b.txt"), header=str(tmp_path / "params.header"), map=str(tmp_path / "ff_groups.map"), dim=str(tmp_path / "dim.txt"), machine=None, output_dir=str(tmp_path)))


def test_dlars_requires_dim(tmp_path):
    for name in ("A.txt", "b.txt", "params.header", "ff_groups.map"):
        (tmp_path / name).write_text("x\n")

    with pytest.raises(ValueError, match="requires --dim"):
        solve.run(
            _args(
                A=str(tmp_path / "A.txt"), b=str(tmp_path / "b.txt"),
                header=str(tmp_path / "params.header"), map=str(tmp_path / "ff_groups.map"),
                dim=None, machine="dane", output_dir=str(tmp_path),
            )
        )


def test_dlars_dispatches_to_dlars_hpc_with_resolved_paths(tmp_path, monkeypatch):
    for name in ("A.txt", "b.txt", "params.header", "ff_groups.map", "dim.txt"):
        (tmp_path / name).write_text("x\n")

    captured = {}

    def fake_run_dlars_hpc(**kwargs):
        captured.update(kwargs)
        return {"job_id": "1", "cliff_detected": False, "cliff_report": None, "params": str(tmp_path / "params.txt"), "log": str(tmp_path / "dlars.log")}

    import agentic_chimes.stages._dlars_hpc as dlars_hpc_mod

    monkeypatch.setattr(dlars_hpc_mod, "run_dlars_hpc", fake_run_dlars_hpc)
    monkeypatch.setattr(solve, "_dlars_hpc", dlars_hpc_mod)

    result = solve.run(
        _args(
            A=str(tmp_path / "A.txt"), b=str(tmp_path / "b.txt"),
            header=str(tmp_path / "params.header"), map=str(tmp_path / "ff_groups.map"),
            dim=str(tmp_path / "dim.txt"), machine="dane", algorithm="dlars", alpha=2e-5,
            output_dir=str(tmp_path),
        )
    )

    assert result["algorithm"] == "dlars"
    assert captured["alpha"] == 2e-5
    assert captured["header"] == str((tmp_path / "params.header").resolve())
    assert captured["profile"].name == "dane"


def test_ensure_canonical_link_symlinks_when_names_differ(tmp_path):
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    src = other_dir / "myA.txt"
    src.write_text("data\n")

    work_dir = tmp_path / "work"
    work_dir.mkdir()

    solve._ensure_canonical_link(str(src), work_dir, "A.txt")

    dst = work_dir / "A.txt"
    assert dst.is_symlink()
    assert dst.resolve() == src.resolve()


def test_ensure_canonical_link_noop_when_already_canonical(tmp_path):
    work_dir = tmp_path
    a_txt = work_dir / "A.txt"
    a_txt.write_text("data\n")

    # should not raise or touch the file when src already IS work_dir/A.txt
    solve._ensure_canonical_link(str(a_txt), work_dir, "A.txt")
    assert a_txt.read_text() == "data\n"
    assert not a_txt.is_symlink()
