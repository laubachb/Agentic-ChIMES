"""Unit tests for dataset-select using a small synthetic frame pool (no
codes/ dependency, so these run unconditionally in CI)."""

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic_chimes.io import xyzf as xyzf_io
from agentic_chimes.stages import dataset_select


def _make_pool(tmp_path: Path) -> Path:
    """40 frames: 15 C-only, 15 H-only, 10 C+H mixed, each with a distinct
    per-atom energy so composition classes are cleanly separable and the
    energy descriptor has real signal."""
    frames = []
    for i in range(15):
        frames.append(xyzf_io.Frame(symbols=["C", "C"], positions=[[0, 0, 0], [1, 0, 0]], forces=[[0, 0, 0]] * 2, box=[10, 10, 10], energy=-100.0 + i))
    for i in range(15):
        frames.append(xyzf_io.Frame(symbols=["H", "H"], positions=[[0, 0, 0], [1, 0, 0]], forces=[[0, 0, 0]] * 2, box=[10, 10, 10], energy=-10.0 + i))
    for i in range(10):
        frames.append(xyzf_io.Frame(symbols=["C", "H"], positions=[[0, 0, 0], [1, 0, 0]], forces=[[0, 0, 0]] * 2, box=[10, 10, 10], energy=-50.0 + i))

    path = tmp_path / "pool.xyzf"
    xyzf_io.write_xyzf(frames, path)
    return path


def _run(tmp_path, **kwargs):
    out_dir = tmp_path / "out"
    args = SimpleNamespace(output_dir=str(out_dir), **kwargs)
    return dataset_select.run(args)


def test_fps_disjoint_and_complete(tmp_path):
    pool = _make_pool(tmp_path)
    result = _run(tmp_path, frames=str(pool), method="fps", n_select=10, holdout_fraction=None, seed=1, descriptor="composition")
    sel, hold = set(result["selected_indices"]), set(result["holdout_indices"])
    assert len(sel) == 10
    assert sel.isdisjoint(hold)
    assert sel | hold == set(range(40))


def test_fps_deterministic_given_seed(tmp_path):
    pool = _make_pool(tmp_path)
    r1 = _run(tmp_path / "a", frames=str(pool), method="fps", n_select=12, holdout_fraction=None, seed=99, descriptor="composition")
    r2 = _run(tmp_path / "b", frames=str(pool), method="fps", n_select=12, holdout_fraction=None, seed=99, descriptor="composition")
    assert r1["selected_indices"] == r2["selected_indices"]


def test_fps_composition_covers_all_classes(tmp_path):
    """With a composition descriptor and pure-class clusters, FPS starting
    from any point should still spread across all three composition
    classes well before selecting 15/40 frames (it's the farthest-point
    property) -- a regression guard against a descriptor that accidentally
    collapses composition to a single dimension."""
    pool = _make_pool(tmp_path)
    result = _run(tmp_path, frames=str(pool), method="fps", n_select=15, holdout_fraction=None, seed=3, descriptor="composition")
    frames = xyzf_io.read_xyzf(pool)
    classes = {tuple(sorted(set(frames[i].symbols))) for i in result["selected_indices"]}
    assert classes == {("C",), ("H",), ("C", "H")}


def test_random_respects_holdout_fraction(tmp_path):
    pool = _make_pool(tmp_path)
    result = _run(tmp_path, frames=str(pool), method="random", n_select=None, holdout_fraction=0.25, seed=5)
    assert result["n_selected"] == 30  # round(40 * 0.75)
    assert result["n_holdout"] == 10


def test_stratified_holdout_every_class_in_both_splits(tmp_path):
    pool = _make_pool(tmp_path)
    result = _run(tmp_path, frames=str(pool), method="stratified_holdout", n_select=None, holdout_fraction=0.2, seed=11)
    frames = xyzf_io.read_xyzf(pool)

    sel_classes = {tuple(sorted(set(frames[i].symbols))) for i in result["selected_indices"]}
    hold_classes = {tuple(sorted(set(frames[i].symbols))) for i in result["holdout_indices"]}
    assert sel_classes == {("C",), ("H",), ("C", "H")}
    assert hold_classes == {("C",), ("H",), ("C", "H")}
    assert result["n_selected"] + result["n_holdout"] == 40


def test_energy_descriptor_requires_energy_on_all_frames(tmp_path):
    pool = _make_pool(tmp_path)
    # append one frame with no energy
    frames = xyzf_io.read_xyzf(pool)
    frames.append(xyzf_io.Frame(symbols=["C"], positions=[[0, 0, 0]], forces=[[0, 0, 0]], box=[10, 10, 10], energy=None))
    xyzf_io.write_xyzf(frames, pool)

    with pytest.raises(ValueError, match="requires every frame to have an energy"):
        _run(tmp_path, frames=str(pool), method="fps", n_select=5, holdout_fraction=None, seed=1, descriptor="energy")


def test_output_xyzf_files_are_readable_and_correct_size(tmp_path):
    pool = _make_pool(tmp_path)
    result = _run(tmp_path, frames=str(pool), method="random", n_select=8, holdout_fraction=None, seed=2)
    selected = xyzf_io.read_xyzf(result["selected_xyzf"])
    holdout = xyzf_io.read_xyzf(result["holdout_xyzf"])
    assert len(selected) == 8
    assert len(holdout) == 32


def test_n_select_out_of_range_raises(tmp_path):
    pool = _make_pool(tmp_path)
    with pytest.raises(ValueError, match="out of range"):
        _run(tmp_path, frames=str(pool), method="random", n_select=1000, holdout_fraction=None, seed=1)
