"""al-batch ranking, al-merge duplicate handling, al-status verdicts,
learning-curve subsets/tail fit, QUESTS stage helpers (mocked where the
calculator or QUESTS would be needed)."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from agentic_chimes.io import xyzf as xyzf_io
from agentic_chimes.stages import al_batch, al_merge, al_status, learning_curve


def _frames(n, seed=0, natoms=8, box=8.0, sym="Cu"):
    rng = np.random.default_rng(seed)
    return [xyzf_io.Frame(symbols=[sym] * natoms, positions=rng.uniform(0, box, size=(natoms, 3)).tolist(),
                          forces=[[0, 0, 0]] * natoms, box=[box] * 3, energy=-1.0 * natoms) for _ in range(n)]


def test_ranks_are_normalized_and_nan_safe():
    r = al_batch._ranks([3.0, np.nan, 1.0, 2.0])
    assert r.tolist() == [1.0, 0.0, 0.0, 0.5]


def test_nested_subsets_are_nested_and_keep_closest_contacts():
    frames = _frames(20, seed=3)
    subs = learning_curve.nested_subsets(frames, [0.25, 0.5, 1.0], seed=0)
    sizes = [len(idx) for _, idx in subs]
    assert sizes[-1] == 20 and sizes[0] < sizes[1] < sizes[2]
    assert set(subs[0][1]) <= set(subs[1][1]) <= set(subs[2][1])
    from agentic_chimes.stages.dataset_select import closest_contact_frames

    assert closest_contact_frames(frames) <= set(subs[0][1])


def test_tail_fit_extrapolates_a_power_law():
    ns = [25, 50, 100, 200]
    errs = [0.8 * n ** -0.3 for n in ns]
    t = learning_curve.fit_tail(ns, errs)
    assert t["slope"] == pytest.approx(-0.3, abs=1e-6)
    assert t["err_at_2x"] == pytest.approx(0.8 * 400 ** -0.3, abs=1e-4)   # reported to 4 decimals


def _al_dir(tmp_path, rounds):
    for k, spec in rounds.items():
        d = tmp_path / f"round{k}"
        for sub, payload in spec.items():
            f = {"md": d / "md" / "run" / "md_check.json", "quests": d / "quests" / "quests.json",
                 "merge": d / "merge" / "data_manifest.json"}[sub]
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(payload))
    return tmp_path


def test_al_status_verdicts(tmp_path):
    stable = {"models": [{"stable_at_all_temperatures": True, "unstable_temperatures": [], "below_inner_cutoff_frames": 0,
                          "max_close_contact_fraction": 0.1}]}
    unstable = {"models": [{"stable_at_all_temperatures": False, "unstable_temperatures": [1200.0], "below_inner_cutoff_frames": 7,
                            "max_close_contact_fraction": 0.9}]}
    d = _al_dir(tmp_path / "a", {1: {"md": unstable, "quests": {"fraction_novel_frames": 0.9}}})
    r = al_status.run(SimpleNamespace(study=None, al_dir=str(d), base_manifest=None, base_params=None, plot=False))
    assert r["verdict"] == "CONTINUE" and any("inner cutoff" in x for x in r["reasons"])
    d = _al_dir(tmp_path / "b", {1: {"md": unstable}, 2: {"md": stable, "quests": {"fraction_novel_frames": 0.02}}})
    r = al_status.run(SimpleNamespace(study=None, al_dir=str(d), base_manifest=None, base_params=None, plot=False))
    assert r["verdict"] == "CONVERGED" and (d / "AL_STATUS.md").is_file()
    r = al_status.run(SimpleNamespace(study=None, al_dir=str(tmp_path / "empty"), base_manifest=None, base_params=None, plot=False))
    assert r["verdict"] == "NO_ROUNDS"


def test_al_merge_refuses_holdout_duplicates_and_drops_training_duplicates(tmp_path):
    train, hold, fresh = _frames(4, seed=1), _frames(2, seed=2), _frames(2, seed=3)
    base = tmp_path / "base"
    base.mkdir()
    xyzf_io.write_xyzf(train, base / "train.xyzf")
    xyzf_io.write_xyzf(hold, base / "holdout.xyzf")
    prov = {"source": "x", "label_policy": "source", "level_of_theory": {"methods": {"DFT": 1}, "software": {}}}
    (base / "provenance.json").write_text(json.dumps(prov))
    manifest = {"elements": ["Cu"], "train_xyzf": str(base / "train.xyzf"), "holdout_xyzf": str(base / "holdout.xyzf"),
                "fit_hints": {"fitener": True, "fitstrs": False}, "sources": [prov]}
    (base / "data_manifest.json").write_text(json.dumps(manifest))

    def batch(name, frames):
        d = tmp_path / name
        d.mkdir()
        xyzf_io.write_xyzf(frames, d / "labeled.xyzf")
        (d / "provenance.json").write_text(json.dumps(prov))
        return str(d / "labeled.xyzf")

    with pytest.raises(ValueError, match="holdout"):
        al_merge.run(SimpleNamespace(data_manifest=str(base / "data_manifest.json"), new_xyzf=[batch("leak", hold[:1] + fresh)],
                                     cycle=1, allow_mixed_theory=False, output_dir=str(tmp_path / "m1")))
    r = al_merge.run(SimpleNamespace(data_manifest=str(base / "data_manifest.json"), new_xyzf=[batch("dup", train[:2] + fresh)],
                                     cycle=1, allow_mixed_theory=False, output_dir=str(tmp_path / "m2")))
    assert r["n_added"] == 2 and r["n_duplicates_dropped"] == 2 and r["n_train"] == 6


def test_quests_frame_scores_and_greedy_selection_logic():
    pytest.importorskip("quests")
    from agentic_chimes.stages import quests_stage

    dh = np.array([0.1, 5.0, -0.5, 0.2, 0.3])
    owner = np.array([0, 0, 1, 1, 1])
    assert quests_stage.frame_scores(dh, owner, 2, "max").tolist() == [5.0, 0.3]
    assert quests_stage.frame_scores(dh, owner, 2, "mean")[1] == pytest.approx(0.0)
