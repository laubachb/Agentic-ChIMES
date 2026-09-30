"""hyper-analyze on geometry with known answers, the selection rule, and
hyper-search's staging logic with a synthetic fit function (no chimes_lsq)."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("ase")
pytest.importorskip("scipy")

from ase.build import bulk  # noqa: E402

from agentic_chimes.data_sources import convert  # noqa: E402
from agentic_chimes.io import xyzf as xyzf_io  # noqa: E402
from agentic_chimes.stages import _hyper, hyper_search  # noqa: E402


def _fcc_frames(n=6, a=3.61, rattle=0.03, triclinic=False):
    rng = np.random.default_rng(0)
    frames = []
    for _ in range(n):
        at = bulk("Cu", "fcc", a=a) if triclinic else bulk("Cu", "fcc", a=a, cubic=True).repeat((2, 2, 2))
        at.rattle(rattle, rng=rng)
        frames.append(convert.to_frame(at.get_chemical_symbols(), at.cell[:], at.positions, -1.0, np.zeros((len(at), 3))))
    return frames


@pytest.mark.parametrize("triclinic", [False, True])
def test_analyze_finds_fcc_shells(triclinic):
    """fcc a=3.61: first neighbours 2.553 A, second 3.61 A. The primitive
    triclinic cell (1 atom, 2.55 A wide) must give the same answer."""
    an = _hyper.analyze(_fcc_frames(triclinic=triclinic), ["Cu"])
    cu = an["pairs"]["Cu-Cu"]
    assert cu["rdf_peaks"][0] == pytest.approx(2.553, abs=0.06)
    assert cu["suggested"]["morse_lambda"] == pytest.approx(2.553, abs=0.06)
    assert cu["suggested"]["s_minim"] == pytest.approx(cu["min_distance"] - 0.02, abs=1e-3)
    assert 2.6 < an["global"]["first_shell_end"] < 3.6
    c3 = an["candidates"]["s_maxim_3b"]
    assert c3[0] == an["global"]["first_shell_end"] and max(c3) > c3[0] + 0.5  # reaches past the first shell
    if triclinic:
        assert an["thinnest_cell_width"] < 2.2 and an["nlayers_required"]["8.00"] >= 3


def test_nlayers_rule():
    assert _hyper.nlayers_required(3.0, 7.0) == 0
    assert _hyper.nlayers_required(8.0, 2.0) == 4


def _r(score, se=0.0, n_params=10, s2=6.0, o3=0, s3=None, o4=0, s4=None, n_eq=1000):
    return {"status": "done", "holdout_relative_force_error": score, "holdout_relative_force_se": se,
            "holdout_rmse_energy_per_atom": 0.0, "n_params": n_params, "n_equations": n_eq,
            "cfg": {"s_maxim_2b": s2, "order_3b": o3, "s_maxim_3b": s3, "order_4b": o4, "s_maxim_4b": s4}}


def test_select_one_standard_error_rule_prefers_short_cutoffs():
    best = _r(0.40, se=0.03, n_params=50, s2=8.0)
    short_cheap = _r(0.42, n_params=60, s2=5.0)    # within 1 SE, shorter cutoff (more coefficients, still cheaper MD)
    far = _r(0.50, n_params=5, s2=5.0)             # outside the margin
    sel = _hyper.select([best, short_cheap, far], objective="force", energy_weight=0, tolerance=0.01, max_param_ratio=0.5)
    assert sel["chosen"] is short_cheap and "statistically tied" in sel["reason"]


def test_select_respects_parameter_budget():
    over = _r(0.10, n_params=900, n_eq=1000)
    ok = _r(0.40, n_params=100)
    sel = _hyper.select([over, ok], objective="force", energy_weight=0, tolerance=0.03, max_param_ratio=0.5)
    assert sel["chosen"] is ok and sel["n_excluded_over_budget"] == 1


def _fake_run_point(task):
    """Synthetic landscape: 2b best at order 12 / 6 A; 3-body helps a lot at
    6 A and not at the first shell; 4-body helps nothing."""
    c = task["cfg"]
    s = 0.6 + 0.01 * abs(c["order_2b"] - 12) + 0.02 * abs(c["s_maxim_2b"] - 6.0)
    if c.get("order_3b"):
        s -= 0.25 if c["s_maxim_3b"] >= 5.0 else 0.01
        s += 0.02 * abs(c["order_3b"] - 6)
    if c.get("order_4b"):
        s += 0.001
    s *= c.get("lambda_scale", 1.0) ** 0  # lambda neutral
    n = 3 * c["order_2b"] + 60 * bool(c.get("order_3b")) * c.get("order_3b", 0) + 400 * bool(c.get("order_4b"))
    return {"key": _hyper.config_key(c), "cfg": c, "status": "done", "params": task["point_dir"] + "/params.txt",
            "fm_setup_in": task["point_dir"] + "/fm_setup.in", "nlayers": 1, "n_params": n, "n_equations": 5000,
            "train_relative_force_error": s * 0.9, "holdout_relative_force_error": s, "holdout_relative_force_se": 0.005,
            "holdout_rmse_energy_per_atom": 0.5, "holdout_rmse_force": s * 5, "signal": {}, "solver_alpha": 1e-4}


def test_hyper_search_staging(tmp_path, monkeypatch):
    frames = _fcc_frames()
    xyzf_io.write_xyzf(frames, tmp_path / "train.xyzf")
    xyzf_io.write_xyzf(frames[:3], tmp_path / "holdout.xyzf")
    monkeypatch.setattr(_hyper, "run_point", _fake_run_point)
    monkeypatch.setattr(hyper_search.shutil, "copy", lambda *a, **k: None)
    args = SimpleNamespace(
        data_manifest=None, train_xyzf=str(tmp_path / "train.xyzf"), holdout_xyzf=str(tmp_path / "holdout.xyzf"),
        elements=["Cu"], hyper_analysis=None, stages=None, four_body="auto", orders_2b=[8, 10, 12, 14],
        orders_3b=[4, 6, 8], orders_4b=[2, 3], s_maxim_2b=[5.0, 6.0, 7.0], s_maxim_3b=[3.2, 5.0, 6.0],
        s_maxim_4b=[3.2, 4.5], lambda_scales=[0.9, 1.0, 1.1], objective="force", energy_weight=0.1, tolerance=0.03,
        min_gain=0.05, min_signal=1e-9, exclude_inert=False, max_param_ratio=0.5, fitener=False,
        algorithm="nridgecv", alpha=0.0, masses=None, workers=1, machine=None, max_fit_seconds=60,
        output_dir=str(tmp_path / "search"))
    (tmp_path / "search" / "best").mkdir(parents=True)
    res = hyper_search.run(args)
    hp = res["hyperparameters"]
    assert hp["order"]["2"] in (10, 12, 14) and hp["order"]["3"] == 6
    assert hp["special_maxim_3b"] == 5.0          # 5 and 6 tie within noise -> shorter cutoff
    assert "4" not in hp["order"]                  # 4-body adds nothing
    assert hp["pair_cutoffs"]["Cu-Cu"][1] == 6.0
    report = json.loads((tmp_path / "search" / "hyper_report.json").read_text())
    stages = [s["stage"] for s in report["stages"]]
    assert stages[:3] == ["2b", "3b", "4b"] and "lambda" in stages


def test_paired_comparison_separates_consistent_small_differences():
    """Frames vary 10x in difficulty, so each model's own SE is large, but
    model B is 5% worse on *every* frame: paired, that is not a tie."""
    rng = np.random.default_rng(0)
    ref = rng.uniform(1, 100, size=30)
    err_a = 0.16 * ref
    rows_a = [[e, r, 3] for e, r in zip(err_a, ref)]
    rows_b = [[e * 1.1025, r, 3] for e, r in zip(err_a, ref)]  # relative error x1.05
    a = {**_r(0.40, se=0.08, n_params=200, s2=6.0), "holdout_per_frame_force": rows_a}
    b = {**_r(0.42, se=0.08, n_params=20, s2=5.0), "holdout_per_frame_force": rows_b}
    assert _hyper.paired_se(rows_b, rows_a) < 1e-6
    sel = _hyper.select([a, b], objective="force", energy_weight=0, tolerance=0.01, max_param_ratio=0.5)
    assert sel["chosen"] is a            # unpaired (SE 0.08) would have called it a tie and taken b



def _fake_point_4b_and_exclusions(task):
    """4-body helps only with blocklasso (raw lassolars zeroes it); removing
    3-body type 'Cu Cu Cu' hurts, removing 'X X X' costs nothing."""
    c = task["cfg"]
    solver = c.get("solver", task["algorithm"])
    s = 0.6 + 0.01 * abs(c["order_2b"] - 12) + 0.02 * abs(c["s_maxim_2b"] - 6.0)
    n = 3 * c["order_2b"]
    if c.get("order_3b"):
        s -= 0.25 if c["s_maxim_3b"] >= 5.0 else 0.01
        s += 0.02 * abs(c["order_3b"] - 6)
        excluded = {" ".join(e) for e in c.get("exclude_3b") or []}
        if "Cu Cu Cu" in excluded:
            s += 0.2
        n += 60 * c["order_3b"] * (2 - len(excluded)) // 2
    if c.get("order_4b") and solver == "blocklasso":
        s -= 0.08 - 0.01 * abs(c["order_4b"] - 3)
        n += 150
    return {"key": _hyper.config_key(c), "cfg": c, "status": "done", "params": task["point_dir"] + "/params.txt",
            "fm_setup_in": task["point_dir"] + "/fm_setup.in", "nlayers": 1, "n_params": n, "n_equations": 5000,
            "train_relative_force_error": s * 0.9, "holdout_relative_force_error": s, "holdout_relative_force_se": 0.005,
            "holdout_rmse_energy_per_atom": 0.5, "holdout_rmse_force": s * 5, "signal": {}, "solver": solver,
            "solver_alpha": 1e-5, "cluster_coverage": {"3b": {"Cu Cu Cu": {"instances": 500}, "X X X": {"instances": 0}},
                                                        "4b": {"Cu Cu Cu Cu": {"instances": 50}}}}


def test_four_body_solver_sweep_and_exclusions(tmp_path, monkeypatch):
    frames = _fcc_frames()
    xyzf_io.write_xyzf(frames, tmp_path / "train.xyzf")
    xyzf_io.write_xyzf(frames[:3], tmp_path / "holdout.xyzf")
    monkeypatch.setattr(_hyper, "run_point", _fake_point_4b_and_exclusions)
    monkeypatch.setattr(hyper_search.shutil, "copy", lambda *a, **k: None)
    args = SimpleNamespace(
        data_manifest=None, train_xyzf=str(tmp_path / "train.xyzf"), holdout_xyzf=str(tmp_path / "holdout.xyzf"),
        elements=["Cu"], hyper_analysis=None, stages=None, four_body="on", orders_2b=[10, 12, 14],
        orders_3b=[4, 6], orders_4b=[2, 3], s_maxim_2b=[6.0], s_maxim_3b=[5.0, 6.0], s_maxim_4b=[4.5],
        four_body_solvers=["lassolars", "blocklasso"], exclude_rounds=2, lambda_scales=[1.0], objective="force", energy_weight=0.1,
        tolerance=0.01, min_gain=0.05, min_signal=1e-9, exclude_inert=False, max_param_ratio=0.5, fitener=False,
        algorithm="lassolars", alpha=1e-5, masses=None, workers=1, machine=None, max_fit_seconds=60,
        output_dir=str(tmp_path / "search"))
    (tmp_path / "search" / "best").mkdir(parents=True)
    res = hyper_search.run(args)
    hp = res["hyperparameters"]
    assert hp["algorithm"] == "blocklasso" and hp["order"].get("4") == 3
    assert hp["exclude_3b"] == [["X", "X", "X"]]
    report = json.loads((tmp_path / "search" / "hyper_report.json").read_text())
    four = next(s for s in report["stages"] if s["stage"] == "4b")
    assert set(four["four_body_gain_by_solver"]) == {"lassolars", "blocklasso"}
    assert four["four_body_gain_by_solver"]["blocklasso"]["best_four_body_score"] < four["four_body_gain_by_solver"]["lassolars"]["best_four_body_score"]
    excl = next(s for s in report["stages"] if s["stage"] == "exclude")
    rows = {r["type"]: r for r in excl["cluster_types"]}
    assert rows["X X X"]["excluded"] and not rows["Cu Cu Cu"]["excluded"]
    assert rows["Cu Cu Cu"]["score_change_when_removed"]["round1"] > 0.1
    assert "absent" in rows["X X X"]["note"]


def test_md_cost_prefers_fewer_coefficients_over_slightly_shorter_cutoff():
    """Real Cu-Zr tie: 3-body order 6 @ 6.33 A (330 3-body coefficients) vs
    order 4 @ 7.0 A (98). Triplets grow ~1.8x, coefficients drop 3.4x."""
    a = {**_r(0.37, n_params=380, s2=7.0, o3=6, s3=6.33), "signal": {"n_2b": 42, "n_3b": 330}}
    b = {**_r(0.35, n_params=148, s2=7.0, o3=4, s3=7.0), "signal": {"n_2b": 42, "n_3b": 98}}
    assert _hyper.md_cost(b) < 0.6 * _hyper.md_cost(a)
    sel = _hyper.select([a, b], objective="force", energy_weight=0, tolerance=0.1, max_param_ratio=0.5)
    assert sel["chosen"] is b



def test_cluster_coverage_reads_excluded_types(tmp_path):
    (tmp_path / "fm_setup.log").write_text(
        "\tTotal number of configurations contributing to each triplet type:\n"
        "\t\t0\tCuCu CuCu CuCu 100\n"
        "\t\t3\tZrZr ZrZr ZrZr excluded\n"
        "...matrix printing complete:\n")
    cov = _hyper.cluster_coverage(tmp_path)
    assert cov["3b"]["Cu Cu Cu"]["instances"] == 100
    assert cov["3b"]["Zr Zr Zr"] == {"instances": 0, "excluded": True}


def test_hyper_search_smoothing_reaches_every_fit(tmp_path, monkeypatch):
    frames = _fcc_frames()
    xyzf_io.write_xyzf(frames, tmp_path / "train.xyzf")
    xyzf_io.write_xyzf(frames[:3], tmp_path / "holdout.xyzf")
    seen = []

    def spy(task):
        seen.append(task["cfg"].get("fcuttyp", "CUBIC"))
        return _fake_run_point(task)

    monkeypatch.setattr(_hyper, "run_point", spy)
    monkeypatch.setattr(hyper_search.shutil, "copy", lambda *a, **k: None)
    args = SimpleNamespace(
        data_manifest=None, train_xyzf=str(tmp_path / "train.xyzf"), holdout_xyzf=str(tmp_path / "holdout.xyzf"),
        elements=["Cu"], hyper_analysis=None, stages=["2b", "3b"], four_body="off", orders_2b=[8, 10],
        orders_3b=[4, 6], orders_4b=None, s_maxim_2b=[5.0], s_maxim_3b=[3.2], s_maxim_4b=None,
        lambda_scales=[1.0], objective="force", energy_weight=0.1, tolerance=0.03, smoothing="tersoff 0.5",
        min_gain=0.05, min_signal=1e-9, exclude_inert=False, max_param_ratio=0.5, fitener=False,
        algorithm="nridgecv", alpha=0.0, masses=None, workers=1, machine=None, max_fit_seconds=60,
        output_dir=str(tmp_path / "search"))
    (tmp_path / "search" / "best").mkdir(parents=True)
    res = hyper_search.run(args)
    assert seen and set(seen) == {"TERSOFF 0.5"}
    assert res["hyperparameters"]["fcuttyp"] == "TERSOFF 0.5"
    report = json.loads((tmp_path / "search" / "hyper_report.json").read_text())
    assert not any("CUBIC smoothing" in n for n in report["notes"])

    args.smoothing = "TERSOFF 1.5"
    args.output_dir = str(tmp_path / "bad")
    with pytest.raises(ValueError):
        hyper_search.run(args)


def test_point_cache_is_invalidated_by_new_data_or_solver(tmp_path):
    import os
    import time

    train, hold = tmp_path / "train.xyzf", tmp_path / "hold.xyzf"
    train.write_text("a\n")
    hold.write_text("b\n")
    task = {"cfg": {"order_2b": 12}, "train_xyzf": str(train), "holdout_xyzf": str(hold), "n_train": 1,
            "algorithm": "lassolars", "alpha": 1e-5, "masses": {"Cu": 63.5}}
    base = _hyper.fit_context(task)
    assert _hyper.fit_context(dict(task)) == base
    assert _hyper.fit_context({**task, "alpha": 1e-3}) != base
    assert _hyper.fit_context({**task, "cfg": {"order_2b": 12, "solver": "nridgecv"}}) != base
    train.write_text("a\nregenerated\n")
    os.utime(train, (time.time() + 5, time.time() + 5))
    assert _hyper.fit_context(task) != base


def test_select_prefer_richer_takes_the_largest_tied_model():
    def pt(n, s):
        return {"status": "done", "holdout_relative_force_error": s, "holdout_relative_force_se": 0.05, "n_params": n,
                "n_equations": 10000, "cfg": {"order_2b": 12, "s_maxim_2b": 6.0, "order_3b": n // 10, "s_maxim_3b": 5.0},
                "holdout_rmse_energy_per_atom": 0.1}
    pts = [pt(40, 0.40), pt(200, 0.39), pt(600, 0.385)]   # all tied within SE 0.05
    cheap = _hyper.select(pts, objective="force", energy_weight=0.0, tolerance=0.03, max_param_ratio=0.5)
    rich = _hyper.select(pts, objective="force", energy_weight=0.0, tolerance=0.03, max_param_ratio=0.5, prefer="richer")
    assert cheap["chosen"]["n_params"] == 40
    assert rich["chosen"]["n_params"] == 600
    assert "richer" in rich["reason"] or rich["chosen"] is rich["best"]
