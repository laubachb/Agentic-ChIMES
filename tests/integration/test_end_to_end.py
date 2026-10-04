"""End-to-end run of the real toolchain on a tiny system, with ASE's EMT
potential standing in for DFT (deterministic, seconds). It exercises what
unit tests cannot: chimes_lsq builds and solves, chimes_calculator
evaluation, LAMMPS MD, stresses carried from ASE's sign convention into a
stress fit, and the fingerprint tool.

Needs the built components (`chimes-agent setup ...`); skips cleanly
otherwise. Run: pytest tests/integration -v
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from agentic_chimes import config

pytestmark = pytest.mark.skipif(
    not all(config.resolve_component(c, required=False) for c in ("chimes_lsq_bin", "chimescalc_lib", "lammps_bin")),
    reason="needs chimes_lsq, chimes_calculator and LAMMPS built (chimes-agent setup)",
)


def _emt_frames(path: Path, n: int = 40, seed: int = 0):
    from ase.build import bulk
    from ase.calculators.emt import EMT
    from ase.io import write

    rng = np.random.default_rng(seed)
    frames = []
    for k in range(n):
        a = 3.61 * (1 + rng.uniform(-0.04, 0.04))
        at = bulk("Cu", "fcc", a=a, cubic=True).repeat((2, 2, 2))
        at.rattle(stdev=0.05 + 0.1 * rng.random(), seed=int(seed * 1000 + k))
        at.calc = EMT()
        at.get_potential_energy()
        at.get_stress()
        frames.append(at)
    write(str(path), frames, format="extxyz")


def test_emt_copper_study(tmp_path):
    from agentic_chimes.stages import data_curate, data_fetch, evaluate, fingerprint, hyper_search, md_check

    src = tmp_path / "emt.extxyz"
    _emt_frames(src)
    fetch = data_fetch.run(SimpleNamespace(source=str(src), elements=["Cu"], composition_rule="subset", method=None,
                                           max_frames=None, count_only=False, max_scan_files=None, label_policy="source",
                                           level_of_theory="EMT (ASE)", seed=42, output_dir=str(tmp_path / "fetch")))
    assert fetch["n_with_stress"] == 40
    assert fetch["stress_sign_check"]["verdict"] == "ok"          # ASE Cauchy sign converted to ChIMES pressure sign

    cur = data_curate.run(SimpleNamespace(frames=[fetch["pool_xyzf"]], elements=["Cu"], min_distance_ang=0.5,
                                          max_force_ev_ang=50.0, energy_outlier_mad=8.0, energy_outlier_floor=1.0,
                                          dedupe=True, natoms_min=None, natoms_max=None, require_orthorhombic=False,
                                          max_volume_ratio=3.0, coverage_cutoff_ang=6.0, target_size=None,
                                          holdout_fraction=0.25, split_by="group", allow_mixed_theory=False, seed=42,
                                          output_dir=str(tmp_path / "curate")))
    manifest = json.loads(Path(cur["data_manifest"]).read_text())
    assert manifest["fit_hints"]["fitstrs"] == "ALL"

    search = hyper_search.run(SimpleNamespace(
        data_manifest=cur["data_manifest"], train_xyzf=None, holdout_xyzf=None, elements=None, hyper_analysis=None,
        stages=["2b", "stress"], four_body="off", orders_2b=[6, 8], orders_3b=None, orders_4b=None,
        s_maxim_2b=[5.0], s_maxim_3b=None, s_maxim_4b=None, lambda_scales=None, objective="auto", energy_weight=0.1,
        tolerance=0.03, min_gain=0.05, min_signal=1e-9, exclude_inert=False, max_param_ratio=0.5, fitener=None,
        fitstrs=None, stress_weights=[1.0, 3.0], algorithm="lassolars", alpha=1e-5, masses=None, workers=1, machine=None,
        max_fit_seconds=300, output_dir=str(tmp_path / "search")))
    params = search["params"]
    assert search["final"]["holdout_relative_force_error"] < 0.5

    ev = evaluate.run(SimpleNamespace(params=[params], holdout_xyzf=manifest["holdout_xyzf"], max_frames=None,
                                      per_frame=False))["results"][0]
    assert ev["n_frames_below_inner_cutoff"] == 0
    assert ev["rmse_pressure_gpa"] is not None

    md = md_check.run(SimpleNamespace(params=[params], structure_xyzf=None, frame_index=0,
                                      prototype={"name": "Cu", "crystalstructure": "fcc", "a": 3.61},
                                      elements=["Cu"], masses={"Cu": 63.546}, temperatures=[300.0], nsteps=200,
                                      timestep=1.0, dump_every=20, min_atoms=100, temperature_tolerance=0.5,
                                      max_pe_jump_per_atom=1.0, close_contact_margin=0.1, reference_xyzf=None,
                                      rdf_rmax=4.0, harvest_close=5, harvest_other=5, workers=1, run_timeout_s=600,
                                      seed=7, machine=None, output_dir=str(tmp_path / "md")))
    assert md["models"][0]["stable_at_all_temperatures"]

    fpr = fingerprint.run(SimpleNamespace(params=params, reference_xyzf=manifest["train_xyzf"],
                                          candidates_xyzf=manifest["holdout_xyzf"], orders=[2], max_frames=50,
                                          max_clusters=2000, alpha=0.1, workers=1, machine=None,
                                          output_dir=str(tmp_path / "fp")))
    assert "novelty" in fpr

    # the rest of the toolchain on the same tiny study: learning curve, committee, EOS, batch, status, deploy, report
    from agentic_chimes.stages import (al_batch, al_status, committee, deploy, eos_check, learning_curve, study,
                                       study_report)

    fm = search["fm_setup_in"]
    lc = learning_curve.run(SimpleNamespace(fm_setup_in=fm, holdout_xyzf=manifest["holdout_xyzf"], fractions=[0.5, 1.0],
                                            repeats=1, algorithm="lassolars", alpha=1e-5, weights_preset=None,
                                            stress_weight=None, seed=0, plot=True, output_dir=str(tmp_path / "lc")))
    assert lc["verdict"] in ("data-limited", "plateau", "undetermined") and len(lc["curve"]) == 2
    cm = committee.run(SimpleNamespace(fm_setup_in=fm, algorithm="lassolars", alpha=1e-5, n_models=3, seed=0,
                                       candidates_xyzf=manifest["holdout_xyzf"], n_select=3, output_dir=str(tmp_path / "cm")))
    assert cm["n_models"] == 3 and Path(cm["uncertain_xyzf"]).is_file()
    eos = eos_check.run(SimpleNamespace(params=params, structure_xyzf=None, frame_index=0,
                                        prototype={"name": "Cu", "crystalstructure": "fcc", "a": 3.61}, volume_range=0.06,
                                        n_points=7, strain=0.005, reference_xyzf=manifest["holdout_xyzf"], plot=True,
                                        output_dir=str(tmp_path / "eos")))
    assert eos["eos"]["B0_GPa"] > 0 and eos["elastic"]["born_stable"]
    batch = al_batch.run(SimpleNamespace(candidates_xyzf=md["harvest_xyzf"] or manifest["holdout_xyzf"],
                                         train_xyzf=manifest["train_xyzf"], params=params, fm_setup_in=None, budget=5,
                                         min_close=2, close_margin=0.1, signals=["quests", "fingerprint"], n_models=3,
                                         algorithm="lassolars", alpha=1e-5, seed=0, output_dir=str(tmp_path / "batch")))
    assert batch["n_selected"] <= 5 and Path(batch["batch_xyzf"]).is_file()

    root = tmp_path / "study"
    study.run(SimpleNamespace(init=str(root), study=None, name="emt-cu", goal="integration test", elements=["Cu"],
                              register=None, extra_roots=None))
    regs = [f"data_manifest={cur['data_manifest']}", f"hyper_report={search['hyper_report']}", f"params={params}",
            f"fm_setup={fm}", f"md_check={tmp_path / 'md'}", f"eos_check={tmp_path / 'eos'}", f"fingerprint={tmp_path / 'fp'}",
            f"committee={tmp_path / 'cm'}", f"learning_curve={tmp_path / 'lc'}"]
    study.run(SimpleNamespace(init=None, study=str(root), name=None, goal=None, elements=None, register=regs, extra_roots=None))
    dep = deploy.run(SimpleNamespace(study=str(root), model_name="emt-cu", output=None, penalty_dist=0.02, penalty_scaling=1e5, reduce=True))
    assert "params.txt" in dep["files"]
    st = al_status.run(SimpleNamespace(study=str(root), al_dir=None, base_manifest=None, base_params=None, plot=False))
    assert st["verdict"] == "NO_ROUNDS"
    rep = study_report.run(SimpleNamespace(study=str(root)))
    text = Path(rep["report"]).read_text()
    assert "Equation of state" in text and "data sufficiency" in text.lower() and "## Figures" in text
