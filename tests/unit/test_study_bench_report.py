"""study registry, usage accounting (faked sacct), benchmark parsing/scaling
math, deploy's model card and study-report on a synthetic study."""

import json
from types import SimpleNamespace

import pytest

from agentic_chimes.stages import benchmark, deploy, study, study_report, usage


def _init(tmp_path, **kw):
    return study.run(SimpleNamespace(init=str(tmp_path / "s"), study=None, name=kw.get("name"), goal="test goal",
                                     elements=["Cu", "Zr"], register=None, extra_roots=None))


def _reg(root, *entries):
    return study.run(SimpleNamespace(init=None, study=str(root), name=None, goal=None, elements=None,
                                     register=list(entries), extra_roots=None))


def test_study_init_layout_and_registry(tmp_path):
    st = _init(tmp_path)
    root = tmp_path / "s"
    assert (root / "study.json").is_file() and (root / "02_fit").is_dir() and (root / "STUDY.md").is_file()
    assert "params" in st["missing"]
    (root / "p.txt").write_text("x")
    st = _reg(root / "02_fit", f"params={root / 'p.txt'}")  # any path inside the study works
    assert st["artifacts"]["params"] == str(root / "p.txt")
    with pytest.raises(ValueError, match="unknown artifact"):
        _reg(root, f"nope={root / 'p.txt'}")
    assert study.find_study(root / "01_data") == root


def test_usage_by_phase_with_fake_sacct(tmp_path, monkeypatch):
    _init(tmp_path)
    root = tmp_path / "s"
    rows = [
        {"JobID": "1", "JobName": "qe-relabel", "WorkDir": str(root / "01_data/qe"), "AllocCPUS": "112", "NNodes": "1",
         "ElapsedRaw": "3600", "CPUTimeRAW": "403200", "TotalCPU": "1-00:00:00", "State": "COMPLETED", "Partition": "pbatch", "Submit": "a"},
        {"JobID": "2", "JobName": "hyper-search", "WorkDir": str(root / "02_fit/search"), "AllocCPUS": "30", "NNodes": "1",
         "ElapsedRaw": "600", "CPUTimeRAW": "18000", "TotalCPU": "30:00.000", "State": "COMPLETED", "Partition": "pdebug", "Submit": "b"},
        {"JobID": "3", "JobName": "other", "WorkDir": "/elsewhere", "AllocCPUS": "1", "NNodes": "1",
         "ElapsedRaw": "10", "CPUTimeRAW": "10", "TotalCPU": "00:10", "State": "COMPLETED", "Partition": "pdebug", "Submit": "c"},
    ]
    monkeypatch.setattr(usage, "_sacct", lambda since: rows)
    (root / "usage").mkdir(exist_ok=True)
    (root / "usage/local.jsonl").write_text(json.dumps({"stage": "data-curate", "output_dir": str(root / "01_data/curate"),
                                                        "cpu_s": 7200, "wall_s": 7000}) + "\n")
    res = usage.run(SimpleNamespace(study=str(root), since="2026-01-01", phase_hints=None))
    assert res["n_jobs"] == 2
    assert res["by_phase"]["data"]["cpu_hours"] == 112.0 and res["by_phase"]["data"]["local_cpu_hours"] == 2.0
    assert res["by_phase"]["fit"]["used_cpu_hours"] == 0.5
    assert res["total_used_cpu_hours"] == 24.5 and res["allocation_efficiency"] == pytest.approx(24.5 / 117.0, abs=1e-3)


def test_slurm_duration_parsing():
    assert usage.parse_slurm_duration("20:01.212") == pytest.approx(1201.212)
    assert usage.parse_slurm_duration("1-02:00:00") == 93600
    assert usage.parse_slurm_duration("") == 0.0


LOG = """LAMMPS
Loop time of 0.6 on {p} procs for 10 steps with {n} atoms
Step Temp PotEng TotEng Press
Loop time of {t} on {p} procs for 100 steps with {n} atoms
"""


def test_benchmark_scaling_math_and_estimates(tmp_path):
    cases = []
    # strong: 8000 atoms, near-ideal to 16 ranks then efficiency drops
    for p, t in [(1, 100.0), (2, 50.5), (4, 25.6), (16, 7.0), (112, 2.0)]:
        cases.append({"mode": "strong", "ranks": p, "atoms": 8000, "loop_time_s": t, "steps": 100})
    for p, t in [(1, 2.9), (16, 3.0), (112, 3.4)]:
        cases.append({"mode": "weak", "ranks": p, "atoms": 250 * p, "loop_time_s": t, "steps": 100})
    res = benchmark.analyze(cases, 1.0, 0.7)
    s = {r["ranks"]: r for r in res["strong"]["rows"]}
    assert s[2]["efficiency"] == pytest.approx(100 / (2 * 50.5))
    assert res["strong"]["recommended_ranks"] == 16  # 112 ranks: 100/(112*2) = 0.45 < 0.7
    assert res["weak"]["recommended_ranks"] == 112     # 2.9/3.4 = 0.85
    model = benchmark.cost_model(res)
    assert model["atoms_per_rank"] == 250
    assert model["core_s_per_atom_step"] == pytest.approx(112 * 3.4 / (100 * 28000))
    est = benchmark.estimate(model, 100_000, 1.0, 1.0)
    full_node = s_w = {r["ranks"]: r for r in res["weak"]["rows"]}[112]["core_s_per_atom_step"]
    assert est["suggested_nodes"] == 4 and est["suggested_ranks"] == 448  # whole nodes
    assert est["core_s_per_atom_step_used"] == full_node                  # multi-node runs pay full-node cost
    assert est["cpu_hours"] == pytest.approx(full_node * 1e5 * 1e6 / 3600, rel=1e-3)
    small = benchmark.estimate(model, 4_000, 1.0, 1.0)                     # 16 ranks on one node
    assert small["core_s_per_atom_step_used"] == {r["ranks"]: r for r in res["weak"]["rows"]}[16]["core_s_per_atom_step"]
    assert benchmark.parse_log(LOG.format(p=4, n=1000, t=12.5)) == (12.5, 4, 100, 1000)


def test_replicate_to_is_near_target():
    from ase.build import bulk
    atoms, reps = benchmark.replicate_to(bulk("CuZr", "cesiumchloride", a=3.26, cubic=True), 2000)
    assert len(atoms) == 2000 and reps == [10, 10, 10]


def _synthetic_study(tmp_path):
    _init(tmp_path)
    root = tmp_path / "s"
    (root / "01_data/curate").mkdir(parents=True, exist_ok=True)
    (root / "01_data/curate/curation_report.json").write_text(json.dumps({"n_input": 10, "n_removed": 2,
        "removed": [{"reason": "low_density x"}, {"reason": "duplicate_of:a"}]}))
    (root / "01_data/curate/data_manifest.json").write_text(json.dumps({
        "elements": ["Cu", "Zr"], "level_of_theory": ["DFT-PBE (VASP)"], "labeled": True, "n_train": 8,
        "summary": {"n_frames": 8, "compositions": {"Cu-Zr": 8}, "natoms_range": [2, 4], "thinnest_cell_width_ang": 2.0},
        "pairs": {"Cu-Zr": {"n_frames": 8, "min_distance": 2.3, "n_within_1.2x_min": 50}},
        "sources": [{"source": "colabfit:x", "license": "CC-BY-4.0", "doi": "10.1/x"}],
        "curation_report": str(root / "01_data/curate/curation_report.json"), "warnings": ["w1"]}))
    best = root / "02_fit/search/best"
    best.mkdir(parents=True)
    (best / "params.txt").write_text("PARAMS\nENDFILE\n")
    (root / "02_fit/search/hyper_report.json").write_text(json.dumps({
        "final": {"holdout_relative_force_error": 0.31, "holdout_relative_force_se": 0.06, "holdout_rmse_force": 2.0,
                  "holdout_rmse_energy_per_atom": 0.85, "train_relative_force_error": 0.3, "n_params": 52, "nlayers": 4},
        "hyperparameters": {"elements": ["Cu", "Zr"], "order": {"2": 6, "3": 4}, "pair_cutoffs": {"Cu-Zr": [2.26, 8.0]},
                            "morse_lambda": {"Cu-Zr": 2.79}, "special_maxim_3b": 7.0, "exclude_3b": [["Cu", "Cu", "Zr"]]},
        "stages": [{"stage": "2b", "n_points": 30, "reason": "r"},
                   {"stage": "exclude", "n_points": 4, "reason": "excluded 1",
                    "cluster_types": [{"type": "Cu Cu Zr", "instances": 5, "coefficients_saved": 33,
                                       "score_change_when_removed": {"round1": -0.001}, "excluded": True}]}],
        "notes": ["n1"], "n_fits": 34}))
    md = root / "04_md/run300"
    md.mkdir(parents=True)
    (md / "log.lammps").write_text("fix 1 all nve\ntimestep 1.0\nStep Temp PotEng TotEng Press\n0 300 -100 -90 0\n100 290 -99 -90.5 0\n"
                                   "200 305 -98 -90.2 0\nLoop time of 5.0 on 4 procs for 200 steps with 100 atoms\n")
    return root


def test_deploy_and_study_report(tmp_path):
    root = _synthetic_study(tmp_path)
    dep = deploy.run(SimpleNamespace(study=str(root), model_name="cuzr-test", output=None))
    card = (root / "06_deploy/MODEL_CARD.md").read_text()
    assert "Cu-Zr" in card and "0.31" in card and "CC-BY-4.0" in card and "Cu Cu Zr" in card
    assert "no benchmark" in " ".join(dep["gaps"])
    assert (root / "06_deploy/params.txt").read_text().startswith("PARAMS")
    rep = study_report.run(SimpleNamespace(study=str(root)))
    text = (root / "REPORT.md").read_text()
    facts = json.loads((root / "REPORT_FACTS.json").read_text())
    assert rep["narrative_placeholders"] >= 6
    assert facts["md_runs"][0]["simulated_ps"] == 0.2 and facts["md_runs"][0]["unstable"] is False
    assert facts["md_runs"][0]["energy_drift_kcal_mol_atom"] == pytest.approx(-0.002)
    assert facts["data"]["removed_by_reason"] == {"low_density": 1, "duplicate_of": 1}
    assert "| Cu Cu Zr | 5 | 33 | -0.001 | yes |" in text
    assert "no active-learning run" in text
