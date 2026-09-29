"""sweep's many-body cutoff and exclusion grid keys reach fm_setup.in."""

from types import SimpleNamespace

from agentic_chimes.stages import sweep


def test_exclusion_and_4b_cutoff_grid_reach_fm_setup(tmp_path, monkeypatch):
    seen = []

    def fake_model_build(args):
        seen.append(open(args.fm_setup_in).read())
        return {"params": "p"}

    monkeypatch.setattr(sweep.model_build, "run", fake_model_build)
    monkeypatch.setattr(sweep.evaluate, "run", lambda a: {"results": [{"rmse_force_kcal_mol_ang": 1.0,
                        "relative_force_error": 0.3, "rmse_energy_kcal_mol": 1.0}]})
    base = {"trjfile": "/x.xyzf", "nframes": 1, "elements": ["Cu", "Zr"], "order": {"2": 12, "3": 6, "4": 3},
            "default_s_maxim": 7.0}
    grid = {"exclude_3b": [[], [["Zr", "Zr", "Zr"]]], "special_maxim_4b": [4.5, 5.5]}
    res = sweep.run(SimpleNamespace(base=base, grid=grid, holdout_xyzf="/h.xyzf", output_dir=str(tmp_path)))
    assert res["n_points"] == 4 and res["n_done"] == 4
    assert sum("EXCLUDE 3B INTERACTION: 1" in s and "Zr Zr Zr" in s for s in seen) == 2
    assert sum("SPECIAL 4B S_MAXIM: ALL 5.5" in s for s in seen) == 2
    assert "relative_force_error" in (tmp_path / "sweep_results.csv").read_text().splitlines()[0]
