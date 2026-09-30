"""al-merge: theory check, fixed holdout, cycle bookkeeping, fit-flag checks;
fm-setup-gen --hyper-choice refits the chosen model on the merged data."""

import json
from types import SimpleNamespace

import pytest

from agentic_chimes.io import xyzf as xyzf_io
from agentic_chimes.stages import al_merge, fm_setup_gen


def _frames(n, energy=True, stress=True):
    return [xyzf_io.Frame(symbols=["Cu", "Zr"], positions=[[0, 0, 0], [1.6, 1.6, 1.6]], forces=[[0, 0, 0]] * 2,
                          box=[3.26] * 3, energy=-270.0 if energy else None,
                          stress=[1, 1, 1, 0, 0, 0] if stress else None) for _ in range(n)]


def _prov(tmp, name, method="DFT-PBE", source="colabfit:x"):
    d = tmp / name
    d.mkdir()
    (d / "provenance.json").write_text(json.dumps({"source": source, "label_policy": "source",
                                                  "level_of_theory": {"methods": {method: 1}, "software": {"VASP": 1}}}))
    return d


def _base(tmp):
    d = _prov(tmp, "base")
    xyzf_io.write_xyzf(_frames(4), d / "train.xyzf")
    xyzf_io.write_xyzf(_frames(2), d / "holdout.xyzf")
    manifest = {"elements": ["Cu", "Zr"], "train_xyzf": str(d / "train.xyzf"), "holdout_xyzf": str(d / "holdout.xyzf"),
                "fit_hints": {"fitener": True, "fitstrs": "ALL"},
                "sources": [json.loads((d / "provenance.json").read_text())]}
    (d / "data_manifest.json").write_text(json.dumps(manifest))
    return d / "data_manifest.json"


def test_merge_grows_train_keeps_holdout_and_records_cycles(tmp_path):
    new = _prov(tmp_path, "round1")
    xyzf_io.write_xyzf(_frames(3), new / "labeled.xyzf")
    res = al_merge.run(SimpleNamespace(data_manifest=str(_base(tmp_path)), new_xyzf=[str(new / "labeled.xyzf")], cycle=1,
                                       allow_mixed_theory=False, output_dir=str(tmp_path / "m1")))
    assert res["n_train"] == 7 and res["n_added"] == 3
    assert json.loads((tmp_path / "m1" / "frame_cycles.json").read_text()) == [0] * 4 + [1] * 3
    man = json.loads((tmp_path / "m1" / "data_manifest.json").read_text())
    assert man["holdout_xyzf"].endswith("base/holdout.xyzf")
    # a second round builds on the first
    new2 = _prov(tmp_path, "round2")
    xyzf_io.write_xyzf(_frames(2), new2 / "labeled.xyzf")
    res2 = al_merge.run(SimpleNamespace(data_manifest=str(tmp_path / "m1" / "data_manifest.json"),
                                        new_xyzf=[str(new2 / "labeled.xyzf")], cycle=2, allow_mixed_theory=False,
                                        output_dir=str(tmp_path / "m2")))
    assert json.loads((tmp_path / "m2" / "frame_cycles.json").read_text()) == [0] * 4 + [1] * 3 + [2] * 2
    assert res2["n_train"] == 9


def test_merge_refuses_other_theory_and_missing_stresses(tmp_path):
    base = _base(tmp_path)
    other = _prov(tmp_path, "other", method="DFT-r2SCAN")
    xyzf_io.write_xyzf(_frames(2), other / "labeled.xyzf")
    with pytest.raises(ValueError, match="level"):
        al_merge.run(SimpleNamespace(data_manifest=str(base), new_xyzf=[str(other / "labeled.xyzf")], cycle=1,
                                     allow_mixed_theory=False, output_dir=str(tmp_path / "x")))
    nostress = _prov(tmp_path, "nostress")
    xyzf_io.write_xyzf(_frames(2, stress=False), nostress / "labeled.xyzf")
    with pytest.raises(ValueError, match="stresses"):
        al_merge.run(SimpleNamespace(data_manifest=str(base), new_xyzf=[str(nostress / "labeled.xyzf")], cycle=1,
                                     allow_mixed_theory=False, output_dir=str(tmp_path / "y")))


def test_fm_setup_gen_from_hyper_choice(tmp_path):
    choice = {"elements": ["Cu", "Zr"], "order": {"2": 6, "3": 4}, "masses": {"Cu": 63.546, "Zr": 91.224},
              "pair_cutoffs": {"Cu-Cu": [2.2, 8.0], "Cu-Zr": [2.3, 8.0], "Zr-Zr": [2.3, 8.0]},
              "morse_lambda": {"Cu-Cu": 2.5, "Cu-Zr": 2.8, "Zr-Zr": 3.2}, "special_maxim_3b": 7.0, "nlayers": 4,
              "fitener": True, "fitstrs": "ALL", "algorithm": "lassolars", "alpha": 1e-5, "stress_weight": 3.0}
    (tmp_path / "choice.json").write_text(json.dumps(choice))
    args = SimpleNamespace(hyper_choice=str(tmp_path / "choice.json"), trjfile="train.xyzf", nframes=9, output_dir=str(tmp_path),
                           elements=None, order=None, pair_cutoffs=None, morse_lambda=None, masses=None, charges=None,
                           fitener="false", fitstrs="false", nlayers=1, fcuttyp="CUBIC", special_maxim_3b=None,
                           special_maxim_3b_pairs=None, special_maxim_4b=None, exclude_3b=None, exclude_4b=None)
    fm_setup_gen.run(args)
    text = (tmp_path / "fm_setup.in").read_text()
    assert "ALL" in text.split("# FITSTRS #")[1].splitlines()[1]
    assert "63.546" in text and "SPECIAL 3B S_MAXIM: ALL 7.0" in text
    assert text.split("# NLAYERS #")[1].split()[0] == "4"
