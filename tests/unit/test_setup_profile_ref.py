"""setup --ref (one-off fork commits), setup --init-profile, and the study dashboard."""

import json
from types import SimpleNamespace

import pytest

from agentic_chimes import machines
from agentic_chimes.setup import clone_codes, init_profile
from agentic_chimes.stages import study


def test_parse_refs_accepts_short_names_and_rejects_junk():
    assert clone_codes.parse_refs(["chimes_lsq=abc123", "al_driver-LLfork=main"]) == {
        "chimes_lsq-LLfork": "abc123", "al_driver-LLfork": "main"}
    assert clone_codes.parse_refs(None) == {}
    for bad in ("chimes_lsq", "nope=abc", "chimes_lsq="):
        with pytest.raises(ValueError, match="REPO=COMMIT"):
            clone_codes.parse_refs([bad])


def test_ref_override_checks_out_only_that_repo(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(clone_codes, "run_hosttype_script", lambda **kw: calls.append((kw["cwd"].name, kw["script"])))
    codes = tmp_path / "codes"
    for name in clone_codes.REPOS:
        (codes / name).mkdir(parents=True)
    res = clone_codes.ensure_all(codes_dir=codes, refs={"chimes_lsq-LLfork": "deadbeef"})
    assert calls == [("chimes_lsq-LLfork", "git fetch origin"), ("chimes_lsq-LLfork", "git checkout deadbeef")]
    assert res["chimes_lsq-LLfork"]["ref"] == "deadbeef" and res["chimes_lsq-LLfork"]["override"]
    assert res["chimes_lsq-LLfork"]["pinned_ref"] == clone_codes.REPOS["chimes_lsq-LLfork"]["ref"]
    assert res["al_driver-LLfork"]["status"] == "already_present"


def test_init_profile_writes_a_loadable_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(init_profile, "detect", lambda: {"slurm": True, "accounts": ["bankA"], "partitions": [
        {"name": "pdebug", "default": False, "cores_per_node": 112, "time_limit": "1:00:00"},
        {"name": "pbatch", "default": True, "cores_per_node": 112, "time_limit": "1-00:00:00"}]})
    monkeypatch.delenv("CHIMES_ACCOUNT", raising=False)
    path = tmp_path / "mycluster.yaml"
    r = init_profile.write(path, scratch="/shared/me", modules=["gcc/12", "openmpi/4"])
    assert r["partitions"] == {"debug": "pdebug", "batch": "pbatch"} and r["default_ntasks_per_node"] == 112
    assert r["account"] == "bankA" and r["todo"] == []
    monkeypatch.setenv("CHIMES_ACCOUNT", "other")                # the environment still wins, so the file is shareable
    assert machines.load_profile(str(path)).account == "other"
    with pytest.raises(ValueError, match="exists"):
        init_profile.write(path)
    # nothing detectable: the file is still valid, and says what to fill in
    monkeypatch.setattr(init_profile, "detect", lambda: {"slurm": False, "accounts": [], "partitions": []})
    r = init_profile.write(tmp_path / "bare.yaml")
    assert len(r["todo"]) == 5 and "# TODO" in (tmp_path / "bare.yaml").read_text()


def test_study_dashboard_phases_jobs_and_waiting(tmp_path, monkeypatch):
    root = tmp_path / "s"
    study.run(SimpleNamespace(init=str(root), study=None, name="demo", goal="g", elements=["Cu"], register=None, extra_roots=None))
    cur = root / "01_data" / "curate"
    cur.mkdir(parents=True)
    (cur / "data_manifest.json").write_text(json.dumps({"n_train": 10, "n_holdout": 3, "warnings": []}))
    done = root / "02_fit" / "jobA"
    done.mkdir(parents=True)
    (done / "result.json").write_text("{}")
    (done / "job.json").write_text(json.dumps({"job_id": "11", "job_name": "fit", "submitted_at": 2.0, "expect": [str(done / "result.json")]}))
    lost = root / "02_fit" / "jobB"
    lost.mkdir()
    (lost / "job.json").write_text(json.dumps({"job_id": "12", "job_name": "fit", "submitted_at": 1.0, "expect": [str(lost / "result.json")]}))
    prepared = root / "04_md" / "check"
    prepared.mkdir(parents=True)
    (prepared / "run.cmd").write_text("#!/bin/bash\n")
    ran = root / "04_md" / "old"                                   # ran before job records existed: not "prepared"
    ran.mkdir()
    (ran / "run.cmd").write_text("#!/bin/bash\n")
    (ran / "stdoutmsg").write_text("")
    (root / "usage" / "local.jsonl").write_text(json.dumps({"cpu_s": 7200.0}) + "\n")
    import subprocess

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="", returncode=0))
    r = study.run(SimpleNamespace(init=None, study=str(root), name=None, goal=None, elements=None, register=None,
                                  extra_roots=None, status=True))
    d = r["dashboard"]
    states = {p["phase"]: p["state"] for p in d["phases"]}
    assert states["Data"] == "done" and states["Hyperparameters"] == "not started" and states["Report"] == "not started"
    assert d["next"].startswith("Hyperparameters")
    assert {j["job_id"]: j["state"] for j in d["jobs_recent"]} == {"11": "FINISHED", "12": "NO_RESULTS"}
    assert len(d["waiting_on_user"]) == 2 and any(str(prepared) in w for w in d["waiting_on_user"])
    assert not any(str(ran) in w for w in d["waiting_on_user"])
    assert d["cpu_hours"]["login_node_cpu_hours"] == 2.0
    assert "10 train / 3 holdout" in r["dashboard_text"] and (root / "STATUS.md").is_file()
    # without --status the plain registry view is unchanged
    plain = study.run(SimpleNamespace(init=None, study=str(root), name=None, goal=None, elements=None, register=None, extra_roots=None))
    assert "dashboard" not in plain and "missing" in plain
