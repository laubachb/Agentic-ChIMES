"""job.json records and job-status verdicts (scheduler mocked)."""

import json
import subprocess
from types import SimpleNamespace

from agentic_chimes import machines
from agentic_chimes.hpc import slurm
from agentic_chimes.stages import job_status


def _fake_sched(monkeypatch, squeue="", sacct=""):
    def run(cmd, **k):
        out = {"squeue": squeue, "sacct": sacct, "sbatch": "Submitted batch job 777\n"}[cmd[0]]
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(slurm.subprocess, "run", run)


def _submit(tmp_path, monkeypatch):
    _fake_sched(monkeypatch)
    slurm.submit_job(machines.load_profile("dane"), job_name="t", commands=["echo"], work_dir=tmp_path,
                     expect=["result.json"])
    return json.loads((tmp_path / "job.json").read_text())


def test_submission_records_job_and_expected_results(tmp_path, monkeypatch):
    rec = _submit(tmp_path, monkeypatch)
    assert rec["job_id"] == "777" and rec["expect"] == [str(tmp_path / "result.json")]


def _status(tmp_path):
    return job_status.run(SimpleNamespace(work_dir=str(tmp_path), job_id=None, expect=None))


def test_verdicts(tmp_path, monkeypatch):
    _submit(tmp_path, monkeypatch)
    _fake_sched(monkeypatch, squeue="PENDING|0:00|1:00:00|(Priority)\n")
    assert _status(tmp_path)["verdict"] == "QUEUED"

    _fake_sched(monkeypatch, sacct="COMPLETED|0:0|00:05:00|01:00:00\n")
    v = _status(tmp_path)
    assert v["verdict"] == "COMPLETED_WITHOUT_RESULTS" and "missing" in v["reason"]

    (tmp_path / "result.json").write_text('{"trunc')
    assert "unreadable" in _status(tmp_path)["reason"]

    (tmp_path / "result.json").write_text('{"ok": 1}')
    assert _status(tmp_path)["verdict"] == "SUCCEEDED"

    _fake_sched(monkeypatch, sacct="TIMEOUT|0:0|01:00:00|01:00:00\n")
    v = _status(tmp_path)
    assert v["verdict"] == "FAILED" and v["reason"] == "TIMEOUT" and "walltime" in v["fix"]
