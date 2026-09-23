"""stages/_dlars_hpc.py's poll-loop control flow tested with the HPC
boundary (hpc.submit_job/job_in_queue/cancel_job/poll_job) and
config.resolve_component mocked out -- this validates the actual
submit -> poll -> {finish normally | detect cliff -> cancel -> finalize}
logic without needing a real Slurm allocation or a real cliff to occur.
stages/_cliff_monitor.py's own detection logic is separately unit-tested
(tests/unit/test_cliff_monitor.py) against synthetic log fixtures; this
file tests that _dlars_hpc.py *wires it up* correctly to a (simulated)
live job.
"""

from dataclasses import dataclass

import pytest

import agentic_chimes.hpc as hpc
from agentic_chimes import config
from agentic_chimes.stages import _dlars_hpc


@dataclass
class FakeProfile:
    name: str = "fake"
    launcher: str = "srun"
    default_ntasks_per_node: int = 4
    account: str = "acct"


def _handle(job_id, job_name, work_dir):
    return hpc.JobHandle(job_id=job_id, job_name=job_name, work_dir=work_dir, job_file=work_dir / "run.cmd", dry_run=False, machine="fake", queue="debug")


@pytest.fixture(autouse=True)
def _stub_dlars_bin(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "resolve_component", lambda name, required=True: str(tmp_path / "dlars"))


def test_normal_completion_no_cliff(tmp_path, monkeypatch):
    (tmp_path / "params.txt").write_text("...\nENDFILE\n")
    (tmp_path / "dlars.log").write_text("Finished iteration 100\nStopping: no more iterations possible\n")

    monkeypatch.setattr(hpc, "submit_job", lambda profile, **kw: _handle("111", kw["job_name"], tmp_path))
    monkeypatch.setattr(hpc, "job_in_queue", lambda job_id: False)  # already finished by the first check
    monkeypatch.setattr(hpc, "cancel_job", lambda job_id: pytest.fail("must not cancel a normally-completing job"))
    monkeypatch.setattr(_dlars_hpc.time, "sleep", lambda s: None)

    result = _dlars_hpc.run_dlars_hpc(
        profile=FakeProfile(), work_dir=tmp_path, header="header.txt", map_file="map.txt",
        algorithm="dlars", alpha=1e-5, poll_interval_s=1,
    )

    assert result["cliff_detected"] is False
    assert result["cliff_report"] is None
    assert result["params"] == str(tmp_path / "params.txt")
    assert result["job_id"] == "111"


def test_missing_params_after_normal_completion_raises(tmp_path, monkeypatch):
    # job "finished" but never actually wrote a valid params.txt -- must
    # surface a clear error, not silently return a bad/missing path
    monkeypatch.setattr(hpc, "submit_job", lambda profile, **kw: _handle("111", kw["job_name"], tmp_path))
    monkeypatch.setattr(hpc, "job_in_queue", lambda job_id: False)
    monkeypatch.setattr(_dlars_hpc.time, "sleep", lambda s: None)

    with pytest.raises(RuntimeError, match="did not produce a valid ENDFILE"):
        _dlars_hpc.run_dlars_hpc(
            profile=FakeProfile(), work_dir=tmp_path, header="header.txt", map_file="map.txt",
            algorithm="dlars", alpha=1e-5, poll_interval_s=1,
        )


def test_cliff_detected_triggers_cancel_and_finalize(tmp_path, monkeypatch):
    log_path = tmp_path / "dlars.log"
    log_path.write_text("")

    poll_count = {"n": 0}

    def fake_job_in_queue(job_id):
        poll_count["n"] += 1
        n = poll_count["n"]
        if n == 1:
            with open(log_path, "a") as f:
                f.write("Finished iteration 200\n")
            return True
        elif n in (2, 3, 4):
            with open(log_path, "a") as f:
                f.write("Non-incremental Cholesky failed\nFinished iteration 201\n")
            return True
        raise AssertionError("job_in_queue polled too many times -- cliff should have been caught by poll 4")

    submitted = {"main": 0, "finalize": 0}

    def fake_submit_job(profile, **kw):
        if kw["job_name"] == "chimes-dlars-finalize":
            submitted["finalize"] += 1
            (tmp_path / "x.txt").write_text("0.1\n0.2\n")
            return _handle("1000", kw["job_name"], tmp_path)
        submitted["main"] += 1
        return _handle("999", kw["job_name"], tmp_path)

    cancelled = {"job_id": None}

    def fake_cancel_job(job_id):
        cancelled["job_id"] = job_id

    def fake_subprocess_run(cmd, cwd, capture_output, text):
        class _Result:
            returncode = 0
            stdout = "...\nENDFILE\n"
            stderr = ""

        return _Result()

    monkeypatch.setattr(hpc, "submit_job", fake_submit_job)
    monkeypatch.setattr(hpc, "job_in_queue", fake_job_in_queue)
    monkeypatch.setattr(hpc, "cancel_job", fake_cancel_job)
    monkeypatch.setattr(hpc, "poll_job", lambda profile, handle, verbose=True: None)
    monkeypatch.setattr(_dlars_hpc.time, "sleep", lambda s: None)
    monkeypatch.setattr(_dlars_hpc.subprocess, "run", fake_subprocess_run)

    result = _dlars_hpc.run_dlars_hpc(
        profile=FakeProfile(), work_dir=tmp_path, header="header.txt", map_file="map.txt",
        algorithm="dlars", alpha=1e-5, poll_interval_s=1,
        cliff_kwargs={"min_iter_advance": 10, "max_stall_polls": 3, "finalize_margin": 3},
    )

    assert cancelled["job_id"] == "999"
    assert submitted["finalize"] == 1
    assert result["cliff_detected"] is True
    assert result["cliff_report"]["target_iteration"] == 200 - 3  # first failure at iter 200, minus the margin
    assert result["params"] == str(tmp_path / "params.txt")
    assert (tmp_path / "params.txt").read_text() == "...\nENDFILE\n"


def test_dlasso_algorithm_maps_to_lasso_flag(tmp_path, monkeypatch):
    (tmp_path / "params.txt").write_text("ENDFILE\n")
    (tmp_path / "dlars.log").write_text("Stopping: converged\n")

    captured = {}

    def fake_submit_job(profile, *, commands, **kw):
        captured["cmd"] = commands[0]
        return _handle("1", kw["job_name"], tmp_path)

    monkeypatch.setattr(hpc, "submit_job", fake_submit_job)
    monkeypatch.setattr(hpc, "job_in_queue", lambda job_id: False)
    monkeypatch.setattr(_dlars_hpc.time, "sleep", lambda s: None)

    _dlars_hpc.run_dlars_hpc(
        profile=FakeProfile(), work_dir=tmp_path, header="header.txt", map_file="map.txt",
        algorithm="dlasso", alpha=1e-5, poll_interval_s=1,
    )
    assert "--algorithm dlasso" in captured["cmd"]


def test_unknown_algorithm_rejected(tmp_path):
    with pytest.raises(ValueError, match="algorithm must be one of"):
        _dlars_hpc.run_dlars_hpc(
            profile=FakeProfile(), work_dir=tmp_path, header="h", map_file="m", algorithm="svd", alpha=1e-5
        )
