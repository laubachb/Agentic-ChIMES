"""Tests al-run's launch/detach/status/stop mechanics against a fake
`main.py` substitute (a trivial script under a temp AGENTIC_CHIMES_ROOT) --
this deliberately does not exercise al_driver's own real orchestration
logic (that's LLNL/lab code, out of scope to test here), only that this
stage correctly launches, detaches, logs, and can check on/stop whatever
`codes/al_driver-LLfork/src/main.py` turns out to be.
"""

import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic_chimes.stages import al_run


@pytest.fixture
def fake_al_driver_root(tmp_path, monkeypatch):
    """A temp repo layout with codes/al_driver-LLfork/src/main.py replaced
    by a trivial script, and agentic_chimes.config repointed at it."""
    root = tmp_path / "fake_root"
    main_py_dir = root / "codes" / "al_driver-LLfork" / "src"
    main_py_dir.mkdir(parents=True)
    (main_py_dir / "main.py").write_text(
        "import sys, time\n"
        "print('fake al_driver started, cycles=', sys.argv[1:])\n"
        "time.sleep(float(sys.argv[-1]) if sys.argv[-1].replace('.','',1).isdigit() else 0)\n"
        "print('fake al_driver done')\n"
    )

    import agentic_chimes.config as cfg

    monkeypatch.setattr(cfg, "AL_DRIVER_SRC", main_py_dir)
    return root


@pytest.fixture
def study_dir(tmp_path):
    d = tmp_path / "study"
    d.mkdir()
    (d / "config.py").write_text("# not a real al_driver config -- fake main.py doesn't read it\n")
    return d


def test_launch_writes_log_and_returns_alive_pid(fake_al_driver_root, study_dir):
    args = SimpleNamespace(work_dir=str(study_dir), config_py=None, cycles=[0], python_bin=None, status_of=None, stop=None)
    result = al_run.run(args)

    assert result["status"] == "launched"
    assert result["cycles"] == [0]
    assert Path(result["log"]).is_file()
    assert al_run._pid_alive(result["pid"]) or True  # may finish instantly; just must not raise

    # give the detached process a moment to finish and flush its log
    time.sleep(0.3)
    log_text = Path(result["log"]).read_text()
    assert "fake al_driver started" in log_text
    assert "cycles= ['0']" in log_text
    assert "fake al_driver done" in log_text


def test_launch_copies_provided_config_py(fake_al_driver_root, tmp_path, study_dir):
    external_config = tmp_path / "external_config.py"
    external_config.write_text("EXTERNAL = True\n")

    args = SimpleNamespace(work_dir=str(study_dir), config_py=str(external_config), cycles=[0], python_bin=None, status_of=None, stop=None)
    result = al_run.run(args)

    assert Path(result["config_py"]).read_text() == "EXTERNAL = True\n"


def test_missing_config_py_raises_clear_error(fake_al_driver_root, tmp_path):
    empty_study = tmp_path / "empty_study"
    empty_study.mkdir()
    args = SimpleNamespace(work_dir=str(empty_study), config_py=None, cycles=[0], python_bin=None, status_of=None, stop=None)
    with pytest.raises(FileNotFoundError, match="config.py"):
        al_run.run(args)


def test_status_and_stop_a_long_running_process(fake_al_driver_root, study_dir):
    args = SimpleNamespace(work_dir=str(study_dir), config_py=None, cycles=[0, "5"], python_bin=None, status_of=None, stop=None)
    launched = al_run.run(args)
    pid = launched["pid"]

    status = al_run.run(SimpleNamespace(status_of=pid, work_dir=None, config_py=None, cycles=None, python_bin=None, stop=None))
    assert status["alive"] is True

    stopped = al_run.run(SimpleNamespace(stop=pid, work_dir=None, config_py=None, cycles=None, python_bin=None, status_of=None))
    assert stopped["stopped"] is True

    # In real usage each `chimes-agent al-run` invocation is a short-lived
    # process that exits right after launching/checking/stopping, so a
    # terminated child is promptly reaped by init once we exit. Here, this
    # test process stays alive and remains the child's real parent (a new
    # process *group* via start_new_session isn't reparenting), so we must
    # reap it ourselves before its PID stops showing up as "alive" --
    # otherwise it lingers as a zombie, which os.kill(pid, 0) still finds.
    os.waitpid(pid, 0)
    status_after = al_run.run(SimpleNamespace(status_of=pid, work_dir=None, config_py=None, cycles=None, python_bin=None, stop=None))
    assert status_after["alive"] is False


def test_status_of_dead_pid_is_false():
    # a PID essentially guaranteed not to exist
    result = al_run.run(SimpleNamespace(status_of=2**30, work_dir=None, config_py=None, cycles=None, python_bin=None, stop=None))
    assert result["alive"] is False
