"""Launch al_driver's own active-learning loop (`main.py`) -- this stage
does not reimplement active-learning orchestration, it correctly *calls*
the existing, working driver.

Scope, deliberately minimal ("just calling the driver to start"): you
prepare an al_driver study the normal way al_driver itself documents --
`ALL_BASE_FILES/` (ALC-0 training data, MD templates, QM templates) plus a
`config.py` (see codes/al_driver-LLfork/examples/*/config.py for real
worked examples, and codes/al_driver-LLfork/doc/source/options.rst for the
full config surface -- e.g. `examples/simple_iter_single_statepoint-lmp-test/`
uses `MD_STYLE="LMP"` + `BULK_QM_METHOD="LMP"` with the already-built
ChIMES-patched LAMMPS binary standing in as both the MD engine and the
"QM" reference method, which is the fastest way to smoke-test the driver
end to end without a real DFT code). This stage does not generate that
config.py for you (al_driver's config surface is large and deeply
system-specific -- see docs/commands/al-run.md for why templating it
generically is out of scope here) -- it copies a given config.py into
place and launches `main.py <cycles>` correctly.

`main.py` is itself a long-lived process (it submits and polls its own
Slurm jobs internally, potentially over hours to days -- al_driver's own
docs recommend running it under screen/tmux/nohup). This stage launches it
as a detached background process (a new session, like nohup) and returns
immediately with its PID and log path -- it does not block waiting for
completion. Use --status-of/--stop (mirroring `submit`) to check on or
stop a previously launched run.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

from .. import config

NAME = "al-run"
SUMMARY = "Launch al_driver's main.py (active-learning loop) as a detached background process."
USES_OUTPUT_DIR = False
SCHEMA = {
    "type": "object",
    "properties": {
        "work_dir": {"type": "string", "description": "An al_driver study directory (ALL_BASE_FILES/ already prepared)."},
        "config_py": {"type": ["string", "null"], "description": "Path to a config.py to copy into work_dir; omit if work_dir/config.py already exists."},
        "cycles": {"type": "array", "items": {"type": "integer"}, "default": [0], "description": "ALC cycle indices to run, passed as main.py's argv."},
        "python_bin": {"type": ["string", "null"], "description": "Override which python runs main.py; default sys.executable."},
        "status_of": {"type": ["integer", "null"], "description": "PID to check instead of launching."},
        "stop": {"type": ["integer", "null"], "description": "PID to SIGTERM instead of launching."},
    },
}


def add_arguments(parser) -> None:
    parser.add_argument("--work-dir", dest="work_dir", default=None)
    parser.add_argument("--config-py", dest="config_py", default=None)
    parser.add_argument("--cycles", type=lambda s: [int(x) for x in s.split(",")], default=None)
    parser.add_argument("--python-bin", dest="python_bin", default=None)
    parser.add_argument("--status-of", dest="status_of", type=int, default=None)
    parser.add_argument("--stop", dest="stop", type=int, default=None)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def run(args) -> dict:
    if getattr(args, "status_of", None):
        pid = args.status_of
        return {"pid": pid, "alive": _pid_alive(pid)}

    if getattr(args, "stop", None):
        pid = args.stop
        if not _pid_alive(pid):
            return {"pid": pid, "stopped": False, "note": "process was not running"}
        os.kill(pid, signal.SIGTERM)
        return {"pid": pid, "stopped": True}

    if not getattr(args, "work_dir", None):
        raise ValueError("al-run requires --work-dir (an al_driver study directory; see this stage's module docstring)")

    work_dir = Path(args.work_dir).resolve()
    if not work_dir.is_dir():
        raise FileNotFoundError(f"work_dir not found: {work_dir}")

    config_py_dest = work_dir / "config.py"
    if getattr(args, "config_py", None):
        shutil.copy(Path(args.config_py).resolve(), config_py_dest)
    if not config_py_dest.is_file():
        raise FileNotFoundError(
            f"{config_py_dest} does not exist and no --config-py was given. al-run expects an "
            "already-prepared al_driver study (ALL_BASE_FILES/ + config.py) -- see "
            "codes/al_driver-LLfork/examples/*/config.py for real worked examples, and this "
            "stage's module docstring / docs/commands/al-run.md for scope."
        )

    main_py = config.AL_DRIVER_SRC / "main.py"
    if not main_py.is_file():
        raise FileNotFoundError(f"al_driver main.py not found at {main_py} -- run `chimes-agent setup --component codes` first")

    cycles = getattr(args, "cycles", None) or [0]
    python_bin = getattr(args, "python_bin", None) or sys.executable

    log_path = work_dir / "driver.log"
    cmd = [python_bin, str(main_py)] + [str(c) for c in cycles]

    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            cmd,
            cwd=str(work_dir),
            stdout=logf,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # detach like nohup -- survives this CLI call exiting
        )

    return {
        "work_dir": str(work_dir),
        "config_py": str(config_py_dest),
        "cycles": cycles,
        "pid": proc.pid,
        "log": str(log_path),
        "status": "launched",
        "command": " ".join(cmd),
        "note": (
            "main.py is now running detached in the background (it submits and polls its own "
            "Slurm jobs internally, potentially for hours/days). This call does not wait for it. "
            "Check progress with `tail -f " + str(log_path) + "`, `chimes-agent al-run --status-of "
            + str(proc.pid) + "`, or stop it with `chimes-agent al-run --stop " + str(proc.pid) + "`."
        ),
    }
