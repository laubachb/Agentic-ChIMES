"""Slurm submission for every stage.

One renderer (`hpc.dry_run.render_sbatch_script`) writes the job script in
both modes, so a `--dry-run` preview is byte-for-byte the script a real
submission sends to `sbatch`. (Real submissions used to go through
al_driver's `helpers.create_and_launch_job`, which writes its own, different
script: the preview showed a `conda activate` the real job never ran.)
al_driver's helpers are still used to *wait* for jobs.

Added value over calling sbatch directly:
  1. `ntasks_per_node` always defaults from the machine profile (closes the
     Dane "-N 1 gives 1 CPU" gotcha permanently, not as an opt-in flag).
  2. Logical queue names ("debug"/"batch") translate to each machine's real
     partition string via `profile.queue_for`.
  3. Jobs run the Python interpreter that submitted them (its bin dir is put
     first on PATH), so login-node and compute-node environments match.
  4. Job directories on node-local storage (/tmp, ...) are refused: compute
     nodes cannot see them, and the job would "complete" with no output.
"""

from __future__ import annotations

import contextlib
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .. import config
from . import dry_run as _dry_run


@contextlib.contextmanager
def _chdir(path: Path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _al_driver_helpers():
    """Import al_driver's helpers.py from its vendored location, on demand
    (not a hard package dependency -- only needed once a stage actually
    submits to HPC)."""
    src = str(config.AL_DRIVER_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)
    import helpers  # type: ignore

    return helpers


@dataclass
class JobHandle:
    job_id: Optional[str]
    job_name: str
    work_dir: Path
    job_file: Path
    dry_run: bool
    machine: str
    queue: str


def submit_job(
    profile,
    *,
    job_name: str,
    commands: list,
    work_dir: Path,
    nodes: int = 1,
    ntasks_per_node: Optional[int] = None,
    walltime_hours: float = 1.0,
    queue: str = "debug",
    job_file: str = "run.cmd",
    email: bool = False,
    dry_run: bool = False,
) -> JobHandle:
    """Submit (or, with dry_run=True, just render) a Slurm job on `profile`.

    `commands` is the list of shell command lines that make up the job body
    (e.g. the `srun ... chimes_lsq ...` invocation) -- this function only
    handles the sbatch envelope around them.
    """
    ntasks_per_node = ntasks_per_node or profile.default_ntasks_per_node
    if profile.require_ntasks_per_node_or_exclusive and not ntasks_per_node:
        raise ValueError(
            f"machine profile {profile.name!r} requires an explicit "
            "ntasks_per_node (or --exclusive) and none was given or defaulted"
        )

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    job_file_path = work_dir / job_file
    partition = profile.queue_for(queue)

    check_shared_dir(work_dir)
    rendered = _dry_run.render_sbatch_script(
        profile,
        job_name=job_name,
        commands=commands,
        nodes=nodes,
        ntasks_per_node=ntasks_per_node,
        walltime_hours=walltime_hours,
        queue=queue,
        email=email,
    )
    job_file_path.write_text(rendered.script)
    if dry_run:
        return JobHandle(
            job_id=None,
            job_name=job_name,
            work_dir=work_dir,
            job_file=job_file_path,
            dry_run=True,
            machine=profile.name,
            queue=partition,
        )

    job_id = _launch(profile, job_file_path)
    return JobHandle(
        job_id=job_id,
        job_name=job_name,
        work_dir=work_dir,
        job_file=job_file_path,
        dry_run=False,
        machine=profile.name,
        queue=partition,
    )


_NODE_LOCAL = ("/tmp", "/var/tmp", "/dev/shm")


def check_shared_dir(work_dir) -> None:
    """Refuse job directories on node-local storage. Compute nodes cannot see
    a login node's /tmp, so such a job reports COMPLETED and writes nothing
    the caller can read. Tests set CHIMES_AGENT_ALLOW_LOCAL_JOB_DIRS=1."""
    if os.environ.get("CHIMES_AGENT_ALLOW_LOCAL_JOB_DIRS") == "1":
        return
    path = str(Path(work_dir).resolve())
    local = list(_NODE_LOCAL)
    if os.environ.get("TMPDIR"):
        local.append(str(Path(os.environ["TMPDIR"]).resolve()))
    for root in local:
        if path == root or path.startswith(root.rstrip("/") + "/"):
            raise ValueError(
                f"job directory {path} is on node-local storage ({root}); compute nodes cannot see it and the job "
                "would finish with no output. Use a shared filesystem, e.g. the machine profile's scratch_root."
            )


def _launch(profile, job_file: Path) -> str:
    """sbatch (or qsub) the rendered file from its own directory; return the job id."""
    torque = profile.job_system == "torque"
    cmd = ["qsub" if torque else "sbatch", job_file.name]
    proc = subprocess.run(cmd, cwd=job_file.parent, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed (exit {proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}")
    out = proc.stdout.strip()
    m = re.search(r"Submitted batch job (\d+)", out) if not torque else None
    if m:
        return m.group(1)
    if torque and out:
        return out.split()[0]
    raise RuntimeError(f"could not parse a job id from {cmd[0]} output: {out!r}")


def poll_job(profile, job_handle: JobHandle, *, verbose: bool = True) -> None:
    """Block until `job_handle` leaves the queue (60s poll, matching
    al_driver's existing cadence). No-op for a dry-run handle."""
    if job_handle.dry_run or job_handle.job_id is None:
        return
    helpers = _al_driver_helpers()
    helpers.wait_for_job(
        [job_handle.job_id],
        job_system=profile.job_system,
        job_name=job_handle.job_name,
        verbose=verbose,
    )


def poll_jobs(profile, job_handles: list, *, verbose: bool = True) -> None:
    real = [h.job_id for h in job_handles if not h.dry_run and h.job_id is not None]
    if not real:
        return
    helpers = _al_driver_helpers()
    helpers.wait_for_jobs(real, job_system=profile.job_system, verbose=verbose)


def job_in_queue(job_id: str) -> bool:
    """A single, non-blocking squeue check (unlike poll_job, which blocks
    in a loop until the job is gone) -- for callers that need to do
    something else (e.g. tail a log and react) between checks."""
    proc = subprocess.run(["squeue", "-j", job_id], capture_output=True, text=True)
    return job_id in proc.stdout


def cancel_job(job_id: str) -> None:
    subprocess.run(["scancel", job_id], capture_output=True, text=True)
