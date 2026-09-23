"""Thin wrapper around al_driver's existing, working Slurm submission
primitive (`codes/al_driver-LLfork/src/helpers.py:create_and_launch_job` /
`wait_for_job(s)`), imported from its vendored path rather than copied.

Added value over calling that module directly:
  1. `ntasks_per_node` always defaults from the machine profile (closes the
     Dane "-N 1 gives 1 CPU" gotcha permanently, not as an opt-in flag).
  2. Logical queue names ("debug"/"batch") translate to each machine's real
     partition string via `profile.queue_for`, so callers never hardcode
     "pdebug" vs "skx-dev".
  3. `dry_run=True` renders the script (hpc.dry_run) without calling sbatch.
"""

from __future__ import annotations

import contextlib
import os
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

    if dry_run:
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
        return JobHandle(
            job_id=None,
            job_name=job_name,
            work_dir=work_dir,
            job_file=job_file_path,
            dry_run=True,
            machine=profile.name,
            queue=partition,
        )

    helpers = _al_driver_helpers()
    with _chdir(work_dir):
        job_id = helpers.create_and_launch_job(
            commands,
            job_name=job_name,
            job_nodes=str(nodes),
            job_ppn=str(ntasks_per_node),
            job_walltime=str(walltime_hours),
            job_queue=partition,
            job_account=profile.account,
            job_system=profile.job_system,
            job_file=job_file,
            job_email=email,
            job_modules=" ".join(profile.modules),
        )

    return JobHandle(
        job_id=job_id,
        job_name=job_name,
        work_dir=work_dir,
        job_file=job_file_path,
        dry_run=False,
        machine=profile.name,
        queue=partition,
    )


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
