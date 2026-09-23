"""Sbatch-script preview rendering, usable without any HPC allocation.

This mirrors (but does not literally call) the script-writing half of
`codes/al_driver-LLfork/src/helpers.py:create_and_launch_job` -- that
function has no dry-run mode (it always ends in `sbatch <file>`), so a real
submission always goes through it (see hpc/slurm.py), while `--dry-run`
previews go through this renderer instead. Keep the flag set here (`-J -N
--ntasks-per-node -t -p -A`) in sync with create_and_launch_job if that
function's flags ever change.

The one behavior this module exists to guarantee, independent of that
upstream function: every rendered script carries an explicit
`--ntasks-per-node` (or `--exclusive`), so a Dane job can never silently
fall back to the 1-CPU/~2.3GB default of a bare `-N 1`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RenderedJob:
    script: str
    job_name: str
    nodes: int
    ntasks_per_node: int
    queue: str
    account: str


def render_sbatch_script(
    profile,
    *,
    job_name: str,
    commands: list,
    nodes: int,
    ntasks_per_node: int,
    walltime_hours: float,
    queue: str,
    job_mem_gb: int = 128,
    email: bool = False,
) -> RenderedJob:
    if not ntasks_per_node:
        raise ValueError(
            f"ntasks_per_node must be set explicitly (profile default is "
            f"{profile.default_ntasks_per_node}) -- refusing to render a "
            f"bare '-N {nodes}' script that would silently under-allocate "
            f"on machines like Dane."
        )

    partition = profile.queue_for(queue)

    lines = ["#!/bin/bash"]
    sbatch_flags = [
        f"-J {job_name}",
        f"-N {nodes}",
        f"--ntasks-per-node {ntasks_per_node}",
    ]
    if job_mem_gb and profile.job_system == "UM-ARC":
        sbatch_flags.append(f"--mem-per-cpu={int(job_mem_gb / ntasks_per_node)}G")
    sbatch_flags.append(f"-t {walltime_hours}")
    sbatch_flags.append(f"-p {partition}")
    if email:
        sbatch_flags.append("--mail-type=ALL")
    sbatch_flags.append(f"-A {profile.account}")
    sbatch_flags.append("-V")
    sbatch_flags.append("-o stdoutmsg")

    directive = "#SBATCH" if profile.job_system in ("slurm", "conda-slurm", "TACC", "UM-ARC") else "#PBS"
    for flag in sbatch_flags:
        lines.append(f"{directive} {flag}")

    if profile.modules:
        lines.append("module load " + " ".join(profile.modules))
    if profile.conda_env:
        lines.append(f"conda activate {profile.conda_env}")

    lines.extend(commands)

    return RenderedJob(
        script="\n".join(lines) + "\n",
        job_name=job_name,
        nodes=nodes,
        ntasks_per_node=ntasks_per_node,
        queue=partition,
        account=profile.account,
    )
