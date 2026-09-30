"""The one sbatch-script renderer, used for both `--dry-run` previews and
real submissions (hpc/slurm.py writes this exact text and sbatches it), so
what the user approves is what runs. Directives follow al_driver's
`create_and_launch_job` (`-J -N --ntasks-per-node -t -p -A -V -o`).

The one behavior this module exists to guarantee, independent of that
upstream function: every rendered script carries an explicit
`--ntasks-per-node` (or `--exclusive`), so a Dane job can never silently
fall back to the 1-CPU/~2.3GB default of a bare `-N 1`.
"""

from __future__ import annotations

import shlex
import sys
from dataclasses import dataclass
from pathlib import Path


def hours_to_slurm_time(hours: float) -> str:
    """Slurm's -t/--time wants HH:MM:SS (or a similar qualified format) --
    a bare decimal like "1.0" is NOT valid Slurm time syntax (a bare
    number is parsed as *minutes*, and Slurm rejects the decimal point
    regardless), so every walltime_hours value must go through this
    before reaching sbatch."""
    if hours <= 0:
        raise ValueError(f"walltime_hours must be positive, got {hours}")
    total_seconds = round(hours * 3600)
    h, remainder = divmod(total_seconds, 3600)
    m, s = divmod(remainder, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


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
    sbatch_flags.append(f"-t {hours_to_slurm_time(walltime_hours)}")
    sbatch_flags.append(f"-p {partition}")
    if email:
        sbatch_flags.append("--mail-type=ALL")
    sbatch_flags.append(f"-A {profile.account}")
    sbatch_flags.append("-V")
    sbatch_flags.append("-o stdoutmsg")

    directive = "#SBATCH" if profile.job_system in ("slurm", "conda-slurm", "TACC", "UM-ARC") else "#PBS"
    for flag in sbatch_flags:
        lines.append(f"{directive} {flag}")

    if not profile.account:
        raise ValueError(
            f"machine profile {profile.name!r} has no account: set CHIMES_ACCOUNT (or edit the profile's `account`)"
        )

    if profile.modules:
        lines.append("module load " + " ".join(profile.modules))
    if profile.conda_env:
        # Guarded: `conda activate` fails in a batch shell where conda was never initialized.
        env = shlex.quote(profile.conda_env)
        lines.append(f'if command -v conda >/dev/null 2>&1; then eval "$(conda shell.bash hook)" && conda activate {env}; fi')
    # Run the interpreter that submitted the job (same packages as the login-node stages).
    lines.append(f"export PATH={shlex.quote(str(Path(sys.executable).parent))}:$PATH")

    lines.extend(commands)

    return RenderedJob(
        script="\n".join(lines) + "\n",
        job_name=job_name,
        nodes=nodes,
        ntasks_per_node=ntasks_per_node,
        queue=partition,
        account=profile.account,
    )
