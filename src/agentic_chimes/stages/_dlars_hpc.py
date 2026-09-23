"""DLARS/DLASSO solve on HPC, with the ill-conditioning "cliff" detected
live against the running job's dlars.log and auto-finalized -- formalizes
the hand-validated workaround stages/_cliff_monitor.py's docstring
describes (see docs/commands/solve.md).

Submits `chimes_lsq.py --algorithm dlars|dlasso` (which itself shells out
to `srun ... dlars ...` -- see codes/chimes_lsq-LLfork/src/chimes_lsq.py:
fit_dlars) as one Slurm job, then polls: while the job is queued/running,
tail dlars.log into a CliffMonitor every `poll_interval_s`. If it decides
"finalize_from_restart", cancel the job and run a short, fresh dlars
invocation capped at `--iterations=<target_iteration>` (LARS' path is
deterministic, so re-running from scratch with a lower cap reproduces the
pre-cliff solution -- no need to parse dlars' own native restart files),
then `chimes_lsq.py --read_output true` to emit params.txt from the
resulting x.txt/Ax.txt. If the job finishes on its own first, no
finalize step runs at all.

Command-string correctness notes (verified against
codes/chimes_lsq-LLfork/src/chimes_lsq.py and contrib/dlars/src/dlars.C
directly, not just prior audit notes, since these submit real jobs):
  - chimes_lsq.py's own --normalize/--split_files flags are `str2bool`
    (accept true/false/yes/no/t/f/y/n, case-insensitive) and need a value
    ("--normalize true"), unlike a bare store_true flag.
  - The `dlars` BINARY's flags are a different convention: --normalize=y|n
    (not true/false), --split_files is a bare flag (no value),
    --algorithm=lars|lasso (not "dlars"/"dlasso" -- that mapping is
    chimes_lsq.py's own algorithm name, translated here).
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Optional

from .. import config, hpc
from ._cliff_monitor import CliffMonitor

_ALGO_TO_DLARS_FLAG = {"dlars": "lars", "dlasso": "lasso"}


def _chimes_lsq_py() -> Path:
    return config.CHIMES_LSQ_ROOT / "src" / "chimes_lsq.py"


def run_dlars_hpc(
    *,
    profile,
    work_dir: Path,
    header: str,
    map_file: str,
    algorithm: str,
    alpha: float,
    normalize: bool = False,
    split_files: bool = False,
    weights: Optional[str] = None,
    nodes: int = 1,
    ntasks_per_node: Optional[int] = None,
    walltime_hours: float = 2.0,
    queue: str = "batch",
    poll_interval_s: int = 60,
    cliff_kwargs: Optional[dict] = None,
    dry_run: bool = False,
) -> dict:
    if algorithm not in _ALGO_TO_DLARS_FLAG:
        raise ValueError(f"algorithm must be one of {list(_ALGO_TO_DLARS_FLAG)}, got {algorithm!r}")

    work_dir = Path(work_dir)
    ntasks_per_node = ntasks_per_node or profile.default_ntasks_per_node
    cores = nodes * ntasks_per_node

    dlars_bin = config.resolve_component("dlars_bin")
    dlars_dir = str(Path(dlars_bin).parent) + "/"
    chimes_lsq_py = _chimes_lsq_py()

    solve_cmd_parts = [
        f"python3 {chimes_lsq_py}",
        "--A A.txt --b b.txt",
        f"--header {header} --map {map_file}",
        f"--algorithm {algorithm} --alpha {alpha}",
        f"--nodes {nodes} --cores {cores} --mpistyle srun",
        f"--dlasso_dlars_path {dlars_dir}",
        f"--normalize {'true' if normalize else 'false'}",
        f"--split_files {'true' if split_files else 'false'}",
    ]
    if weights:
        solve_cmd_parts.append(f"--weights {weights}")
    solve_cmd = " ".join(solve_cmd_parts) + " > params.txt 2> chimes_lsq_stderr.log"

    handle = hpc.submit_job(
        profile,
        job_name="chimes-dlars-solve",
        commands=[solve_cmd],
        work_dir=work_dir,
        nodes=nodes,
        ntasks_per_node=ntasks_per_node,
        walltime_hours=walltime_hours,
        queue=queue,
        dry_run=dry_run,
    )

    if dry_run:
        return {"job_id": None, "dry_run": True, "job_file": str(handle.job_file), "cliff_detected": False, "cliff_report": None, "params": None, "log": None}

    monitor = CliffMonitor(**(cliff_kwargs or {}))
    log_path = work_dir / "dlars.log"
    last_pos = 0
    finalized = False
    last_decision = None

    while hpc.job_in_queue(handle.job_id):
        time.sleep(poll_interval_s)
        if log_path.is_file():
            with open(log_path) as f:
                f.seek(last_pos)
                new_lines = f.readlines()
                last_pos = f.tell()
            monitor.feed_lines(new_lines)
        last_decision = monitor.poll()
        if last_decision.action == "finalize_from_restart":
            hpc.cancel_job(handle.job_id)
            finalized = True
            break

    cliff_report = None
    if finalized:
        cliff_report = _finalize(
            profile=profile,
            work_dir=work_dir,
            dlars_bin=dlars_bin,
            chimes_lsq_py=chimes_lsq_py,
            header=header,
            map_file=map_file,
            algorithm=algorithm,
            alpha=alpha,
            normalize=normalize,
            split_files=split_files,
            weights=weights,
            target_iteration=last_decision.target_iteration or 0,
            nodes=nodes,
            ntasks_per_node=ntasks_per_node,
            queue=queue,
            evidence=last_decision.evidence,
        )

    params_path = work_dir / "params.txt"
    if not params_path.is_file() or "ENDFILE" not in params_path.read_text():
        raise RuntimeError(
            f"dlars solve did not produce a valid ENDFILE-terminated params.txt at {params_path} "
            f"(job {handle.job_id}, finalized={finalized}); check {log_path} and "
            f"{work_dir / 'chimes_lsq_stderr.log'}"
        )

    return {
        "job_id": handle.job_id,
        "cliff_detected": finalized,
        "cliff_report": cliff_report,
        "params": str(params_path),
        "log": str(log_path),
    }


def _finalize(
    *,
    profile,
    work_dir: Path,
    dlars_bin: str,
    chimes_lsq_py: Path,
    header: str,
    map_file: str,
    algorithm: str,
    alpha: float,
    normalize: bool,
    split_files: bool,
    weights: Optional[str],
    target_iteration: int,
    nodes: int,
    ntasks_per_node: int,
    queue: str,
    evidence: list,
) -> dict:
    lars_flag = _ALGO_TO_DLARS_FLAG[algorithm]
    cores = nodes * ntasks_per_node

    dlars_cmd_parts = [
        f"srun -n {cores} {dlars_bin} A.txt b.txt dim.txt",
        f"--lambda={alpha}",
        f"--algorithm={lars_flag}",
        f"--normalize={'y' if normalize else 'n'}",
        f"--iterations={target_iteration}",
    ]
    if split_files:
        dlars_cmd_parts.append("--split_files")
    if weights:
        dlars_cmd_parts.append(f"--weights={weights}")
    dlars_cmd = " ".join(dlars_cmd_parts) + " > dlars_finalize.log 2>&1"

    finalize_handle = hpc.submit_job(
        profile,
        job_name="chimes-dlars-finalize",
        commands=[dlars_cmd],
        work_dir=work_dir,
        nodes=nodes,
        ntasks_per_node=ntasks_per_node,
        walltime_hours=0.5,
        queue=queue,
        dry_run=False,
    )
    hpc.poll_job(profile, finalize_handle, verbose=True)

    x_path = work_dir / "x.txt"
    if not x_path.is_file():
        raise RuntimeError(
            f"finalize dlars run (job {finalize_handle.job_id}, target_iteration={target_iteration}) "
            f"did not produce x.txt -- see {work_dir / 'dlars_finalize.log'}"
        )

    readout_cmd = (
        f"python3 {chimes_lsq_py} --A A.txt --b b.txt --header {header} --map {map_file} "
        f"--algorithm {algorithm} --read_output true"
    )
    proc = subprocess.run(["bash", "-c", readout_cmd], cwd=str(work_dir), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"chimes_lsq.py --read_output true failed (exit {proc.returncode}): {proc.stderr[-2000:]}")
    (work_dir / "params.txt").write_text(proc.stdout)

    return {
        "target_iteration": target_iteration,
        "finalize_job_id": finalize_handle.job_id,
        "evidence": evidence,
    }
