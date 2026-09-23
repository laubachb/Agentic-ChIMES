"""Build A.txt/b.txt/dim.txt by subprocessing the `chimes_lsq` binary
against an fm_setup.in whose TRJFILE already points at a labeled .xyzf
(chimes_lsq's own CLI is exactly `chimes_lsq <fm_setup.in>`, confirmed
against codes/chimes_lsq-LLfork/src/chimes_lsq.C's `argc != 2` check;
outputs land in the process's cwd).

Runs locally by default. Pass `machine` for large/SPLITFI-true runs (the
binary is MPI-capable): submits `<launcher> [-n <cores>] chimes_lsq
<fm_setup.in>` via hpc.submit_job and blocks until it completes
(hpc.poll_job) -- no cliff-style monitoring needed here, generation
either succeeds or fails cleanly, unlike DLARS's solve path.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from .. import config, hpc, machines

NAME = "amat-build"
SUMMARY = "Build A.txt/b.txt/dim.txt from fm_setup.in via the chimes_lsq binary."
SCHEMA = {
    "type": "object",
    "required": ["fm_setup_in"],
    "properties": {
        "fm_setup_in": {"type": "string"},
        "chimes_lsq_bin": {"type": ["string", "null"], "description": "Override the resolved chimes_lsq binary path."},
        "machine": {"type": ["string", "null"], "description": "If given, submits via Slurm instead of running locally."},
        "queue": {"type": "string", "default": "debug"},
        "walltime_hours": {"type": "number", "default": 1.0},
        "nodes": {"type": "integer", "default": 1},
        "ntasks_per_node": {"type": ["integer", "null"]},
    },
}

_OUTPUT_FILES = {
    "A": "A.txt",
    "b": "b.txt",
    "b_labeled": "b-labeled.txt",
    "dim": "dim.txt",
    "natoms": "natoms.txt",
    "params_header": "params.header",
    "ff_groups_map": "ff_groups.map",
}


def add_arguments(parser) -> None:
    parser.add_argument("--fm-setup-in", dest="fm_setup_in", default=None)
    parser.add_argument("--chimes-lsq-bin", dest="chimes_lsq_bin", default=None)
    parser.add_argument("--machine", default=None)
    parser.add_argument("--queue", default="debug")
    parser.add_argument("--walltime-hours", dest="walltime_hours", type=float, default=1.0)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--ntasks-per-node", dest="ntasks_per_node", type=int, default=None)


def _collect_outputs(work_dir: Path, log_path: Path) -> dict:
    outputs = {}
    missing = []
    for key, filename in _OUTPUT_FILES.items():
        p = work_dir / filename
        if p.is_file():
            outputs[key] = str(p)
        else:
            missing.append(filename)

    split = (work_dir / "A.0000.txt").is_file()
    if missing and not split:
        raise RuntimeError(f"chimes_lsq exited 0 but expected output(s) missing: {missing}; see {log_path}")

    return {"log": str(log_path), "split": split, "missing_outputs": missing, **outputs}


def run(args) -> dict:
    if not args.fm_setup_in:
        raise ValueError("amat-build requires --fm-setup-in (or 'fm_setup_in' in --json-in)")

    fm_setup_in = Path(args.fm_setup_in).resolve()
    if not fm_setup_in.is_file():
        raise FileNotFoundError(f"fm_setup.in not found: {fm_setup_in}")

    chimes_lsq_bin = (
        Path(args.chimes_lsq_bin) if getattr(args, "chimes_lsq_bin", None) else config.resolve_component("chimes_lsq_bin")
    )

    work_dir = Path(getattr(args, "output_dir", None) or fm_setup_in.parent)
    work_dir.mkdir(parents=True, exist_ok=True)
    log_path = work_dir / "fm_setup.log"

    machine = getattr(args, "machine", None)
    if not machine:
        proc = subprocess.run([str(chimes_lsq_bin), str(fm_setup_in)], cwd=str(work_dir), capture_output=True, text=True)
        log_path.write_text((proc.stdout or "") + (proc.stderr or ""))
        if proc.returncode != 0:
            raise RuntimeError(f"chimes_lsq exited {proc.returncode}; see {log_path}")
    else:
        profile = machines.load_profile(machine)
        ntasks_per_node = getattr(args, "ntasks_per_node", None) or profile.default_ntasks_per_node
        cores = getattr(args, "nodes", 1) * ntasks_per_node
        launch = f"srun -n {cores}" if profile.launcher == "srun" else profile.launcher
        cmd = f"{launch} {chimes_lsq_bin} {fm_setup_in} > {log_path.name} 2>&1"

        handle = hpc.submit_job(
            profile,
            job_name="chimes-amat-build",
            commands=[cmd],
            work_dir=work_dir,
            nodes=getattr(args, "nodes", 1),
            ntasks_per_node=ntasks_per_node,
            walltime_hours=getattr(args, "walltime_hours", 1.0),
            queue=getattr(args, "queue", "debug"),
            dry_run=bool(getattr(args, "dry_run", False)),
        )
        if handle.dry_run:
            return {"work_dir": str(work_dir), "dry_run": True, "job_file": str(handle.job_file)}

        hpc.poll_job(profile, handle, verbose=True)
        if not log_path.is_file():
            raise RuntimeError(f"job {handle.job_id} finished but {log_path} was never written -- check Slurm output")

    return {"work_dir": str(work_dir), **_collect_outputs(work_dir, log_path)}
