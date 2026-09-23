"""Build A.txt/b.txt/dim.txt by subprocessing the `chimes_lsq` binary
against an fm_setup.in whose TRJFILE already points at a labeled .xyzf
(chimes_lsq's own CLI is exactly `chimes_lsq <fm_setup.in>`, confirmed
against codes/chimes_lsq-LLfork/src/chimes_lsq.C's `argc != 2` check;
outputs land in the process's cwd).

Local execution only in this phase -- --hpc / SPLITFI support for large
DLARS-bound runs lands alongside solve's HPC path (Phase 2).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from .. import config

NAME = "amat-build"
SUMMARY = "Build A.txt/b.txt/dim.txt from fm_setup.in via the chimes_lsq binary (local only for now)."
SCHEMA = {
    "type": "object",
    "required": ["fm_setup_in"],
    "properties": {
        "fm_setup_in": {"type": "string"},
        "chimes_lsq_bin": {"type": ["string", "null"], "description": "Override the resolved chimes_lsq binary path."},
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


def run(args) -> dict:
    if not args.fm_setup_in:
        raise ValueError("amat-build requires --fm-setup-in (or 'fm_setup_in' in --json-in)")

    fm_setup_in = Path(args.fm_setup_in).resolve()
    if not fm_setup_in.is_file():
        raise FileNotFoundError(f"fm_setup.in not found: {fm_setup_in}")

    chimes_lsq_bin = (
        Path(args.chimes_lsq_bin) if getattr(args, "chimes_lsq_bin", None) else config.resolve_component("chimes_lsq_bin")
    )

    work_dir = Path(args.output_dir) if getattr(args, "output_dir", None) else fm_setup_in.parent
    work_dir.mkdir(parents=True, exist_ok=True)

    log_path = work_dir / "fm_setup.log"
    proc = subprocess.run(
        [str(chimes_lsq_bin), str(fm_setup_in)], cwd=str(work_dir), capture_output=True, text=True
    )
    log_path.write_text((proc.stdout or "") + (proc.stderr or ""))

    if proc.returncode != 0:
        raise RuntimeError(f"chimes_lsq exited {proc.returncode}; see {log_path}")

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

    return {"work_dir": str(work_dir), "log": str(log_path), "split": split, "missing_outputs": missing, **outputs}
