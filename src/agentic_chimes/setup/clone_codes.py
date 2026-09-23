"""Clones the three vendored LLNL/LindseyLab-umich forks into `codes/` at
install time. `codes/` is gitignored -- like `deps/`, it is fetched fresh by
`chimes-agent setup`, never committed to this repo (these are the user's
lab forks; this repo should not embed or push their contents/history).

This is the first thing `chimes-agent setup` does, before building
anything, since every build component (`chimes_lsq`, `chimes_calculator`,
`lammps`) depends on the relevant `codes/<fork>` already existing, and
`hpc/slurm.py` imports `codes/al_driver-LLfork/src/helpers.py` at runtime.

Pinned to the exact commit each fork was developed/validated against, not
a moving branch -- reproducibility over "always latest." To move to a
newer commit: bump the `ref` in `REPOS` below (or pass `--ref` for a
one-off), then `chimes-agent setup --component codes --force`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ._shell import run_hosttype_script

REPOS = {
    "al_driver-LLfork": {
        "url": "git@github.com:LindseyLab-umich/al_driver-LLfork.git",
        "ref": "92db0aa6f4a9eaf5bb993aef94a5a7de81a9dada",
    },
    "chimes_lsq-LLfork": {
        "url": "git@github.com:LindseyLab-umich/chimes_lsq-LLfork.git",
        "ref": "75b51f825b5c6cb1d165f4c43aac99bd82cf867d",
    },
    "chimes_calculator-LLfork": {
        "url": "git@github.com:LindseyLab-umich/chimes_calculator-LLfork.git",
        "ref": "f4ddddd5342031cd8725f7e624f024181d42ef24",
    },
}


def ensure_repo(name: str, *, codes_dir: Path, force: bool = False, ref: Optional[str] = None) -> dict:
    """Clone (or fetch+checkout, if --force) one vendored fork into
    codes_dir/<name> at the pinned (or overridden) ref. Idempotent: if the
    directory already exists and force is False, does nothing."""
    if name not in REPOS:
        raise ValueError(f"unknown vendored repo {name!r}; known: {sorted(REPOS)}")

    spec = REPOS[name]
    target_ref = ref or spec["ref"]
    repo_dir = codes_dir / name
    log_dir = codes_dir.parent / "deps" / "logs"

    if repo_dir.is_dir() and not force:
        return {"path": str(repo_dir), "ref": target_ref, "status": "already_present"}

    codes_dir.mkdir(parents=True, exist_ok=True)

    # Plain `git clone` (not --depth 1): the pinned ref is an arbitrary
    # historical commit that a shallow clone may not contain.
    if not repo_dir.is_dir():
        run_hosttype_script(
            script=f"git clone {spec['url']} {name}",
            cwd=codes_dir,
            hosttype="",  # plain git over ssh needs no compiler/MPI modules
            log_path=log_dir / f"clone_{name}.log",
            timeout_s=1800,
        )
        status = "cloned"
    else:
        run_hosttype_script(
            script="git fetch origin",
            cwd=repo_dir,
            hosttype="",
            log_path=log_dir / f"fetch_{name}.log",
            timeout_s=600,
        )
        status = "fetched"

    run_hosttype_script(
        script=f"git checkout {target_ref}",
        cwd=repo_dir,
        hosttype="",
        log_path=log_dir / f"checkout_{name}.log",
        timeout_s=120,
    )

    return {"path": str(repo_dir), "ref": target_ref, "status": status}


def ensure_all(*, codes_dir: Path, force: bool = False) -> dict:
    return {name: ensure_repo(name, codes_dir=codes_dir, force=force) for name in REPOS}
