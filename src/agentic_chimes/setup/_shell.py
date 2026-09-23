"""Shared subprocess helper for running the vendored forks' own install.sh
scripts (and, for Quantum ESPRESSO, git clone + configure/make) under the
right `hosttype` and modules for a given machine profile.

These upstream install.sh scripts source `modfiles/<hosttype>.mod`, which in
turn does `module load ...` -- that requires an interactive-shell-style
environment (Lmod's `module` shell function), so every command here runs
through `bash -lc` (a login shell) rather than a bare subprocess exec.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


class BuildError(RuntimeError):
    def __init__(self, message: str, *, log_path: Optional[Path] = None):
        super().__init__(message)
        self.log_path = log_path


@dataclass
class BuildResult:
    returncode: int
    log_path: Path


def run_hosttype_script(
    *,
    script: str,
    cwd: Path,
    hosttype: str,
    log_path: Path,
    stdin_text: str = "",
    extra_env: Optional[dict] = None,
    timeout_s: Optional[int] = None,
) -> BuildResult:
    """Run a shell snippet (typically `export hosttype=<X>; ./install.sh ...`)
    from `cwd`, logging combined stdout/stderr to `log_path`. Raises
    BuildError on nonzero exit."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env_prefix = f"export hosttype={hosttype}\n"
    full_script = env_prefix + script

    proc = subprocess.run(
        ["bash", "-lc", full_script],
        cwd=str(cwd),
        input=stdin_text,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    with open(log_path, "w") as f:
        f.write(f"$ bash -lc <<'EOF'\n{full_script}\nEOF\n\n")
        f.write("--- stdout+stderr ---\n")
        f.write(proc.stdout or "")
        f.write(proc.stderr or "")

    if proc.returncode != 0:
        raise BuildError(
            f"command failed (exit {proc.returncode}) in {cwd}; see {log_path}",
            log_path=log_path,
        )
    return BuildResult(returncode=proc.returncode, log_path=log_path)


def find_one(root: Path, filename: str) -> Path:
    """Locate a single build output by filename under `root`, erroring
    clearly if it's missing or ambiguous (used because upstream CMake
    install layouts vary lib/ vs lib64/ by platform)."""
    matches = sorted(root.rglob(filename))
    if not matches:
        raise BuildError(f"expected to find {filename!r} somewhere under {root}, found nothing")
    if len(matches) > 1:
        # Prefer the shallowest match (most likely the real install output,
        # not a leftover from a nested imports/ subbuild).
        matches.sort(key=lambda p: len(p.parts))
    return matches[0]
