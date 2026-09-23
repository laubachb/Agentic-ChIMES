"""Fetch + build Quantum ESPRESSO's `pw.x` (single-point / relax labeling).

Unlike the other three components, QE is not vendored anywhere in `codes/`
(confirmed absent by a full-repo grep during the design audit) -- this is a
new, from-scratch fetch, mirroring the *pattern* the vendored forks already
use for their own external dependencies (chimes_lsq's clone-all.sh clones
upstream chimes_calculator; chimes_calculator's etc/lmp/install.sh
shallow-clones a pinned LAMMPS tag): a pinned release tag, shallow-cloned,
built in place under `deps/`.

Only `pw.x` is built (`make pw`) -- the minimum needed for single-point/relax
QM labeling. Other QE executables (ph.x, etc.) can be added here later if a
stage needs them, without changing the fetch step.

DEFAULT_QE_VERSION is a pinned tag, not "latest" -- check
https://gitlab.com/QEF/q-e/-/tags for the current stable release before
relying on this default; override with --qe-version.
"""

from __future__ import annotations

from pathlib import Path

from .. import config
from ._shell import run_hosttype_script

QE_REPO_URL = "https://gitlab.com/QEF/q-e.git"
DEFAULT_QE_VERSION = "qe-7.3.1"


def build(profile, *, deps_dir: Path = config.DEPS_DIR, qe_version: str = DEFAULT_QE_VERSION) -> dict:
    qe_dir = deps_dir / "quantum-espresso"
    log_dir = deps_dir / "logs"
    extra_configure_args = " ".join(profile.qe.get("configure_extra_args", []))

    if not qe_dir.is_dir():
        deps_dir.mkdir(parents=True, exist_ok=True)
        run_hosttype_script(
            script=f"git clone --depth 1 --branch {qe_version} {QE_REPO_URL} {qe_dir.name}",
            cwd=deps_dir,
            hosttype=profile.hosttype,
            log_path=log_dir / "qe_clone.log",
            timeout_s=1800,
        )
    else:
        _log(log_dir / "qe_clone.log", f"{qe_dir} already exists, skipping clone (pass --force to re-fetch).")

    run_hosttype_script(
        script=f"./configure {extra_configure_args} && make pw",
        cwd=qe_dir,
        hosttype=profile.hosttype,
        log_path=log_dir / "qe_build.log",
        timeout_s=10800,
    )

    pw_bin = qe_dir / "bin" / "pw.x"
    if not pw_bin.is_file():
        raise RuntimeError(
            f"QE configure/make reported success but expected output {pw_bin} "
            f"is missing -- see {log_dir / 'qe_build.log'}"
        )

    return {"qe_pw_bin": str(pw_bin), "qe_version": qe_version}


def _log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(message + "\n")
