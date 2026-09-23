"""Build the ChIMES-patched LAMMPS binary via
codes/chimes_calculator-LLfork/etc/lmp/install.sh (unmodified).

That script shallow-clones the pinned LAMMPS tag `stable_29Aug2024_update1`
into its own `build/<tag>/` subdir, copies in `chimesFF.{h,cpp}` +
`pair_chimes.{h,cpp}` (adds the ChIMES pair style) and *replaces* LAMMPS'
core `src/pair.{h,cpp}` (ChIMES needed base-class changes for many-body
ev_tally, so this is not a drop-in plugin build against a stock LAMMPS),
then builds `lmp_mpi_chimes` via a custom `Makefile.mpi_chimes`
(`Makefile.mpi_chimes.UT-TACC` on Stampede3). Output lands at
`etc/lmp/exe/lmp_mpi_chimes`; this wrapper symlinks it into `deps/` for a
uniform lookup location alongside Quantum ESPRESSO.
"""

from __future__ import annotations

from pathlib import Path

from .. import config
from ._shell import run_hosttype_script


def build(profile, *, deps_dir: Path = config.DEPS_DIR) -> dict:
    lmp_dir = config.CHIMES_CALCULATOR_ROOT / "etc" / "lmp"
    log_dir = deps_dir / "logs"

    run_hosttype_script(
        script="./install.sh",
        cwd=lmp_dir,
        hosttype=profile.hosttype,
        log_path=log_dir / "lammps_install.log",
        # LAMMPS-from-source is the slowest build in this repo; give it room.
        timeout_s=7200,
    )

    lmp_bin = lmp_dir / "exe" / "lmp_mpi_chimes"
    if not lmp_bin.is_file():
        raise RuntimeError(
            f"LAMMPS install.sh reported success but expected output {lmp_bin} "
            f"is missing -- see {log_dir / 'lammps_install.log'}"
        )

    deps_dir.mkdir(parents=True, exist_ok=True)
    link = deps_dir / "lammps-chimes"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(lmp_dir / "exe")

    return {"lammps_bin": str(lmp_bin)}
