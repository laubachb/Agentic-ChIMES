"""Build codes/chimes_calculator-LLfork's serial/ctypes evaluator library,
via that fork's own (unmodified) `install.sh`.

`install.sh` (non-interactive, unlike chimes_lsq's) defaults its install
prefix to its own `build/` directory and runs `cmake .. && make && make
install` there, producing the CMake target `ChimesCalc_dynamic`
(OUTPUT_NAME `chimescalc_dl`) as `libchimescalc_dl.so`. The exact
lib/ vs lib64/ subdirectory depends on the platform's CMake
GNUInstallDirs default, so this wrapper locates the .so by search rather
than assuming one fixed path.
"""

from __future__ import annotations

from pathlib import Path

from .. import config
from ._shell import find_one, run_hosttype_script


def build(profile, *, deps_dir: Path = config.DEPS_DIR) -> dict:
    root = config.CHIMES_CALCULATOR_ROOT
    log_dir = deps_dir / "logs"

    run_hosttype_script(
        script="./install.sh",
        cwd=root,
        hosttype=profile.hosttype,
        log_path=log_dir / "chimes_calculator_install.log",
        timeout_s=1800,
    )

    lib_name = "libchimescalc_dl.so"
    lib_path = find_one(root / "build", lib_name)

    return {"chimescalc_lib": str(lib_path)}
