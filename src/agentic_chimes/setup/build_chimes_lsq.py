"""Build codes/chimes_lsq-LLfork: the `chimes_lsq` binary + the `dlars`
solver, via that fork's own (unmodified) `install.sh`.

Notes on the upstream script, load-bearing for this wrapper:
  - `install.sh` unconditionally runs `./uninstall.sh` first (`rm -rf build`),
    so every call is a full rebuild -- there is no incremental-build mode to
    preserve, which is why `chimes-agent setup` treats this whole component
    as build-once-then-skip (via deps/installed.json) rather than trying to
    be clever about partial rebuilds.
  - It interactively asks "Imports directory will be deleted and
    re-cloned/installed. Proceed? (y/n)" before touching `imports/`. Feeding
    "n" is always safe and idempotent: it keeps any existing `imports/`
    (which `clone-all.sh` populates if missing anyway) and never deletes data.
  - `clone-all.sh` clones the *upstream* rk-lindsey/chimes_calculator repo
    into `imports/chimes_calculator` and builds it there -- this is a second,
    independent build of chimes_calculator, separate from
    `codes/chimes_calculator-LLfork` (which `build_chimes_calculator.py`
    builds for the ctypes/evaluate stage). That duplication is how the
    upstream chimes_lsq build already works; this wrapper does not change it.
  - Outputs: `build/chimes_lsq` and `contrib/dlars/src/dlars`.
"""

from __future__ import annotations

from pathlib import Path

from .. import config
from ._shell import run_hosttype_script


def build(profile, *, deps_dir: Path = config.DEPS_DIR) -> dict:
    root = config.CHIMES_LSQ_ROOT
    log_dir = deps_dir / "logs"

    run_hosttype_script(
        script="./install.sh",
        cwd=root,
        hosttype=profile.hosttype,
        log_path=log_dir / "chimes_lsq_install.log",
        stdin_text="n\n",
        timeout_s=3600,
    )

    chimes_lsq_bin = root / "build" / "chimes_lsq"
    dlars_bin = root / "contrib" / "dlars" / "src" / "dlars"

    for p in (chimes_lsq_bin, dlars_bin):
        if not p.is_file():
            raise RuntimeError(
                f"chimes_lsq install.sh reported success but expected output "
                f"{p} is missing -- see {log_dir / 'chimes_lsq_install.log'}"
            )

    return {
        "chimes_lsq_bin": str(chimes_lsq_bin),
        "dlars_bin": str(dlars_bin),
    }
