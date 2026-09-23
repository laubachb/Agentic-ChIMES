"""Path resolution for the vendored forks under ``codes/`` and the built/fetched
dependencies under ``deps/``.

This package is meant to be used from an editable install (``pip install -e .``)
inside the Agentic-ChIMES repo, since it shells out to sibling ``codes/`` and
``deps/`` directories rather than bundling them. :data:`REPO_ROOT` is found by
walking up from this file to the directory containing ``pyproject.toml``; set
``AGENTIC_CHIMES_ROOT`` to override (e.g. for tests against a fixture repo).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _find_repo_root() -> Path:
    env = os.environ.get("AGENTIC_CHIMES_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    # NOTE: do not require codes/ to exist here -- on a fresh checkout it
    # doesn't yet (it's gitignored, cloned by `chimes-agent setup`, which
    # itself needs REPO_ROOT/CODES_DIR to know where to clone to). Use a
    # marker that's always present instead: this file's own known location
    # relative to the repo root.
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "agentic_chimes" / "config.py").is_file():
            return parent
    raise RuntimeError(
        "Could not locate the Agentic-ChIMES repo root (no pyproject.toml + "
        "src/agentic_chimes/config.py found above agentic_chimes/config.py). Set AGENTIC_CHIMES_ROOT "
        "to the repo root explicitly."
    )


REPO_ROOT = _find_repo_root()
CODES_DIR = REPO_ROOT / "codes"
DEPS_DIR = REPO_ROOT / "deps"
INSTALLED_JSON = DEPS_DIR / "installed.json"

AL_DRIVER_ROOT = CODES_DIR / "al_driver-LLfork"
AL_DRIVER_SRC = AL_DRIVER_ROOT / "src"
CHIMES_LSQ_ROOT = CODES_DIR / "chimes_lsq-LLfork"
CHIMES_CALCULATOR_ROOT = CODES_DIR / "chimes_calculator-LLfork"


class ComponentNotInstalled(RuntimeError):
    """Raised when a stage needs a binary/library that `chimes-agent setup` has
    not built yet, or that isn't recorded for the requested machine."""


# component name -> (env var override, human description) — kept in one place so
# `chimes-agent setup --status` and every stage's error message stay consistent.
_COMPONENTS = {
    "chimes_lsq_bin": ("AGENTIC_CHIMES_LSQ_BIN", "chimes_lsq C++ binary (fm_setup.in -> A.txt/b.txt)"),
    "dlars_bin": ("AGENTIC_CHIMES_DLARS_BIN", "dlars DLARS/LASSO solver binary"),
    "chimescalc_lib": ("AGENTIC_CHIMES_CALC_LIB", "libchimescalc_dl.so serial/ctypes evaluator"),
    "lammps_bin": ("AGENTIC_CHIMES_LMP_BIN", "lmp_mpi_chimes (ChIMES-patched LAMMPS)"),
    "qe_pw_bin": ("AGENTIC_CHIMES_QE_BIN", "pw.x (Quantum ESPRESSO)"),
}


def known_components() -> list[str]:
    return sorted(_COMPONENTS)


def _read_installed_json() -> dict:
    if not INSTALLED_JSON.is_file():
        return {}
    with open(INSTALLED_JSON) as f:
        return json.load(f)


def installed_components() -> dict:
    """Full contents of deps/installed.json (written by `chimes-agent setup`),
    or {} if setup has never been run."""
    return _read_installed_json()


def write_installed_component(component: str, *, path: str, machine: str, version: str = "") -> None:
    if component not in _COMPONENTS:
        raise ValueError(f"unknown component {component!r}; known: {known_components()}")
    data = _read_installed_json()
    data[component] = {
        "path": path,
        "machine": machine,
        "version": version,
    }
    DEPS_DIR.mkdir(parents=True, exist_ok=True)
    with open(INSTALLED_JSON, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def resolve_component(component: str, *, required: bool = True) -> Optional[Path]:
    """Resolve a built/fetched dependency's path.

    Resolution order: explicit env var override, then `deps/installed.json`
    (written by `chimes-agent setup`). Raises ComponentNotInstalled with a
    clear "run chimes-agent setup" message if neither is set and `required`.
    """
    if component not in _COMPONENTS:
        raise ValueError(f"unknown component {component!r}; known: {known_components()}")
    env_var, description = _COMPONENTS[component]

    env_val = os.environ.get(env_var)
    if env_val:
        return Path(env_val)

    entry = _read_installed_json().get(component)
    if entry and entry.get("path"):
        return Path(entry["path"])

    if required:
        raise ComponentNotInstalled(
            f"'{component}' ({description}) is not installed. Run "
            f"`chimes-agent setup --component {_setup_component_for(component)}` "
            f"or set {env_var} to an existing binary/library path."
        )
    return None


def _setup_component_for(component: str) -> str:
    """Map a low-level resolved-path component name to the `setup --component`
    name that builds it (several map to the same setup step)."""
    return {
        "chimes_lsq_bin": "chimes_lsq",
        "dlars_bin": "chimes_lsq",
        "chimescalc_lib": "chimes_calculator",
        "lammps_bin": "lammps",
        "qe_pw_bin": "quantum_espresso",
    }[component]


@dataclass
class RepoLayout:
    """Convenience bundle of the fixed vendored-repo paths a stage may need."""

    root: Path = REPO_ROOT
    codes: Path = CODES_DIR
    deps: Path = DEPS_DIR
    al_driver_src: Path = AL_DRIVER_SRC
    chimes_lsq_root: Path = CHIMES_LSQ_ROOT
    chimes_calculator_root: Path = CHIMES_CALCULATOR_ROOT


LAYOUT = RepoLayout()
