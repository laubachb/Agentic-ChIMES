"""`chimes-agent setup` orchestrator: builds/fetches the binaries and
libraries every later stage depends on, machine-profile aware and
idempotent (skips a component already recorded in deps/installed.json for
the requested machine, unless --force).

Every call first ensures the three vendored forks are cloned into `codes/`
(see clone_codes.py) -- codes/ is gitignored and never pushed to this repo,
so a fresh checkout of Agentic-ChIMES has no `codes/` directory at all until
`chimes-agent setup` populates it. This happens even if the caller only
asked for e.g. `--component quantum_espresso`, since it's a cheap idempotent
check once cloned, and every other component depends on it regardless.
"""

from __future__ import annotations

from typing import Optional

from .. import config, machines
from . import build_chimes_calculator, build_chimes_lsq, build_lammps, build_quantum_espresso, clone_codes

# setup-component name -> (builder function, low-level component keys it produces)
_COMPONENTS = {
    "chimes_lsq": (build_chimes_lsq.build, ["chimes_lsq_bin", "dlars_bin"]),
    "chimes_calculator": (build_chimes_calculator.build, ["chimescalc_lib"]),
    "lammps": (build_lammps.build, ["lammps_bin"]),
    "quantum_espresso": (build_quantum_espresso.build, ["qe_pw_bin"]),
}

ALL_COMPONENTS = list(_COMPONENTS) + ["codes"]


def _is_installed_for(component: str, machine: str) -> bool:
    installed = config.installed_components()
    _, keys = _COMPONENTS[component]
    return all(k in installed and installed[k].get("machine") == machine for k in keys)


def run_setup(
    components: list,
    *,
    machine: str,
    force: bool = False,
    qe_version: Optional[str] = None,
) -> dict:
    if "all" in components:
        components = list(ALL_COMPONENTS)

    unknown = [c for c in components if c not in ALL_COMPONENTS]
    if unknown:
        raise ValueError(f"unknown setup component(s) {unknown}; known: {ALL_COMPONENTS}")

    results = {"codes": clone_codes.ensure_all(codes_dir=config.CODES_DIR, force=force)}

    build_components = [c for c in components if c != "codes"]
    if not build_components:
        return results

    profile = machines.load_profile(machine)

    for name in build_components:
        if not force and _is_installed_for(name, profile.name):
            entry_keys = _COMPONENTS[name][1]
            installed = config.installed_components()
            results[name] = {
                "status": "skipped_already_installed",
                **{k: installed[k] for k in entry_keys},
            }
            continue

        builder, keys = _COMPONENTS[name]
        build_kwargs = {}
        if name == "quantum_espresso" and qe_version:
            build_kwargs["qe_version"] = qe_version

        try:
            out = builder(profile, deps_dir=config.DEPS_DIR, **build_kwargs)
        except Exception as exc:  # noqa: BLE001 - surface any build failure as a structured result
            results[name] = {"status": "failed", "error": str(exc)}
            continue

        version = out.pop("qe_version", "") if name == "quantum_espresso" else ""
        for key in keys:
            if key not in out:
                continue
            config.write_installed_component(key, path=out[key], machine=profile.name, version=version)

        results[name] = {"status": "built", "machine": profile.name, **out}

    return results


def status() -> dict:
    installed = config.installed_components()
    report = {
        "codes": {
            name: {"path": str(config.CODES_DIR / name), "present": (config.CODES_DIR / name).is_dir()}
            for name in clone_codes.REPOS
        }
    }
    for name, (_, keys) in _COMPONENTS.items():
        report[name] = {k: installed.get(k, {"path": None, "machine": None}) for k in keys}
    return report
