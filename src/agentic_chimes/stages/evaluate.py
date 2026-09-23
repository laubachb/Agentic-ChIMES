"""Holdout force/energy RMSE for one or more params.txt models, computed
in-process via chimes_calculator's serial ctypes API
(`serial_interface/api/chimescalc_serial_py.py`, loaded from its vendored
path) -- no subprocess, no Slurm. Validated against
`codes/chimes_calculator-LLfork/serial_interface/tests/` fixtures (see
tests/unit/test_evaluate.py).

Passing more than one --params opens N concurrent chimes instances (the
serial interface explicitly supports this) and evaluates all of them on the
same frames; `committee_spread` is the plug-in point a future
uncertainty-based active-learning selector would use, deliberately not
implemented here -- see docs/concepts/qm_driver_plugins.md.
"""

from __future__ import annotations

import importlib.util
import math

from .. import config
from ..io import xyzf as xyzf_io

NAME = "evaluate"
SUMMARY = "Holdout force/energy RMSE against one or more params.txt models via the ctypes evaluator."
SCHEMA = {
    "type": "object",
    "required": ["params", "holdout_xyzf"],
    "properties": {
        "params": {"type": "array", "items": {"type": "string"}, "description": ">1 entry = committee mode"},
        "holdout_xyzf": {"type": "string"},
        "max_frames": {"type": ["integer", "null"]},
    },
}


def add_arguments(parser) -> None:
    parser.add_argument("--params", action="append", default=None, help="Path to a params.txt (repeatable for committee mode).")
    parser.add_argument("--holdout-xyzf", dest="holdout_xyzf", default=None)
    parser.add_argument("--max-frames", dest="max_frames", type=int, default=None)


def _cell_vectors(frame):
    if frame.non_ortho:
        return frame.box[0], frame.box[1], frame.box[2]
    lx, ly, lz = frame.box
    return [lx, 0.0, 0.0], [0.0, ly, 0.0], [0.0, 0.0, lz]


def _load_wrapper():
    lib_path = config.resolve_component("chimescalc_lib")
    api_path = config.CHIMES_CALCULATOR_ROOT / "serial_interface" / "api" / "chimescalc_serial_py.py"
    spec = importlib.util.spec_from_file_location("agentic_chimes_chimescalc_serial_py", api_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.chimes_wrapper = mod.init_chimes_wrapper(str(lib_path))
    return mod


def run(args) -> dict:
    params_paths = args.params
    if not params_paths:
        raise ValueError("evaluate requires at least one --params (or 'params' in --json-in)")
    if not args.holdout_xyzf:
        raise ValueError("evaluate requires --holdout-xyzf (or 'holdout_xyzf' in --json-in)")

    frames = xyzf_io.read_xyzf(args.holdout_xyzf)
    if getattr(args, "max_frames", None):
        frames = frames[: args.max_frames]

    wrapper = _load_wrapper()
    instances = []
    for rank, params_path in enumerate(params_paths):
        ptr = wrapper.chimes_open_instance()
        wrapper.set_chimes_instance(ptr, small=False)
        wrapper.init_chimes_instance(ptr, params_path, rank)
        instances.append(ptr)

    n_models = len(instances)
    force_sqerr = [0.0] * n_models
    force_n = [0] * n_models
    energy_sqerr = [0.0] * n_models
    energy_n = [0] * n_models
    per_frame_energy = [[] for _ in range(n_models)]

    try:
        for frame in frames:
            cell_a, cell_b, cell_c = _cell_vectors(frame)
            xcrd = [p[0] for p in frame.positions]
            ycrd = [p[1] for p in frame.positions]
            zcrd = [p[2] for p in frame.positions]
            for mi, ptr in enumerate(instances):
                fx0 = [0.0] * frame.natoms
                fy0 = [0.0] * frame.natoms
                fz0 = [0.0] * frame.natoms
                stress0 = [0.0] * 9
                fx, fy, fz, _stress, energy = wrapper.calculate_chimes_instance(
                    ptr,
                    frame.natoms,
                    xcrd,
                    ycrd,
                    zcrd,
                    frame.symbols,
                    cell_a,
                    cell_b,
                    cell_c,
                    0.0,
                    fx0,
                    fy0,
                    fz0,
                    stress0,
                )
                for (pfx, pfy, pfz), (rfx, rfy, rfz) in zip(zip(fx, fy, fz), frame.forces):
                    force_sqerr[mi] += (pfx - rfx) ** 2 + (pfy - rfy) ** 2 + (pfz - rfz) ** 2
                    force_n[mi] += 3
                per_frame_energy[mi].append(energy)
                if frame.energy is not None:
                    energy_sqerr[mi] += (energy - frame.energy) ** 2
                    energy_n[mi] += 1
    finally:
        for ptr in instances:
            wrapper.chimes_close_instance(ptr)

    results = []
    for mi, params_path in enumerate(params_paths):
        results.append(
            {
                "params": params_path,
                "rmse_force_kcal_mol_ang": math.sqrt(force_sqerr[mi] / force_n[mi]) if force_n[mi] else None,
                "rmse_energy_kcal_mol": math.sqrt(energy_sqerr[mi] / energy_n[mi]) if energy_n[mi] else None,
            }
        )

    committee_spread = None
    if n_models > 1:
        per_frame_spread = []
        for frame_idx in range(len(frames)):
            energies = [per_frame_energy[mi][frame_idx] for mi in range(n_models)]
            mean = sum(energies) / n_models
            per_frame_spread.append(math.sqrt(sum((e - mean) ** 2 for e in energies) / n_models))
        committee_spread = {
            "description": "stdev across models of per-frame predicted energy; plug-in point for uncertainty-based AL selection (not yet used by al-select)",
            "per_frame_energy_stdev": per_frame_spread,
        }

    return {"n_frames": len(frames), "n_models": n_models, "results": results, "committee_spread": committee_spread}
