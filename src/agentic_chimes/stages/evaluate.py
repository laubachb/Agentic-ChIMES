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
from pathlib import Path

from .. import config
from ..converters import units
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
        "per_frame": {"type": "boolean", "default": False, "description": "Also return per-frame force squared-error sums (for bootstrap uncertainty)."},
    },
}


def add_arguments(parser) -> None:
    parser.add_argument("--params", action="append", default=None, help="Path to a params.txt (repeatable for committee mode).")
    parser.add_argument("--holdout-xyzf", dest="holdout_xyzf", default=None)
    parser.add_argument("--max-frames", dest="max_frames", type=int, default=None)
    parser.add_argument("--per-frame", dest="per_frame", action="store_true", default=False)


def max_outer_cutoff(params_path) -> float:
    """Largest pair S_MAXIM in a params.txt (3-/4-body SPECIAL cutoffs are
    never larger than the 2-body ones)."""
    lines = Path(params_path).read_text().splitlines()
    start = next(i for i, ln in enumerate(lines) if "PAIRIDX" in ln)
    cut = 0.0
    for ln in lines[start + 1:]:
        toks = ln.split()
        if len(toks) < 5 or not toks[0].isdigit():
            break
        cut = max(cut, float(toks[4]))
    return cut


def _supercell(frame, cutoff: float):
    """Replicate a periodic frame until every perpendicular width exceeds
    2*cutoff. chimes_calculator's serial interface is wrong for cells thinner
    than the cutoff (small=False misses images: energies off by ~20 kcal/mol
    on MatPES cells; small=True sums forces over replicas). Replication is
    exact for a periodic frame -- energy scales by the copy count and the
    first block's forces are the frame's forces -- and matched chimes_lsq's
    own training predictions to 5e-6 kcal/mol/A on 122 real frames."""
    import numpy as np

    cell = np.asarray(frame.box if frame.non_ortho else np.diag(frame.box), dtype=float)
    vol = abs(np.linalg.det(cell))
    widths = [vol / np.linalg.norm(np.cross(cell[(k + 1) % 3], cell[(k + 2) % 3])) for k in range(3)]
    reps = [max(1, math.ceil(2.0 * cutoff / w) + 1) if w <= 2.0 * cutoff else 1 for w in widths]
    pos = np.asarray(frame.positions, dtype=float)
    shifts = [i * cell[0] + j * cell[1] + k * cell[2] for i in range(reps[0]) for j in range(reps[1]) for k in range(reps[2])]
    big_pos = np.concatenate([pos + s for s in shifts])
    big_cell = cell * np.asarray(reps)[:, None]
    return big_pos, list(frame.symbols) * len(shifts), big_cell, len(shifts)


def predict(wrapper, ptr, frame, cutoff: float):
    """ChIMES energy (kcal/mol) and forces (kcal/mol/A, one row per atom)
    for one periodic frame, via an exact supercell (see _supercell)."""
    big_pos, big_sym, big_cell, ncopies = _supercell(frame, cutoff)
    n = len(big_sym)
    fx, fy, fz, _stress, energy = wrapper.calculate_chimes_instance(
        ptr, n, big_pos[:, 0].tolist(), big_pos[:, 1].tolist(), big_pos[:, 2].tolist(), big_sym,
        big_cell[0].tolist(), big_cell[1].tolist(), big_cell[2].tolist(), 0.0,
        [0.0] * n, [0.0] * n, [0.0] * n, [0.0] * 9,
    )
    k = frame.natoms
    return energy / ncopies, list(zip(fx[:k], fy[:k], fz[:k]))


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
    energy_pa_sqerr = [0.0] * n_models
    frame_rows = [[] for _ in range(n_models)]
    ref_force_sq = 0.0
    ref_force_n = 0
    energy_sqerr = [0.0] * n_models
    energy_n = [0] * n_models
    per_frame_energy = [[] for _ in range(n_models)]

    cutoffs = [max_outer_cutoff(pp) for pp in params_paths]
    try:
        for frame in frames:
            for mi, ptr in enumerate(instances):
                energy, pred_forces = predict(wrapper, ptr, frame, cutoffs[mi])
                # .xyzf forces are hartree/bohr; chimes_calculator returns kcal/mol/A
                f_err = f_ref = 0.0
                for (pfx, pfy, pfz), ref in zip(pred_forces, frame.forces):
                    rfx, rfy, rfz = (units.hartree_per_bohr_to_kcal_per_mol_ang(c) for c in ref)
                    f_err += (pfx - rfx) ** 2 + (pfy - rfy) ** 2 + (pfz - rfz) ** 2
                    f_ref += rfx**2 + rfy**2 + rfz**2
                force_sqerr[mi] += f_err
                force_n[mi] += 3 * frame.natoms
                frame_rows[mi].append([f_err, f_ref, 3 * frame.natoms])
                if mi == 0:
                    ref_force_sq += f_ref
                    ref_force_n += 3 * frame.natoms
                per_frame_energy[mi].append(energy)
                if frame.energy is not None:
                    energy_sqerr[mi] += (energy - frame.energy) ** 2
                    energy_pa_sqerr[mi] += ((energy - frame.energy) / frame.natoms) ** 2
                    energy_n[mi] += 1
    finally:
        for ptr in instances:
            wrapper.chimes_close_instance(ptr)

    ref_force_rms = math.sqrt(ref_force_sq / ref_force_n) if ref_force_n else None
    results = []
    for mi, params_path in enumerate(params_paths):
        rmse_f = math.sqrt(force_sqerr[mi] / force_n[mi]) if force_n[mi] else None
        results.append(
            {
                "params": params_path,
                "rmse_force_kcal_mol_ang": rmse_f,
                "relative_force_error": rmse_f / ref_force_rms if rmse_f is not None and ref_force_rms else None,
                "rmse_energy_kcal_mol": math.sqrt(energy_sqerr[mi] / energy_n[mi]) if energy_n[mi] else None,
                "rmse_energy_kcal_mol_per_atom": math.sqrt(energy_pa_sqerr[mi] / energy_n[mi]) if energy_n[mi] else None,
            }
        )
        if getattr(args, "per_frame", False):
            results[-1]["per_frame_force"] = frame_rows[mi]  # [sq error sum, reference sq sum, n components]

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

    return {"n_frames": len(frames), "n_models": n_models, "reference_force_rms_kcal_mol_ang": ref_force_rms,
            "results": results, "committee_spread": committee_spread}
