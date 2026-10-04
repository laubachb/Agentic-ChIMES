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
        "plot": {"type": "boolean", "default": False, "description": "With --output-dir: write parity_forces.png (model vs reference force components, by element) and error_by_composition.png."},
        "per_frame": {"type": "boolean", "default": False, "description": "Also return per-frame force squared-error sums (for bootstrap uncertainty)."},
    },
}


def add_arguments(parser) -> None:
    parser.add_argument("--params", action="append", default=None, help="Path to a params.txt (repeatable for committee mode).")
    parser.add_argument("--holdout-xyzf", dest="holdout_xyzf", default=None)
    parser.add_argument("--max-frames", dest="max_frames", type=int, default=None)
    parser.add_argument("--per-frame", dest="per_frame", action="store_true", default=False)
    parser.add_argument("--plot", action="store_true", default=False)


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


def predict(wrapper, ptr, frame, cutoff: float, *, with_stress: bool = False):
    """ChIMES energy (kcal/mol) and forces (kcal/mol/A, one row per atom)
    for one periodic frame, via an exact supercell (see _supercell). With
    with_stress, also the stress as xyzf's [xx, yy, zz, xy, xz, yz] in GPa
    (pressure sign; chimes_calculator returns -dE/dV in kcal/mol/A^3,
    intensive, so the supercell does not change it)."""
    big_pos, big_sym, big_cell, ncopies = _supercell(frame, cutoff)
    n = len(big_sym)
    fx, fy, fz, stress9, energy = wrapper.calculate_chimes_instance(
        ptr, n, big_pos[:, 0].tolist(), big_pos[:, 1].tolist(), big_pos[:, 2].tolist(), big_sym,
        big_cell[0].tolist(), big_cell[1].tolist(), big_cell[2].tolist(), 0.0,
        [0.0] * n, [0.0] * n, [0.0] * n, [0.0] * 9,
    )
    k = frame.natoms
    if with_stress:
        s = [c * units.CHIMES_STRESS_TO_GPA for c in stress9]
        return energy / ncopies, list(zip(fx[:k], fy[:k], fz[:k])), [s[0], s[4], s[8], s[1], s[2], s[5]]
    return energy / ncopies, list(zip(fx[:k], fy[:k], fz[:k]))


def _closest(frame, rmax: float = 4.0) -> dict:
    """Closest distance per element pair "A-B" (sorted) in one periodic frame."""
    from ase import Atoms
    from ase.neighborlist import neighbor_list
    import numpy as np

    cell = np.asarray(frame.box if frame.non_ortho else np.diag(frame.box), dtype=float)
    at = Atoms(symbols=frame.symbols, positions=frame.positions, cell=cell, pbc=True)
    i, j, d = neighbor_list("ijd", at, rmax)
    sym = np.array(frame.symbols)
    out = {}
    for a, b, r in zip(sym[i], sym[j], d):
        key = "-".join(sorted((a, b)))
        out[key] = min(out.get(key, 1e9), float(r))
    return out


def _cell_vectors(frame):
    if frame.non_ortho:
        return frame.box[0], frame.box[1], frame.box[2]
    lx, ly, lz = frame.box
    return [lx, 0.0, 0.0], [0.0, ly, 0.0], [0.0, 0.0, lz]


def composition_label(frame) -> str:
    """Composition class of a frame: its element set, e.g. "Cu-Zr"."""
    return "-".join(sorted(set(frame.symbols)))


def group_breakdown(frame_rows, groups, elem_rows) -> dict:
    """Relative force error per composition class and per element, plus the
    worst frames. A pooled error can hide the group that matters: on Cu-Zr the
    alloy frames were 3x worse than pure Cu at a pooled 0.31."""
    import math

    by_comp = {}
    for (err, ref, n), g in zip(frame_rows, groups):
        acc = by_comp.setdefault(g, [0.0, 0.0, 0])
        acc[0] += err
        acc[1] += ref
        acc[2] += 1
    rel = [math.sqrt(e / r) if r else None for e, r, _ in frame_rows]
    worst = sorted((i for i in range(len(rel)) if rel[i] is not None), key=lambda i: -rel[i])[:5]
    return {
        "by_composition": {g: {"relative_force_error": round(math.sqrt(e / r), 4) if r else None, "n_frames": k}
                           for g, (e, r, k) in sorted(by_comp.items())},
        "by_element": {el: {"relative_force_error": round(math.sqrt(e / r), 4) if r else None, "n_atoms": n // 3}
                       for el, (e, r, n) in sorted(elem_rows.items())},
        "worst_frames": [{"frame": i, "composition": groups[i], "relative_force_error": round(rel[i], 4)} for i in worst],
    }


def check_frames_against_models(frames, params_paths) -> None:
    """Refuse frames with elements a model does not describe, before the
    calculator sees them: chimes_calculator calls exit() inside the C library
    on an unknown atom type, which ended the process silently with status 0."""
    from ..io.params import model_types

    for p in params_paths:
        known = {s for s, _ in model_types(p)}
        bad = [(i, sorted(set(f.symbols) - known)) for i, f in enumerate(frames) if set(f.symbols) - known]
        if bad:
            shown = ", ".join(f"frame {i}: {els}" for i, els in bad[:5])
            raise ValueError(f"{len(bad)} frame(s) contain elements the model {p} does not describe ({shown}"
                             f"{', ...' if len(bad) > 5 else ''}); model types: {sorted(known)}")


def _load_wrapper():
    lib_path = config.resolve_component("chimescalc_lib")
    api_path = config.CHIMES_CALCULATOR_ROOT / "serial_interface" / "api" / "chimescalc_serial_py.py"
    spec = importlib.util.spec_from_file_location("agentic_chimes_chimescalc_serial_py", api_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.chimes_wrapper = mod.init_chimes_wrapper(str(lib_path))
    return mod


def evaluate_frames(frames, params_paths, *, per_frame: bool = False, predictions: bool = False) -> dict:
    """Score in-memory frames against one or more models. Everything `run`
    reports, for callers (cross-validation, learning curves, al-status) that
    should not write temporary .xyzf files. With `predictions`, per-atom
    predicted/reference force components and element labels are returned
    too (for parity plots)."""
    check_frames_against_models(frames, params_paths)
    groups = [composition_label(f) for f in frames]
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
    ref_force_abs = 0.0
    ref_force_n = 0
    energy_sqerr = [0.0] * n_models
    energy_n = [0] * n_models
    per_frame_energy = [[] for _ in range(n_models)]
    pred_rows = [[] for _ in range(n_models)]  # with predictions: (predicted, reference, element) per force component
    frame_e_err = [[] for _ in range(n_models)]  # per frame: energy error per atom (None without a reference energy)
    frame_p_err = [[] for _ in range(n_models)]  # per frame: pressure error, GPa (None without a reference stress)
    elem_rows = [dict() for _ in range(n_models)]  # element -> [sq error, ref sq, n components]
    stress_sqerr = [0.0] * n_models
    pressure_sqerr = [0.0] * n_models
    stress_n = [0] * n_models

    cutoffs = [max_outer_cutoff(pp) for pp in params_paths]
    below = [0] * n_models  # holdout frames with a contact inside the model's inner cutoff (penalty region)
    from .md_check import pair_inner_cutoffs

    inner = [pair_inner_cutoffs(pp) for pp in params_paths]
    closest = [_closest(frame) for frame in frames]
    for mi in range(n_models):
        below[mi] = sum(any(r < inner[mi].get(k, 0.0) for k, r in c.items()) for c in closest)
    try:
        for frame in frames:
            for mi, ptr in enumerate(instances):
                energy, pred_forces, pred_stress = predict(wrapper, ptr, frame, cutoffs[mi], with_stress=True)
                if frame.stress is not None and len(frame.stress) == 6:
                    stress_sqerr[mi] += sum((p - r) ** 2 for p, r in zip(pred_stress, frame.stress))
                    pressure_sqerr[mi] += (sum(pred_stress[:3]) / 3 - sum(frame.stress[:3]) / 3) ** 2
                    stress_n[mi] += 1
                    frame_p_err[mi].append(sum(pred_stress[:3]) / 3 - sum(frame.stress[:3]) / 3)
                else:
                    frame_p_err[mi].append(None)
                # .xyzf forces are hartree/bohr; chimes_calculator returns kcal/mol/A
                f_err = f_ref = f_abs = 0.0
                for sym, (pfx, pfy, pfz), ref in zip(frame.symbols, pred_forces, frame.forces):
                    rfx, rfy, rfz = (units.hartree_per_bohr_to_kcal_per_mol_ang(c) for c in ref)
                    e_atom = (pfx - rfx) ** 2 + (pfy - rfy) ** 2 + (pfz - rfz) ** 2
                    f_err += e_atom
                    f_ref += rfx**2 + rfy**2 + rfz**2
                    f_abs += abs(rfx) + abs(rfy) + abs(rfz)
                    if predictions:
                        pred_rows[mi].extend(((pfx, rfx, sym), (pfy, rfy, sym), (pfz, rfz, sym)))
                    er = elem_rows[mi].setdefault(sym, [0.0, 0.0, 0])
                    er[0] += e_atom
                    er[1] += rfx**2 + rfy**2 + rfz**2
                    er[2] += 3
                force_sqerr[mi] += f_err
                force_n[mi] += 3 * frame.natoms
                frame_rows[mi].append([f_err, f_ref, 3 * frame.natoms])
                if mi == 0:
                    ref_force_abs += f_abs
                    ref_force_sq += f_ref
                    ref_force_n += 3 * frame.natoms
                per_frame_energy[mi].append(energy)
                if frame.energy is not None:
                    energy_sqerr[mi] += (energy - frame.energy) ** 2
                    energy_pa_sqerr[mi] += ((energy - frame.energy) / frame.natoms) ** 2
                    energy_n[mi] += 1
                    frame_e_err[mi].append((energy - frame.energy) / frame.natoms)
                else:
                    frame_e_err[mi].append(None)
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
                # the ChIMES papers' "reduced RMSE": RMSE / <|F_DFT|> (mean absolute force component);
                # larger than relative_force_error (1.25x for Gaussian forces, 1.48x on Cu-Zr). Compare
                # published values against this one.
                "reduced_force_rmse": rmse_f / (ref_force_abs / ref_force_n) if rmse_f is not None and ref_force_abs else None,
                "rmse_energy_kcal_mol": math.sqrt(energy_sqerr[mi] / energy_n[mi]) if energy_n[mi] else None,
                "rmse_energy_kcal_mol_per_atom": math.sqrt(energy_pa_sqerr[mi] / energy_n[mi]) if energy_n[mi] else None,
                # stresses (GPa, pressure sign) when the holdout frames carry them
                "rmse_stress_gpa": math.sqrt(stress_sqerr[mi] / (6 * stress_n[mi])) if stress_n[mi] else None,
                "rmse_pressure_gpa": math.sqrt(pressure_sqerr[mi] / stress_n[mi]) if stress_n[mi] else None,
                "n_frames_with_stress": stress_n[mi],
                "n_frames_below_inner_cutoff": below[mi],
            }
        )
        results[-1].update(group_breakdown(frame_rows[mi], groups, elem_rows[mi]))
        if per_frame:
            results[-1]["per_frame_force"] = frame_rows[mi]  # [sq error sum, reference sq sum, n components]
            results[-1]["per_frame_group"] = groups
            results[-1]["per_frame_energy_err_per_atom"] = frame_e_err[mi]
            results[-1]["per_frame_pressure_err_gpa"] = frame_p_err[mi]
        if predictions:
            results[-1]["predictions"] = pred_rows[mi]

    below_note = [f"{r['params']}: {r['n_frames_below_inner_cutoff']} holdout frame(s) have contacts inside the model's "
                  "inner cutoff (penalty region); their errors dominate the RMSE. Put the closest contacts in training "
                  "(data-curate/dataset-select do) or lower s_minim." for r in results if r["n_frames_below_inner_cutoff"]]
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
            "results": results, "committee_spread": committee_spread, **({"warnings": below_note} if below_note else {})}


def run(args) -> dict:
    params_paths = args.params
    if not params_paths:
        raise ValueError("evaluate requires at least one --params (or 'params' in --json-in)")
    if not args.holdout_xyzf:
        raise ValueError("evaluate requires --holdout-xyzf (or 'holdout_xyzf' in --json-in)")

    frames = xyzf_io.read_xyzf(args.holdout_xyzf)
    if getattr(args, "max_frames", None):
        frames = frames[: args.max_frames]
    plot = bool(getattr(args, "plot", False)) and getattr(args, "output_dir", None)
    out = evaluate_frames(frames, params_paths, per_frame=bool(getattr(args, "per_frame", False)), predictions=plot)
    if plot:
        from ..io import fs, plots

        d = fs.ensure_dir(Path(args.output_dir))
        figs = []
        for mi, r in enumerate(out["results"]):
            rows = r.pop("predictions", None) or []
            if rows:
                tag = f"_{mi}" if len(out["results"]) > 1 else ""
                figs.append(plots.parity([p for p, _, _ in rows], [q for _, q, _ in rows], d / f"parity_forces{tag}.png",
                                         labels=[e for _, _, e in rows], title="Force components: model vs reference"))
                comp = {g: v["relative_force_error"] for g, v in (r.get("by_composition") or {}).items()}
                if comp:
                    figs.append(plots.bars(comp, d / f"error_by_composition{tag}.png", ylabel="relative force error",
                                           title="Holdout error by composition"))
        out["plots"] = [f for f in figs if f]
    elif plot is False:
        for r in out["results"]:
            r.pop("predictions", None)
    return out
