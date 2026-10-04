"""Hierarchical (element-block) fitting, following al_driver 2.0 and
Lindsey et al., npj Comput. Mater. 12, 18 (2026).

A multi-element model's parameters split into blocks:

- pure blocks {X, XX, XXX, XXXX}, fitted once from single-element data and
  reused;
- cross blocks {XY, XXY, XYY, ...}, fitted to mixture data with the pure
  blocks held fixed.

The paper found this as accurate as fitting everything at once. It is also
a principled alternative to dropping cross cluster types.

Two modes, around an ordinary fit:

1. `--subtract`: for each element model, keep only that element's atoms in
   every frame, predict their energy/forces/stress with the element model
   (exact supercells, like `evaluate`) and subtract them from the DFT labels.
   This writes `residual.xyzf`. As in al_driver, each element model's
   repulsive penalty is set to zero first; otherwise close contacts would
   subtract unphysical energy.
2. Fit the cross blocks on `residual.xyzf` with `fm-setup-gen` (`hierarc`
   true; `exclude_1b` [[X], [Y]], `exclude_2b` [[X, X], [Y, Y]], plus pure
   3-/4-body types in `exclude_3b`/`exclude_4b`; each element's pair
   parameters must match its element model), then `model-build`.
3. `--combine`: merge the cross params.txt with the element params.txt
   files into one model (al_driver's `hierarch.py`).
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

import numpy as np

from ..converters import units
from ..io import xyzf as xyzf_io
from ..io import fs

NAME = "hierarch"
SUMMARY = "Hierarchical fitting: subtract fixed element models from mixture data (--subtract), merge cross + element params (--combine)."
SCHEMA = {
    "type": "object",
    "properties": {
        "element_params": {"type": "array", "items": {"type": "string"}, "description": "params.txt of each fixed element model (one element each)."},
        "subtract": {"type": ["string", "null"], "description": "Training .xyzf to subtract the element models from."},
        "combine": {"type": ["string", "null"], "description": "Cross-term params.txt to merge with element_params."},
    },
}


def add_arguments(parser) -> None:
    parser.add_argument("--element-params", dest="element_params", action="append", default=None)
    parser.add_argument("--subtract", default=None)
    parser.add_argument("--combine", default=None)


def _elements_of(params_path) -> list:
    from ..io.params import model_types

    return [el for el, _ in model_types(params_path)]


def zero_penalty_copy(params_path, out_path) -> Path:
    """Copy of params.txt with the pair penalty explicitly off (after FCUT TYPE, like al_driver's files)."""
    from ..io.params import set_penalty

    return set_penalty(params_path, out_path, dist=0.0, scaling=0.0)


def subtract(frames, element_params: list, work: Path) -> tuple:
    from . import evaluate

    wrapper = evaluate._load_wrapper()
    models = []
    for p in element_params:
        els = _elements_of(p)
        if len(els) != 1:
            raise ValueError(f"{p} describes {els}; element models must have exactly one element")
        zp = zero_penalty_copy(p, work / f"{els[0]}.nopenalty.params.txt")
        ptr = wrapper.chimes_open_instance()
        wrapper.set_chimes_instance(ptr, small=False)
        wrapper.init_chimes_instance(ptr, str(zp), 0)
        models.append((els[0], ptr, evaluate.max_outer_cutoff(zp)))
    stats = {el: {"energy_kcal_mol": [], "force_rms_kcal_mol_ang": []} for el, _, _ in models}
    out = []
    try:
        for fr in frames:
            forces = np.asarray(fr.forces, dtype=float).copy()        # hartree/bohr
            energy = fr.energy
            stress = None if fr.stress is None else list(fr.stress)
            for el, ptr, cut in models:
                idx = [i for i, s in enumerate(fr.symbols) if s == el]
                if not idx:
                    continue
                sub = xyzf_io.Frame(symbols=[el] * len(idx), positions=[fr.positions[i] for i in idx],
                                    forces=[[0.0] * 3] * len(idx), box=fr.box, non_ortho=fr.non_ortho)
                e, f, s = evaluate.predict(wrapper, ptr, sub, cut, with_stress=True)
                f = np.asarray(f)
                forces[idx] -= f / units.HARTREE_PER_BOHR_TO_KCAL_PER_MOL_ANG
                if energy is not None:
                    energy -= e
                if stress is not None:
                    stress = [a - b for a, b in zip(stress, s)]
                stats[el]["energy_kcal_mol"].append(e)
                stats[el]["force_rms_kcal_mol_ang"].append(float(np.sqrt(np.mean(f ** 2))))
            out.append(xyzf_io.Frame(symbols=list(fr.symbols), positions=fr.positions, forces=forces.tolist(), box=fr.box,
                                     non_ortho=fr.non_ortho, stress=stress, energy=energy))
    finally:
        for _, ptr, _ in models:
            wrapper.chimes_close_instance(ptr)
    summary = {el: {"frames_with_element": len(v["energy_kcal_mol"]),
                    "mean_force_rms_removed_kcal_mol_ang": float(np.mean(v["force_rms_kcal_mol_ang"])) if v["energy_kcal_mol"] else None}
               for el, v in stats.items()}
    return out, summary


@contextlib.contextmanager
def _chdir(path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def combine(cross_params, element_params: list, work: Path) -> Path:
    from .. import config

    src = str(config.AL_DRIVER_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)
    import hierarch as al_hierarch  # type: ignore

    fs.ensure_dir(work)
    # Mirrors al_hierarch.main (template = cross terms, each element file merged in turn), with one guard:
    # update_from() marks the special 4-body cutoff block "specific" even when neither model has 4-body
    # terms, and print_file() then fails on its empty count. Such a block is switched off instead.
    template_path = Path(cross_params)
    for i, sup in enumerate(element_params):
        template = [x for x in template_path.read_text().splitlines(keepends=True) if not x.startswith("!")]
        supplement = [x for x in Path(sup).read_text().splitlines(keepends=True) if not x.startswith("!")]
        merged = al_hierarch.param_file(template, str(template_path))
        merged.update_from(al_hierarch.param_file(supplement, str(sup)))
        for block in (merged.s3b_block, merged.s4b_block):
            if block.specific and block.number is None:
                block.specific = False
                block.all = False
        out_path = work / f"hierarch.{i}.params.txt"
        with open(out_path, "w") as fh:
            merged.print_file(to=fh)
        template_path = out_path
    final = work / "hierarch.params.txt"
    final.write_text(template_path.read_text())
    return final


def run(args) -> dict:
    element_params = [str(Path(p).resolve()) for p in (getattr(args, "element_params", None) or [])]
    if not element_params:
        raise ValueError("hierarch needs --element-params (one per fixed element model)")
    out = Path(getattr(args, "output_dir", None) or ".").resolve()
    fs.ensure_dir(out)
    if getattr(args, "subtract", None):
        frames = xyzf_io.read_xyzf(args.subtract)
        residual, summary = subtract(frames, element_params, out)
        path = out / "residual.xyzf"
        xyzf_io.write_xyzf(residual, path)
        elements = sorted({s for f in frames for s in f.symbols})
        return {"residual_xyzf": str(path), "n_frames": len(residual), "removed": summary,
                "next": ("fit the cross terms on residual_xyzf: fm-setup-gen with hierarc=true, "
                         f"exclude_1b={[[e] for e in elements]}, exclude_2b={[[e, e] for e in elements]}, pure 3-/4-body "
                         "types in exclude_3b/exclude_4b, element pair parameters equal to the element models'; then "
                         "hierarch --combine <cross params.txt>")}
    if getattr(args, "combine", None):
        path = combine(args.combine, element_params, out)
        return {"params": str(path), "elements": _elements_of(path)}
    raise ValueError("hierarch needs --subtract <xyzf> or --combine <cross params.txt>")
