"""Check an fm_setup.in against its training trajectory before chimes_lsq runs.

chimes_lsq crashes rather than explaining: NFRAMES larger than the trajectory
made it segfault (exit -11, plus a core file in the output directory). This
reproduces its own checks in Python and returns readable problems:

- errors: trajectory missing; NFRAMES > frames in the file; elements
  outside the atom types; an outer cutoff that breaks chimes_lsq's own
  box rule (cutoff < |lattice vector| x (2 N_LAYERS + 1) / 2, as in
  ClassDefs.C IS_RCUT_SAFE); stresses or energies requested but missing.
- warnings: NFRAMES < frames (the rest are ignored); an inner cutoff above
  the closest sampled contact (training distances inside the penalty
  region) or more than 0.1 A below it (the model extrapolates there).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..io import fm_setup
from ..io import xyzf as xyzf_io


def _closest(frames, rmax: float) -> dict:
    from ase import Atoms
    from ase.neighborlist import neighbor_list

    best = {}
    for f in frames:
        cell = np.asarray(f.box if f.non_ortho else np.diag(f.box), dtype=float)
        at = Atoms(symbols=f.symbols, positions=f.positions, cell=cell, pbc=True)
        i, j, d = neighbor_list("ijd", at, rmax)
        sym = np.array(f.symbols)
        for a, b, r in zip(sym[i], sym[j], d):
            key = tuple(sorted((a, b)))
            best[key] = min(best.get(key, np.inf), float(r))
    return best


def check(fm_setup_path) -> dict:
    p = fm_setup.parse(Path(fm_setup_path).read_text())
    errors, warnings = [], []
    trj = str(p.get("trjfile", "")).strip()
    if trj.upper().startswith("MULTI"):
        return {"errors": [], "warnings": ["TRJFILE MULTI: trajectory list not checked"], "checked": False}
    trj_path = Path(trj)
    if not trj_path.is_absolute():
        trj_path = Path(fm_setup_path).parent / trj_path
    if not trj_path.is_file():
        return {"errors": [f"TRJFILE {trj_path} does not exist"], "warnings": [], "checked": False}
    frames = xyzf_io.read_xyzf(trj_path)
    n = int(p.get("nframes") or 0)
    if n > len(frames):
        errors.append(f"NFRAMES is {n} but {trj_path.name} holds {len(frames)} frames (chimes_lsq would segfault)")
    elif n < len(frames):
        warnings.append(f"NFRAMES is {n}; the last {len(frames) - n} of {len(frames)} frames will be ignored")
    used = frames[: min(n, len(frames))] if n else frames

    types = {a["symbol"] for a in p.get("atom_types", [])}
    extra = sorted({s for f in used for s in f.symbols} - types)
    if extra:
        errors.append(f"trajectory contains {extra}, which are not ATOM TYPES {sorted(types)}")

    fitstrs = str(p.get("fitstrs", "false")).lower()
    if fitstrs != "false":
        need = 6 if fitstrs.startswith(("all", "firstall")) else 3
        missing = sum(1 for f in used if f.stress is None or len(f.stress) < need)
        if missing:
            errors.append(f"FITSTRS {p.get('fitstrs')} but {missing} frame(s) have no {'full ' if need == 6 else ''}stress tensor")
    if str(p.get("fitener", "false")).lower() not in ("false", "") and any(f.energy is None for f in used):
        errors.append(f"FITENER {p.get('fitener')} but {sum(f.energy is None for f in used)} frame(s) have no energy")

    nl = int(p.get("nlayers") or 0)
    cut = max([pr["s_maxim"] for pr in p.get("pairs", [])] + [
        float(b["value"]) if b["mode"] == "ALL" else max(float(x) for r in b["rows"] for x in r[len(r) // 2 + 1:])
        for b in p.get("special_blocks", []) if b.get("bound") == "S_MAXIM"] or [0.0])
    worst = None
    for k, f in enumerate(used):
        cell = np.asarray(f.box if f.non_ortho else np.diag(f.box), dtype=float)
        lim = 0.5 * np.linalg.norm(cell, axis=1).min() * (2 * nl + 1)
        if cut >= lim and (worst is None or lim < worst[1]):
            worst = (k, lim)
    if worst:
        errors.append(f"outer cutoff {cut} A >= {worst[1]:.2f} A allowed by frame {worst[0]} with NLAYERS {nl}: "
                      "raise NLAYERS or shorten the cutoff")

    smin = {tuple(sorted((pr["type1"], pr["type2"]))): pr["s_minim"] for pr in p.get("pairs", [])}
    if used and smin:
        close = _closest(used, max(smin.values()) + 1.0)
        for key, s in smin.items():
            r = close.get(key)
            if r is None:
                continue
            if r < s:
                warnings.append(f"{'-'.join(key)}: closest training distance {r:.3f} A is inside S_MINIM {s} "
                                "(those rows sit in the penalty region)")
            elif r - s > 0.1:
                warnings.append(f"{'-'.join(key)}: S_MINIM {s} is {r - s:.2f} A below the closest training distance "
                                f"{r:.3f} A (documented practice: ~0.02 A below); the model extrapolates there")
    return {"errors": errors, "warnings": warnings, "checked": True, "n_frames_in_file": len(frames)}
