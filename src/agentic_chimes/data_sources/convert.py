"""DFT records (eV, eV/Angstrom, any cell orientation) -> ChIMES `.xyzf` frames
(kcal/mol, hartree/bohr -- the same contract as `converters/qe2xyzf.py`).

Cells are rotated to the lower-triangular standard form (a along x, b in the
xy-plane) that chimes_lsq's own NON_ORTHO fixtures and LAMMPS both use;
positions and forces are rotated with them, so every physical quantity is
unchanged. Cells that are orthorhombic after rotation are written in the
plain `Lx Ly Lz` form.
"""

from __future__ import annotations

import numpy as np

from ..converters import units
from ..io.xyzf import Frame

ORTHO_TOL = 1e-6


def to_frame(symbols, cell, positions, energy_ev=None, forces_ev_ang=None, stress_ev_ang3=None,
             stress_sign: str = "cauchy") -> Frame:
    """stress_ev_ang3: 3x3 stress in eV/A^3 in the input orientation, with
    `stress_sign` "cauchy" (tensile positive: ASE) or "pressure" (compressive
    positive: VASP-derived records such as MatPES in ColabFit). Rotated with
    the cell and stored in ChIMES' convention (GPa, pressure sign; see
    converters/units.py)."""
    from ase.cell import Cell

    cell = np.asarray(cell, dtype=float)
    pos = np.asarray(positions, dtype=float)
    rcell, Q = Cell(cell).standard_form()
    pos = pos @ Q.T
    if forces_ev_ang is None:
        forces = np.zeros_like(pos)
    else:
        forces = units.ev_per_ang_to_hartree_per_bohr(np.asarray(forces_ev_ang, dtype=float) @ Q.T)

    stress = None
    if stress_ev_ang3 is not None:
        sigma = np.asarray(stress_ev_ang3, dtype=float).reshape(3, 3)
        if stress_sign not in ("cauchy", "pressure"):
            raise ValueError(f"stress_sign must be cauchy or pressure, got {stress_sign!r}")
        stress = units.cauchy_ev_ang3_to_chimes_gpa(Q @ sigma @ Q.T)
        if stress_sign == "pressure":
            stress = [-x for x in stress]

    if np.all(np.abs(rcell - np.diag(np.diag(rcell))) < ORTHO_TOL):
        box, non_ortho = [float(x) for x in np.diag(rcell)], False
    else:
        box, non_ortho = [[float(x) for x in row] for row in rcell], True

    return Frame(
        symbols=list(symbols),
        positions=pos.tolist(),
        forces=np.asarray(forces).tolist(),
        box=box,
        non_ortho=non_ortho,
        stress=stress,
        energy=None if energy_ev is None else float(units.ev_to_kcal_per_mol(energy_ev)),
    )


def frame_to_atoms(frame: Frame):
    """ChIMES frame -> ase.Atoms (positions/cell only), for geometry analysis."""
    from ase import Atoms

    cell = frame.box if frame.non_ortho else np.diag(frame.box)
    return Atoms(symbols=frame.symbols, positions=frame.positions, cell=cell, pbc=True)


def forces_ev_ang(frame: Frame) -> np.ndarray:
    return units.hartree_per_bohr_to_ev_per_ang(np.asarray(frame.forces, dtype=float))


def energy_ev(frame: Frame):
    return None if frame.energy is None else units.kcal_per_mol_to_ev(frame.energy)


def unique_pairs(atoms, cutoff: float):
    """(i, j, d) for every distinct atom pair within `cutoff`, counting
    periodic images, each pair once.

    ASE lists each pair in both directions. For i != j keep i < j. For
    i == j (an atom and its own periodic image, the *only* neighbours in a
    1-atom cell) keep the direction whose shift vector is lexicographically
    positive. Filtering on i < j alone drops every self-image pair, which
    made 1-atom bulk crystals look like isolated atoms.
    """
    from ase.neighborlist import neighbor_list

    i, j, d, S = neighbor_list("ijdS", atoms, cutoff)
    first_nonzero = np.where(S[:, 0] != 0, S[:, 0], np.where(S[:, 1] != 0, S[:, 1], S[:, 2]))
    keep = (i < j) | ((i == j) & (first_nonzero > 0))
    return i[keep], j[keep], d[keep]


def stress_sign_check(frames, min_group: int = 5) -> dict:
    """Is the stored stress sign right? Within one composition, pressure
    (mean of the xyzf stress diagonal) must fall as volume per atom rises.
    Returns the median correlation over compositions with >= min_group
    stressed frames and a verdict: "ok" (median < -0.3), "flipped"
    (> +0.3) or "unverified"."""
    from collections import defaultdict

    groups = defaultdict(list)
    for f in frames:
        if f.stress is None or len(f.stress) < 3:
            continue
        cell = np.asarray(f.box if f.non_ortho else np.diag(f.box), dtype=float)
        groups[tuple(sorted(f.symbols))].append((abs(np.linalg.det(cell)) / f.natoms, sum(f.stress[:3]) / 3))
    cors = []
    for rows in groups.values():
        if len(rows) >= min_group:
            v, p = np.asarray(rows).T
            if np.std(v) > 0 and np.std(p) > 0:
                cors.append(float(np.corrcoef(v, p)[0, 1]))
    if not cors:
        return {"verdict": "unverified", "median_corr": None, "n_groups": 0}
    med = float(np.median(cors))
    verdict = "ok" if med < -0.3 else "flipped" if med > 0.3 else "unverified"
    return {"verdict": verdict, "median_corr": round(med, 3), "n_groups": len(cors)}
