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


def to_frame(symbols, cell, positions, energy_ev=None, forces_ev_ang=None) -> Frame:
    from ase.cell import Cell

    cell = np.asarray(cell, dtype=float)
    pos = np.asarray(positions, dtype=float)
    rcell, Q = Cell(cell).standard_form()
    pos = pos @ Q.T
    if forces_ev_ang is None:
        forces = np.zeros_like(pos)
    else:
        forces = units.ev_per_ang_to_hartree_per_bohr(np.asarray(forces_ev_ang, dtype=float) @ Q.T)

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
