"""Minimal LAMMPS data-file writer (atom_style atomic, orthorhombic box
only) for feeding an xyzf.Frame into `lmp_mpi_chimes`.

Correctness-critical: LAMMPS atom type IDs are assigned in the order of
`elements`, and must match the SAME element ordering `fm_setup_gen` used to
build the params.txt this structure will be paired with (chimesFF's
`pair_coeff * * params.txt` maps LAMMPS types to params.txt's internal
type indices positionally, not by element symbol) -- masses must also
match the params.txt's declared masses to within ~0.001 amu (LAMMPS'
mass-based type-matching tolerance for ChIMES). Neither is cross-checked
here (no params.txt parser exists yet); it's the caller's responsibility to
pass the same `elements`/`masses` used for that fit.

Triclinic frames are written as a LAMMPS restricted-triclinic box (a along
x, b in the xy plane, tilt factors reduced into LAMMPS' allowed range). That
is a rigid rotation (plus an equivalent choice of lattice vectors) of the
original cell, so energies are unchanged; forces come back in the rotated
frame and `rotation_for(frame)` maps them to the original one
(F_original = F_lammps @ Q).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def restricted_triclinic(cell):
    """(lx, ly, lz, xy, xz, yz, Q, lammps_cell) for a 3x3 cell (rows = lattice
    vectors). Q rotates original Cartesian vectors into the LAMMPS frame:
    v_lammps = v @ Q.T."""
    cell = np.asarray(cell, dtype=float)
    if np.linalg.det(cell) <= 0:
        raise ValueError("cell must be right-handed (positive determinant) for LAMMPS")
    a, b, c = cell
    lx = np.linalg.norm(a)
    ahat = a / lx
    xy = b @ ahat
    ly = np.sqrt(b @ b - xy * xy)
    xz = c @ ahat
    yz = (b @ c - xy * xz) / ly
    lz = np.sqrt(c @ c - xz * xz - yz * yz)
    new = np.array([[lx, 0, 0], [xy, ly, 0], [xz, yz, lz]])
    q_t = np.linalg.solve(cell, new)  # cell @ Q.T = new
    # reduce tilts (equivalent lattice vectors): c -= k b, c -= k a, b -= k a
    k = round(yz / ly)
    yz, xz = yz - k * ly, xz - k * xy
    xz -= round(xz / lx) * lx
    xy -= round(xy / lx) * lx
    lammps_cell = np.array([[lx, 0, 0], [xy, ly, 0], [xz, yz, lz]])
    return lx, ly, lz, xy, xz, yz, q_t.T, lammps_cell


def rotation_for(frame):
    """Q for a frame (identity when orthorhombic): F_original = F_lammps @ Q."""
    if not frame.non_ortho:
        return np.eye(3)
    return restricted_triclinic(frame.box)[6]


def write_lammps_data(frame, elements: list, masses: dict, path) -> None:
    type_of = {el: i + 1 for i, el in enumerate(elements)}
    unknown = sorted(set(frame.symbols) - set(elements))
    if unknown:
        raise ValueError(f"frame contains element(s) not in `elements`: {unknown}")

    positions = np.asarray(frame.positions, dtype=float)
    if frame.non_ortho:
        cell = np.asarray(frame.box, dtype=float)
        lx, ly, lz, xy, xz, yz, q, lcell = restricted_triclinic(cell)
        frac = positions @ np.linalg.inv(cell)
        frac -= np.floor(frac)
        rot_cell = cell @ q.T
        positions = frac @ rot_cell
        # wrap into the reduced LAMMPS cell (same lattice, different vectors)
        f2 = positions @ np.linalg.inv(lcell)
        positions = (f2 - np.floor(f2)) @ lcell
        lx, ly, lz, xy, xz, yz = (float(v) for v in (lx, ly, lz, xy, xz, yz))
        box_lines = [f"0.0 {lx!r} xlo xhi", f"0.0 {ly!r} ylo yhi", f"0.0 {lz!r} zlo zhi", f"{xy!r} {xz!r} {yz!r} xy xz yz"]
    else:
        lx, ly, lz = frame.box
        box_lines = [f"0.0 {lx} xlo xhi", f"0.0 {ly} ylo yhi", f"0.0 {lz} zlo zhi"]
    lines = [
        "LAMMPS data file via agentic-chimes (atom_style atomic)",
        "",
        f"{frame.natoms} atoms",
        f"{len(elements)} atom types",
        "",
        *box_lines,
        "",
        "Masses",
        "",
    ]
    for i, el in enumerate(elements, start=1):
        if el not in masses:
            raise ValueError(f"no mass given for element {el!r} (masses={masses})")
        lines.append(f"{i} {masses[el]}  # {el}")

    lines += ["", "Atoms  # atomic", ""]
    for i, (sym, pos) in enumerate(zip(frame.symbols, positions.tolist()), start=1):
        lines.append(f"{i} {type_of[sym]} {pos[0]} {pos[1]} {pos[2]}")

    Path(path).write_text("\n".join(lines) + "\n")
