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

Non-orthorhombic (triclinic) frames are not supported by this writer --
raises a clear error rather than writing a wrong/ignored box.
"""

from __future__ import annotations

from pathlib import Path


def write_lammps_data(frame, elements: list, masses: dict, path) -> None:
    if frame.non_ortho:
        raise ValueError(
            "write_lammps_data only supports orthorhombic frames (frame.non_ortho=False); "
            "triclinic box writing is not implemented"
        )

    type_of = {el: i + 1 for i, el in enumerate(elements)}
    unknown = sorted(set(frame.symbols) - set(elements))
    if unknown:
        raise ValueError(f"frame contains element(s) not in `elements`: {unknown}")

    lx, ly, lz = frame.box
    lines = [
        "LAMMPS data file via agentic-chimes (atom_style atomic)",
        "",
        f"{frame.natoms} atoms",
        f"{len(elements)} atom types",
        "",
        f"0.0 {lx} xlo xhi",
        f"0.0 {ly} ylo yhi",
        f"0.0 {lz} zlo zhi",
        "",
        "Masses",
        "",
    ]
    for i, el in enumerate(elements, start=1):
        if el not in masses:
            raise ValueError(f"no mass given for element {el!r} (masses={masses})")
        lines.append(f"{i} {masses[el]}  # {el}")

    lines += ["", "Atoms  # atomic", ""]
    for i, (sym, pos) in enumerate(zip(frame.symbols, frame.positions), start=1):
        lines.append(f"{i} {type_of[sym]} {pos[0]} {pos[1]} {pos[2]}")

    Path(path).write_text("\n".join(lines) + "\n")
