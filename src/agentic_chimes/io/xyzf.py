"""Reader/writer for ChIMES `.xyzf` trajectory files.

Format (confirmed against real fixtures in
`codes/chimes_lsq-LLfork/test_suite-lsq/{test_4atoms.2,stress-and-ener-2b1}/`):
per frame, a natoms line, then a comment line of
`Lx Ly Lz [sxx syy szz sxy sxz syz] [energy]` (or, for a non-orthorhombic
cell, `NON_ORTHO <9 lattice components: a, b, c row vectors> [sxx syy szz sxy sxz syz] [energy]`;
chimes_lsq matches the underscore spelling exactly -- ClassDefs.C tests
`tokens[0] == "NON_ORTHO"` -- and its fixtures use lower-triangular cells),
then `natoms` rows of `<element> x y z fx fy fz`. Units follow whatever the
file's own convention is (this module does not convert -- see
converters.units for that).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Frame:
    symbols: list
    positions: list  # N x 3
    forces: list  # N x 3
    box: list  # orthorhombic: [Lx, Ly, Lz]; non-ortho: 3x3 nested list
    non_ortho: bool = False
    stress: Optional[list] = None  # [sxx, syy, szz, sxy, sxz, syz]
    energy: Optional[float] = None

    @property
    def natoms(self) -> int:
        return len(self.symbols)


def read_xyzf(path) -> list:
    frames = []
    with open(path) as f:
        lines = [ln.rstrip("\n") for ln in f]

    i = 0
    n = len(lines)
    while i < n:
        if not lines[i].strip():
            i += 1
            continue
        natoms = int(lines[i].split()[0])
        i += 1
        comment = lines[i].split()
        i += 1

        non_ortho = comment[0].upper() in ("NON_ORTHO", "NON-ORTHO")
        if non_ortho:
            nums = [float(x) for x in comment[1:]]
            box = [nums[0:3], nums[3:6], nums[6:9]]
            rest = nums[9:]
            if all(box[r][c] == 0.0 for r in range(3) for c in range(3) if r != c):
                box, non_ortho = [box[0][0], box[1][1], box[2][2]], False
        else:
            nums = [float(x) for x in comment]
            box = nums[0:3]
            rest = nums[3:]

        stress = None
        energy = None
        if len(rest) == 7:
            stress, energy = rest[:6], rest[6]
        elif len(rest) == 6:
            stress = rest
        elif len(rest) == 1:
            energy = rest[0]

        symbols, positions, forces = [], [], []
        for _ in range(natoms):
            toks = lines[i].split()
            i += 1
            symbols.append(toks[0])
            positions.append([float(toks[1]), float(toks[2]), float(toks[3])])
            forces.append([float(toks[4]), float(toks[5]), float(toks[6])])

        frames.append(
            Frame(
                symbols=symbols,
                positions=positions,
                forces=forces,
                box=box,
                non_ortho=non_ortho,
                stress=stress,
                energy=energy,
            )
        )

    return frames


def write_xyzf(frames: list, path) -> None:
    """If any frame is triclinic, every frame is written as NON_ORTHO
    (orthorhombic ones with zero off-diagonals). chimes_lsq segfaults when a
    plain `Lx Ly Lz` frame follows a NON_ORTHO one in the same trajectory;
    it accepts diagonal NON_ORTHO cells and treats them as orthorhombic."""
    all_non_ortho = any(fr.non_ortho for fr in frames)
    lines = []
    for fr in frames:
        lines.append(str(fr.natoms))
        if fr.non_ortho:
            comment = ["NON_ORTHO"] + [str(x) for row in fr.box for x in row]
        elif all_non_ortho:
            lx, ly, lz = fr.box
            comment = ["NON_ORTHO"] + [str(x) for x in (lx, 0.0, 0.0, 0.0, ly, 0.0, 0.0, 0.0, lz)]
        else:
            comment = [str(x) for x in fr.box]
        if fr.stress is not None:
            comment += [str(x) for x in fr.stress]
        if fr.energy is not None:
            comment += [str(fr.energy)]
        lines.append(" ".join(comment))
        for sym, pos, frc in zip(fr.symbols, fr.positions, fr.forces):
            lines.append(f"{sym} {pos[0]} {pos[1]} {pos[2]} {frc[0]} {frc[1]} {frc[2]}")
    Path(path).write_text("\n".join(lines) + "\n")
