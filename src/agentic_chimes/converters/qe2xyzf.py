"""Quantum ESPRESSO `pw.x` stdout -> ChIMES `.xyzf` frame conversion.

Mirrors `codes/chimes_lsq-LLfork/contrib/vasp2xyzf.py`'s output contract
(same per-frame `.xyzf` block layout, same target convention: energy in
kcal/mol, forces in hartree/bohr) but is a from-scratch Python 3
implementation parsing QE's pw.x text output instead of a VASP OUTCAR --
there is no upstream QE support anywhere in the vendored forks to port.

Scope: **single-point SCF only** (`calculation = 'scf'`). Ionic positions
don't change in an SCF run, so this module does not re-parse QE's echoed
geometry at all -- it takes the *original* structure (the one qe-relabel
generated pw.in from) and merges QE's parsed energy/forces onto it. This
sidesteps QE's several different position-echo unit conventions (alat,
crystal, angstrom) entirely. A relax/vc-relax run's final geometry would
need different handling; not implemented here.

Units: QE reports energy in Rydberg and forces in "Ry/au" (Ry/bohr; QE's
internal length unit is already bohr, so only the energy half of the
Rydberg needs converting) -- see converters/units.py's ry_to_kcal_per_mol
and ry_per_bohr_to_hartree_per_bohr.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from . import units

_ENERGY_RE = re.compile(r"^!\s+total energy\s+=\s+(-?[\d.]+)\s+Ry", re.MULTILINE)
_FORCE_RE = re.compile(r"^\s*atom\s+(\d+)\s+type\s+(\d+)\s+force\s*=\s*(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)", re.MULTILINE)
_CONVERGED_RE = re.compile(r"convergence has been achieved")


@dataclass
class QEResult:
    converged: bool
    energy_ry: Optional[float]
    forces_ry_bohr: Optional[list]  # ordered by QE atom index (1-based in the output, 0-based here)


def parse_pwx_output(text: str) -> QEResult:
    converged = bool(_CONVERGED_RE.search(text))

    energy_matches = _ENERGY_RE.findall(text)
    energy_ry = float(energy_matches[-1]) if energy_matches else None

    force_matches = _FORCE_RE.findall(text)
    forces_ry_bohr = None
    if force_matches:
        # sort by QE's 1-based atom index (group 1) to guarantee order
        # matches the original structure regardless of any reordering in
        # the printed block
        ordered = sorted(force_matches, key=lambda m: int(m[0]))
        forces_ry_bohr = [[float(fx), float(fy), float(fz)] for _idx, _typ, fx, fy, fz in ordered]

    return QEResult(converged=converged, energy_ry=energy_ry, forces_ry_bohr=forces_ry_bohr)


def frame_from_qe_output(base_frame, pwx_stdout: str):
    """base_frame: an xyzf.Frame with the original (pre-DFT) symbols/box/
    positions used to generate the pw.in this output came from. Returns a
    new Frame with energy/forces filled in from the QE output, in the
    target kcal/mol / hartree-bohr convention. Raises ValueError if the
    output has no parseable energy or a force-count mismatch."""
    result = parse_pwx_output(pwx_stdout)

    if result.energy_ry is None:
        raise ValueError("no '!    total energy = ... Ry' line found in pw.x output (job may have crashed or not converged)")
    if result.forces_ry_bohr is None:
        raise ValueError("no 'Forces acting on atoms' block found in pw.x output (was tprnfor=.true. set?)")
    if len(result.forces_ry_bohr) != len(base_frame.symbols):
        raise ValueError(
            f"pw.x output has {len(result.forces_ry_bohr)} force rows but the base structure has "
            f"{len(base_frame.symbols)} atoms"
        )

    from ..io.xyzf import Frame  # local import to avoid a cycle at module load time

    energy_kcal_mol = units.ry_to_kcal_per_mol(result.energy_ry)
    forces_hartree_bohr = [
        [units.ry_per_bohr_to_hartree_per_bohr(c) for c in row] for row in result.forces_ry_bohr
    ]

    return Frame(
        symbols=list(base_frame.symbols),
        positions=[list(p) for p in base_frame.positions],
        forces=forces_hartree_bohr,
        box=base_frame.box,
        non_ortho=base_frame.non_ortho,
        energy=energy_kcal_mol,
    ), result.converged
