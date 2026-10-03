"""Facts read from a ChIMES params.txt that other stages must agree with.

LAMMPS' ChIMES pair style matches atom types to params.txt *by mass*
(tolerance ~1e-3 amu). A mass typed slightly differently (Cu 63.5 instead of
the model's 63.546) is not recognized, and on a Cu-Zr cell the energy came
out at -183.6 instead of -272.3 kcal/mol with no error. Swapped masses
relabel atoms silently. So every LAMMPS stage takes element types and masses
from params.txt and refuses inputs that disagree.
"""

from __future__ import annotations

from pathlib import Path

MASS_TOL = 1e-3


def model_types(params_path) -> list:
    """[(symbol, mass)] in the model's type order."""
    lines = Path(params_path).read_text().splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if "TYPEIDX" in ln and "ATMMASS" in ln)
    except StopIteration:
        raise ValueError(f"{params_path}: no '# TYPEIDX # ... # ATMMASS #' table; is this a ChIMES params.txt?") from None
    out = []
    for ln in lines[start + 1:]:
        t = ln.split()
        if len(t) < 4 or not t[0].lstrip("-").isdigit():
            break
        out.append((t[1], float(t[3])))
    if not out:
        raise ValueError(f"{params_path}: empty atom-type table")
    return out


def resolve_types(params_path, elements=None, masses=None) -> tuple:
    """(elements, masses) for LAMMPS, taken from params.txt.

    `elements` and `masses` given by a caller are checked, not used: every
    element must be one of the model's, and every given mass must match the
    model's within MASS_TOL. Raises ValueError naming each disagreement."""
    types = model_types(params_path)
    model_els = [s for s, _ in types]
    model_mass = dict(types)
    problems = []
    for el in elements or []:
        if el not in model_mass:
            problems.append(f"element {el!r} is not in the model (model types: {model_els})")
    for el, m in (masses or {}).items():
        if el in model_mass and abs(float(m) - model_mass[el]) > MASS_TOL:
            problems.append(f"mass of {el} is {m} but the model was fitted with {model_mass[el]} "
                            "(LAMMPS matches types by mass: a different value silently mis-assigns or drops the type)")
        elif el not in model_mass:
            problems.append(f"mass given for {el!r}, which is not in the model (model types: {model_els})")
    if problems:
        raise ValueError(f"{params_path}: " + "; ".join(problems))
    return model_els, model_mass


def frame_elements_check(params_path, symbols) -> None:
    """Refuse a structure containing an element the model does not describe."""
    model_els = [s for s, _ in model_types(params_path)]
    extra = sorted(set(symbols) - set(model_els))
    if extra:
        raise ValueError(f"structure contains {extra}, which the model {params_path} does not describe ({model_els})")
