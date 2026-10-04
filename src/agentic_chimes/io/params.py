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


# ------------------------------------------------------------------ penalty

# chimes_lsq's documentation: add penalty parameters before MD; 1e5-1e6
# kcal/mol/A^3 and 0.01-0.05 A are reasonable. Without these lines chimesFF
# falls back to 1e4 and 0.01 A.
DEFAULT_PENALTY_DIST = 0.02
DEFAULT_PENALTY_SCALING = 1.0e5


def penalty(params_path) -> dict:
    """{"dist", "scaling", "explicit"}: what chimesFF will use for this file."""
    dist, scaling = None, None
    for ln in Path(params_path).read_text().splitlines():
        if ln.startswith("PAIR CHEBYSHEV PENALTY DIST:"):
            dist = float(ln.split()[-1])
        elif ln.startswith("PAIR CHEBYSHEV PENALTY SCALING:"):
            scaling = float(ln.split()[-1])
    return {"dist": 0.01 if dist is None else dist, "scaling": 1.0e4 if scaling is None else scaling,
            "explicit": dist is not None and scaling is not None}


def set_penalty(src, dst, *, dist: float, scaling: float) -> Path:
    """Copy params.txt with explicit penalty lines (placed after FCUT TYPE, where
    chimes_lsq's documentation and al_driver put them)."""
    lines = [ln for ln in Path(src).read_text().splitlines() if "PAIR CHEBYSHEV PENALTY" not in ln]
    k = next((i for i, ln in enumerate(lines) if ln.startswith("FCUT TYPE:")), None)
    if k is None:  # no FCUT TYPE (hand-edited file): before the coefficient blocks, still in chimesFF's header pass
        k = next((i - 1 for i, ln in enumerate(lines)
                  if ln.startswith(("PAIR CHEBYSHEV PARAMS", "PAIRTYPE PARAMS", "ENDFILE"))), None)
    if k is None:
        raise ValueError(f"{src}: no FCUT TYPE / PAIRTYPE PARAMS / ENDFILE line to place the penalty settings")
    lines[k + 1:k + 1] = ["", f"PAIR CHEBYSHEV PENALTY DIST:    {float(dist)}", f"PAIR CHEBYSHEV PENALTY SCALING: {float(scaling)}"]
    Path(dst).write_text("\n".join(lines) + "\n")
    return Path(dst)


# ------------------------------------------------------------------ coefficients

def nonzero_by_body(params_path) -> dict:
    """{"2b": n, "3b": n, "4b": n}: unique coefficients that are nonzero
    (LASSO zeroes many; zeros still cost MD time unless the file is reduced)."""
    counts = {"2b": 0, "3b": 0, "4b": 0}
    section, seen = None, set()
    for ln in Path(params_path).read_text().splitlines():
        s = ln.strip()
        if s.startswith("PAIRTYPE PARAMS:"):
            section, block = "2b", s
            continue
        if s.startswith("TRIPLETTYPE PARAMS:"):
            section, block = "3b", None
            continue
        if s.startswith(("QUADRUPLETYPE PARAMS:", "QUADRUPLETTYPE PARAMS:")):  # chimes_lsq spells it QUADRUPLETYPE
            section, block = "4b", None
            continue
        if s.startswith("INDEX:") and section in ("3b", "4b"):
            block = (section, s)
            continue
        if s.startswith(("TRIPMAPS", "QUADMAPS", "PAIRMAPS", "ENDFILE", "NO ENERGY")):
            section = None
            continue
        t = s.split()
        if section == "2b" and len(t) == 2 and t[0].isdigit():
            if float(t[1]) != 0.0:
                counts["2b"] += 1
        elif section in ("3b", "4b") and len(t) >= (7 if section == "3b" else 10) and t[0].isdigit():
            key = (block, t[-2])
            if key not in seen:
                seen.add(key)
                if float(t[-1]) != 0.0:
                    counts[section] += 1
    return counts


def reduce(params_path, dst) -> Path:
    """Drop zeroed 3-/4-body parameters with chimes_lsq's post_proc_chimes_lsq.py.
    Predictions are unchanged (checked to 0.0 on a Cu-Zr 4-body model, which
    then evaluated 21% faster)."""
    import shutil
    import subprocess
    import sys
    import tempfile

    from .. import config

    script = config.CHIMES_LSQ_ROOT / "src" / "post_proc_chimes_lsq.py"
    if not script.is_file():
        raise FileNotFoundError(f"{script} not found; run `chimes-agent setup --component codes`")
    with tempfile.TemporaryDirectory() as td:
        shutil.copy(params_path, Path(td) / "params.txt")
        proc = subprocess.run([sys.executable, str(script), "params.txt"], cwd=td, capture_output=True, text=True)
        out = Path(td) / "params.txt.reduced"
        if proc.returncode != 0 or not out.is_file():
            raise RuntimeError(f"post_proc_chimes_lsq.py failed: {(proc.stdout + proc.stderr)[-400:]}")
        shutil.copy(out, dst)
    return Path(dst)
