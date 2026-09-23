"""Single source of truth for unit conversions used across the pipeline.
Centralized so a wrong factor is a one-line fix, not a hunt through every
stage -- these are exactly the constants a past hand-rolled pipeline for
this workflow got bitten by mixing up.

ChIMES params.txt / A-matrix / b-vector convention: energies in kcal/mol,
forces in kcal/mol/Angstrom. DFT codes (VASP, QE) natively report eV and
eV/Angstrom; some upstream ChIMES tooling (e.g. contrib/vasp2xyzf.py)
instead targets hartree/bohr for forces -- both conventions are provided
here, and a converter must be explicit about which one it's writing.
"""

from __future__ import annotations

EV_TO_KCAL_PER_MOL = 23.0605
KCAL_PER_MOL_TO_EV = 1.0 / EV_TO_KCAL_PER_MOL

# eV/Angstrom <-> hartree/bohr for forces (product of the Hartree-eV and
# Bohr-Angstrom conversions: 27.2114 eV/Hartree / 0.529177 Angstrom/Bohr).
EV_PER_ANG_TO_HARTREE_PER_BOHR = 1.0 / 51.4221
HARTREE_PER_BOHR_TO_EV_PER_ANG = 51.4221

# ChIMES-internal stress units -> GPa (from chimes_calculator's serial
# interface example: stress * 6.9479 = GPa).
CHIMES_STRESS_TO_GPA = 6.9479


def ev_to_kcal_per_mol(x):
    return x * EV_TO_KCAL_PER_MOL


def kcal_per_mol_to_ev(x):
    return x * KCAL_PER_MOL_TO_EV


def ev_per_ang_to_hartree_per_bohr(x):
    return x * EV_PER_ANG_TO_HARTREE_PER_BOHR


def hartree_per_bohr_to_ev_per_ang(x):
    return x * HARTREE_PER_BOHR_TO_EV_PER_ANG
