"""Generate unlabeled structures for QE labeling when open data is missing,
too small, or at the wrong level of theory.

Seeds are crystal prototypes (`ase.build.bulk(..., orthorhombic=True)`) or
existing structures (`.xyzf` / any ASE-readable file, e.g. a `data-fetch
--label-policy relabel` pool). Each seed is repeated into a supercell whose
every perpendicular width is at least `min_width_ang`, so the result is
orthorhombic when the seed is (prototypes always are) and an ~8 A ChIMES
outer cutoff needs few periodic layers. Then, per seed:

  - optional random substitution to each requested composition,
  - isotropic volume strains (compression samples the repulsive wall that
    ChIMES needs; expansion samples longer bonds),
  - `n_rattle` Gaussian displacements per strain, at each `rattle_std_ang`.

Output is one packed `structures.xyzf` (zero forces, no energy) plus
`provenance.json` with `label_policy: relabel`, ready for `qe-relabel
--structure-xyzf`. This samples near a crystal; broader coverage (liquids,
defects, finite-temperature disorder) comes from MD, which is a separate
phase.
"""

from __future__ import annotations

import datetime
import json
import math
from pathlib import Path

import numpy as np

from ..data_sources import convert
from ..io import xyzf as xyzf_io

NAME = "data-generate"
SUMMARY = "Generate strained/rattled/substituted supercells (unlabeled) for QE labeling."
QE_FRAME_WARNING = 500
SCHEMA = {
    "type": "object",
    "properties": {
        "prototypes": {"type": ["array", "null"], "items": {"type": "object"}, "description": "ase.build.bulk kwargs, e.g. {\"name\": \"Cu\", \"crystalstructure\": \"fcc\", \"a\": 3.61} or {\"name\": \"Zr\", \"crystalstructure\": \"hcp\", \"a\": 3.23, \"c\": 5.15}."},
        "seeds": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Structure files (.xyzf or ASE-readable); every frame is a seed."},
        "compositions": {"type": ["array", "null"], "items": {"type": "object"}, "description": "Random-substitution targets, e.g. [{\"Cu\": 0.5, \"Zr\": 0.5}]; omit to keep seed chemistry."},
        "min_width_ang": {"type": "number", "default": 8.0, "description": "Minimum perpendicular cell width of each supercell."},
        "max_atoms": {"type": "integer", "default": 128, "description": "Skip seeds whose supercell would exceed this (QE cost grows ~N^3)."},
        "volume_strains": {"type": "array", "items": {"type": "number"}, "default": [-0.08, -0.04, 0.0, 0.04, 0.08], "description": "Fractional volume changes."},
        "rattle_std_ang": {"type": "array", "items": {"type": "number"}, "default": [0.05, 0.15]},
        "n_rattle": {"type": "integer", "default": 2, "description": "Rattled copies per (seed, composition, strain, rattle_std)."},
        "seed": {"type": "integer", "default": 42},
    },
}


def add_arguments(parser) -> None:
    parser.add_argument("--prototypes", type=json.loads, default=None)
    parser.add_argument("--seeds", type=lambda s: [x for x in s.split(",") if x], default=None)
    parser.add_argument("--compositions", type=json.loads, default=None)
    parser.add_argument("--min-width-ang", dest="min_width_ang", type=float, default=8.0)
    parser.add_argument("--max-atoms", dest="max_atoms", type=int, default=128)
    parser.add_argument("--volume-strains", dest="volume_strains", type=lambda s: [float(x) for x in s.split(",")], default=[-0.08, -0.04, 0.0, 0.04, 0.08])
    parser.add_argument("--rattle-std-ang", dest="rattle_std_ang", type=lambda s: [float(x) for x in s.split(",")], default=[0.05, 0.15])
    parser.add_argument("--n-rattle", dest="n_rattle", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)


def _widths(cell) -> np.ndarray:
    cell = np.asarray(cell, dtype=float)
    vol = abs(np.linalg.det(cell))
    return np.array([vol / np.linalg.norm(np.cross(cell[(k + 1) % 3], cell[(k + 2) % 3])) for k in range(3)])


def supercell(atoms, min_width: float):
    reps = [max(1, math.ceil(min_width / w)) for w in _widths(atoms.cell[:])]
    return atoms.repeat(reps), reps


def substitute(atoms, composition: dict, rng):
    """Randomly assign species to all sites in the given fractions
    (largest-remainder rounding, so counts sum to the site count)."""
    n = len(atoms)
    els = sorted(composition)
    raw = np.array([composition[e] for e in els], dtype=float)
    raw = raw / raw.sum() * n
    counts = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - counts))[: n - counts.sum()]:
        counts[i] += 1
    symbols = np.repeat(els, counts)
    rng.shuffle(symbols)
    out = atoms.copy()
    out.set_chemical_symbols(list(symbols))
    return out


def _load_seeds(args):
    from ase.build import bulk
    from ase.io import read

    seeds = []
    for proto in getattr(args, "prototypes", None) or []:
        kwargs = dict(proto)
        name = kwargs.pop("name")
        kwargs.setdefault("orthorhombic", True)
        seeds.append((f"proto:{name}-{kwargs.get('crystalstructure', 'default')}", bulk(name, **kwargs)))
    for path in getattr(args, "seeds", None) or []:
        path = Path(path)
        if path.suffix == ".xyzf":
            items = [convert.frame_to_atoms(f) for f in xyzf_io.read_xyzf(path)]
        else:
            items = read(str(path), index=":")
        seeds.extend((f"{path.name}#{i}", a) for i, a in enumerate(items))
    return seeds


def run(args) -> dict:
    seeds = _load_seeds(args)
    if not seeds:
        raise ValueError("data-generate needs --prototypes and/or --seeds")
    rng = np.random.default_rng(getattr(args, "seed", 42))
    min_width = getattr(args, "min_width_ang", 8.0)
    max_atoms = getattr(args, "max_atoms", 128)
    strains = getattr(args, "volume_strains", None) or [0.0]
    stds = getattr(args, "rattle_std_ang", None) or [0.05]
    n_rattle = getattr(args, "n_rattle", 2)
    compositions = getattr(args, "compositions", None) or [None]

    frames, ids, skipped = [], [], []
    for label, atoms in seeds:
        big, reps = supercell(atoms, min_width)
        if len(big) > max_atoms:
            skipped.append({"seed": label, "natoms": len(big), "reason": f"supercell {reps} exceeds max_atoms"})
            continue
        for comp in compositions:
            base = substitute(big, comp, rng) if comp else big
            ctag = "" if not comp else "+" + "".join(f"{e}{comp[e]:g}" for e in sorted(comp))
            for strain in strains:
                strained = base.copy()
                strained.set_cell(base.cell[:] * (1 + strain) ** (1 / 3), scale_atoms=True)
                for std in stds:
                    for r in range(n_rattle):
                        a = strained.copy()
                        a.positions += rng.normal(scale=std, size=a.positions.shape)
                        frames.append(convert.to_frame(a.get_chemical_symbols(), a.cell[:], a.positions))
                        ids.append(f"{label}{ctag}|strain{strain:+g}|rattle{std:g}#{r}")

    if not frames:
        raise ValueError(f"no structures generated (skipped: {skipped})")

    out = Path(getattr(args, "output_dir", None) or ".")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "structures.xyzf"
    xyzf_io.write_xyzf(frames, path)
    elements = sorted({s for f in frames for s in f.symbols})
    prov = {
        "source": "data-generate",
        "source_metadata": {"name": "generated structures", "settings": {k: getattr(args, k, None) for k in SCHEMA["properties"]}},
        "level_of_theory": None,
        "label_policy": "relabel",
        "units": "unlabeled (structures only)",
        "elements": elements,
        "n_frames": len(frames),
        "frame_ids": ids,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    prov_path = out / "provenance.json"
    prov_path.write_text(json.dumps(prov, indent=1))

    natoms = [f.natoms for f in frames]
    warnings = []
    if len(frames) > QE_FRAME_WARNING:
        warnings.append(f"{len(frames)} structures = {len(frames)} QE jobs and per-frame directories; consider fewer strains/rattles or dataset-select fps first")
    if skipped:
        warnings.append(f"{len(skipped)} seeds skipped for size; raise max_atoms or lower min_width_ang")
    return {
        "structures_xyzf": str(path),
        "provenance": str(prov_path),
        "n_frames": len(frames),
        "elements": elements,
        "natoms_range": [min(natoms), max(natoms)],
        "total_atoms": int(sum(natoms)),
        "n_non_orthorhombic": sum(1 for f in frames if f.non_ortho),
        "skipped": skipped,
        "warnings": warnings,
    }
