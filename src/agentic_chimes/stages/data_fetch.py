"""Pull configurations for a chemical system from one source into a single
packed `.xyzf` (ChIMES units: kcal/mol, hartree/bohr) plus `provenance.json`.

Sources:
  - `colabfit:<repo_id>` -- any ColabFit mirror found by `data-search`
  - a local file or list of files ASE can read with energies/forces (extxyz,
    VASP OUTCAR/vasprun.xml, QE output, ...), or an existing `.xyzf`

One call = one source = one level of theory. Mixing DFT methods, codes or
pseudopotentials in one ChIMES fit mixes incompatible energy references, so
this stage refuses a ColabFit selection spanning several methods unless
`method` picks one; combining pools is `data-curate`'s job and it checks
provenance before merging.

`label_policy: relabel` keeps only structures (energy/forces dropped), for
feeding `qe-relabel` when the curated set must match the level of theory
that active learning will later add.

Configurations are selected by composition: `subset` (default) keeps
configurations whose elements are all within `elements` (so pure-Cu frames
count toward a Cu-Zr set); `exact` keeps only configurations containing
every target element. Frames are sampled uniformly at random (seeded) from
all matches, not taken from the front of the file, so a capped fetch is
representative.
"""

from __future__ import annotations

import datetime
import json
from collections import Counter
from pathlib import Path

from ..data_sources import colabfit, convert
from ..io import xyzf as xyzf_io
from ..io import atomic
from ..io import fs

NAME = "data-fetch"
SUMMARY = "Fetch configurations for a chemical system from ColabFit or local DFT files into one .xyzf + provenance."
SCHEMA = {
    "type": "object",
    "required": ["source", "elements"],
    "properties": {
        "source": {"description": "`colabfit:<repo_id>`, or a path / list of paths to local DFT output or .xyzf.", "type": ["string", "array"]},
        "elements": {"type": "array", "items": {"type": "string"}},
        "composition_rule": {"type": "string", "enum": ["subset", "exact"], "default": "subset"},
        "method": {"type": ["string", "null"], "description": "ColabFit method label to keep (e.g. \"DFT-PBE\"); required when matches span several."},
        "max_frames": {"type": ["integer", "null"], "default": 5000, "description": "Uniform random sample cap; null = everything."},
        "count_only": {"type": "boolean", "default": False, "description": "Report matching counts per method without downloading coordinates."},
        "max_scan_files": {"type": ["integer", "null"], "description": "Only scan the first N parquet files of a very large dataset (count and sample then cover only those)."},
        "label_policy": {"type": "string", "enum": ["source", "relabel"], "default": "source"},
        "level_of_theory": {"type": ["string", "null"], "description": "Local sources only: e.g. \"QE PBE PAW, ecutwfc 60 Ry\". Recorded in provenance."},
        "seed": {"type": "integer", "default": 42},
    },
}


def _list_arg(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def add_arguments(parser) -> None:
    parser.add_argument("--source", default=None, help="colabfit:<repo_id> or local path(s), comma-separated.")
    parser.add_argument("--elements", type=_list_arg, default=None)
    parser.add_argument("--composition-rule", dest="composition_rule", choices=["subset", "exact"], default="subset")
    parser.add_argument("--method", default=None)
    parser.add_argument("--max-frames", dest="max_frames", type=int, default=5000)
    parser.add_argument("--all-frames", dest="max_frames", action="store_const", const=None)
    parser.add_argument("--count-only", dest="count_only", action="store_true", default=False)
    parser.add_argument("--max-scan-files", dest="max_scan_files", type=int, default=None)
    parser.add_argument("--label-policy", dest="label_policy", choices=["source", "relabel"], default="source")
    parser.add_argument("--level-of-theory", dest="level_of_theory", default=None)
    parser.add_argument("--seed", type=int, default=42)


def composition_filter(target, rule):
    t = set(target)

    def keep(elements, _method=None):
        e = set(elements)
        if not e or not e <= t:
            return False
        return rule == "subset" or e == t

    return keep


def _composition_counts(frames) -> dict:
    return dict(Counter("-".join(sorted(set(f.symbols))) for f in frames).most_common())


def _fetch_colabfit(repo_id, args, keep):
    entry = colabfit.catalog_entry(repo_id)
    if "catalog_error" in entry:
        raise RuntimeError(f"cannot read {repo_id}: {entry['catalog_error']}")

    method = getattr(args, "method", None)

    def row_filter(elements, m):
        return keep(elements) and (method is None or m == method)

    files = colabfit.config_files(repo_id)
    if getattr(args, "max_scan_files", None):
        files = files[: args.max_scan_files]
    locations, methods_seen, unreadable = colabfit.scan(repo_id, row_filter, files=files)
    if unreadable and len(unreadable) == len(files):
        raise RuntimeError(f"every scanned file of {repo_id} is unreadable (corrupt at the source; the bytes match the Hub checksum): {sorted(unreadable)}. Pick another dataset.")

    scan_report = {"n_matching": len(locations), "matching_by_method": methods_seen,
                   "files_scanned": len(files), "files_total": len(colabfit.config_files(repo_id)),
                   "unreadable_files": sorted(unreadable)}
    if getattr(args, "count_only", False):
        return None, entry, scan_report
    if not locations:
        raise ValueError(f"no configurations in {repo_id} match elements {sorted(args.elements)} ({args.composition_rule})")
    if method is None and len(methods_seen) > 1:
        raise ValueError(
            f"matches in {repo_id} span several levels of theory {methods_seen}; pass --method to pick one "
            "(one ChIMES fit must use one level of theory)"
        )

    from ase.data import chemical_symbols

    chosen = colabfit.sample(locations, getattr(args, "max_frames", 5000), getattr(args, "seed", 42))
    relabel = getattr(args, "label_policy", "source") == "relabel"
    frames, ids, dropped = [], [], Counter()
    methods, software = Counter(), Counter()
    for row in colabfit.read_rows(repo_id, chosen):
        if not all(row.get("pbc") or []):
            dropped["not_3d_periodic"] += 1
            continue
        if not relabel and (row.get("atomic_forces") is None or row.get("energy") is None):
            dropped["missing_energy_or_forces"] += 1
            continue
        symbols = [chemical_symbols[z] for z in row["atomic_numbers"]]
        stress = None if relabel else row.get("cauchy_stress")
        if stress is not None and row.get("cauchy_stress_volume_normalized"):
            # stored multiplied by the cell volume (a virial): divide it back out
            import numpy as np

            stress = np.asarray(stress, dtype=float) / abs(np.linalg.det(np.asarray(row["cell"], dtype=float)))
        frames.append(convert.to_frame(
            symbols, row["cell"], row["positions"],
            energy_ev=None if relabel else row["energy"],
            forces_ev_ang=None if relabel else row["atomic_forces"],
            stress_ev_ang3=stress, stress_sign="pressure",  # checked below: ColabFit conventions vary by dataset
        ))
        ids.append(row.get("configuration_id"))
        methods[row.get("method")] += 1
        software[row.get("software")] += 1

    stress_check = _settle_stress_sign(frames)
    source_meta = {k: entry.get(k) for k in ("name", "license", "doi", "authors", "links", "publication_year", "description")}
    theory = {"methods": dict(methods), "software": dict(software)}
    source_meta = {**source_meta, "_stress_sign_check": stress_check}
    return (frames, ids, dict(dropped), source_meta, theory), entry, scan_report


def _settle_stress_sign(frames) -> dict:
    """ColabFit stores `cauchy_stress` with whatever sign the source used
    (MatPES: VASP's pressure sign, despite the name). Verify on the data and
    fix it; drop stresses whose sign cannot be verified rather than fit them."""
    if not any(f.stress is not None for f in frames):
        return {"verdict": "no stresses"}
    check = convert.stress_sign_check(frames)
    if check["verdict"] == "flipped":
        for f in frames:
            if f.stress is not None:
                f.stress = [-x for x in f.stress]
        check["action"] = "negated all stresses (stored with the Cauchy sign)"
    elif check["verdict"] == "unverified":
        for f in frames:
            f.stress = None
        check["action"] = ("dropped stresses: sign could not be verified (needs >= 5 frames of one composition "
                           "spanning volumes); fit without stresses or supply them with a known convention")
    else:
        check["action"] = "kept (pressure sign verified)"
    return check


def _fetch_local(paths, args, keep):
    from ase.io import read

    relabel = getattr(args, "label_policy", "source") == "relabel"
    frames, ids, dropped = [], [], Counter()
    for path in paths:
        path = Path(path)
        if path.suffix == ".xyzf":
            items = [(f, None) for f in xyzf_io.read_xyzf(path)]
        else:
            items = [(None, a) for a in read(str(path), index=":")]
        for i, (fr, atoms) in enumerate(items):
            if fr is None:
                if not all(atoms.pbc):
                    dropped["not_3d_periodic"] += 1
                    continue
                energy = forces = stress = None
                if not relabel:
                    try:
                        energy, forces = atoms.get_potential_energy(), atoms.get_forces()
                    except Exception:  # noqa: BLE001 - ASE raises various types when no calculator results exist
                        dropped["missing_energy_or_forces"] += 1
                        continue
                    try:
                        stress = atoms.get_stress(voigt=False)  # ASE: Cauchy sign, eV/A^3
                    except Exception:  # noqa: BLE001 - no stress in the file is normal
                        stress = None
                fr = convert.to_frame(atoms.get_chemical_symbols(), atoms.cell[:], atoms.positions, energy, forces, stress)
            elif relabel:
                fr.energy = None
                fr.forces = [[0.0, 0.0, 0.0]] * fr.natoms
            if not keep(set(fr.symbols)):
                dropped["composition"] += 1
                continue
            frames.append(fr)
            ids.append(f"{path.name}#{i}")

    if getattr(args, "count_only", False):
        return None, {"n_matching": len(frames)}
    max_frames = getattr(args, "max_frames", 5000)
    if max_frames is not None and len(frames) > max_frames:
        import random

        idx = sorted(random.Random(getattr(args, "seed", 42)).sample(range(len(frames)), max_frames))
        frames, ids = [frames[i] for i in idx], [ids[i] for i in idx]
    lot = getattr(args, "level_of_theory", None)
    theory = {"methods": {lot or "unspecified": len(frames)}, "software": {}}
    return (frames, ids, dict(dropped), {"name": ", ".join(str(p) for p in paths)}, theory), {"n_matching": len(frames)}


def run(args) -> dict:
    source = getattr(args, "source", None)
    target = getattr(args, "elements", None)
    if not source or not target:
        raise ValueError("data-fetch requires --source and --elements")
    keep = composition_filter(target, getattr(args, "composition_rule", "subset"))

    if isinstance(source, str) and source.startswith("colabfit:"):
        repo_id = source.split(":", 1)[1]
        payload, _entry, scan_report = _fetch_colabfit(repo_id, args, keep)
    else:
        paths = source if isinstance(source, list) else [p for p in source.split(",") if p]
        payload, scan_report = _fetch_local(paths, args, keep)

    if payload is None:
        return {"source": source, "elements": sorted(target), "count_only": True, **scan_report}

    frames, ids, dropped, source_meta, theory = payload
    stress_check = source_meta.pop("_stress_sign_check", None)
    if stress_check is None:  # local files: ASE's convention is known; still sanity-check it
        stress_check = (convert.stress_sign_check(frames) if any(f.stress is not None for f in frames)
                        else {"verdict": "no stresses"})
    if not frames:
        raise ValueError(f"nothing left to write after filtering (dropped: {dropped})")

    out = Path(getattr(args, "output_dir", None) or ".")
    fs.ensure_dir(out)
    pool = out / "pool.xyzf"
    xyzf_io.write_xyzf(frames, pool)
    label_policy = getattr(args, "label_policy", "source")
    provenance = {
        "source": source,
        "source_metadata": source_meta,
        "level_of_theory": theory,
        "label_policy": label_policy,
        "units": {"energy": "kcal/mol", "forces": "hartree/bohr", "positions": "angstrom"} if label_policy == "source" else "unlabeled (structures only)",
        "elements": sorted(target),
        "composition_rule": getattr(args, "composition_rule", "subset"),
        "scan": scan_report,
        "dropped": dropped,
        "stresses": {"n_frames_with_stress": sum(1 for f in frames if f.stress is not None),
                     "units": "GPa, pressure sign (ChIMES)", "sign_check": stress_check},
        "n_frames": len(frames),
        "frame_ids": ids,
        "fetched_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    prov_path = out / "provenance.json"
    atomic.write_json(prov_path, provenance, indent=1)

    return {
        "pool_xyzf": str(pool),
        "provenance": str(prov_path),
        "n_frames": len(frames),
        "n_matching": scan_report["n_matching"],
        "level_of_theory": theory,
        "label_policy": label_policy,
        "compositions": _composition_counts(frames),
        "n_non_orthorhombic": sum(1 for f in frames if f.non_ortho),
        "natoms_range": [min(f.natoms for f in frames), max(f.natoms for f in frames)],
        "dropped": dropped,
        "n_with_stress": sum(1 for f in frames if f.stress is not None),
        "stress_sign_check": stress_check,
        "license": source_meta.get("license"),
    }
