"""Search open DFT datasets for a chemical system, without downloading any
configurations -- only the cached ColabFit catalog (data_sources/colabfit.py)
is read.

A dataset is relevant to target elements T when its element set D overlaps T:
  - `exact`     D == T      system-specific data, usually the best fit
  - `subsystem` D < T       e.g. pure-Cu data for a Cu-Zr model
  - `superset`  D > T       general databases (MatPES, OMat24, ...); only the
                            rows whose elements are within T get fetched
  - `overlap`   otherwise   may still contain within-T configurations; only
                            listed with include_overlap
Listing a dataset is not a claim it holds useful rows for T: run `data-fetch
--count-only` on it for the actual number.
"""

from __future__ import annotations

from ..data_sources import colabfit

NAME = "data-search"
SUMMARY = "Search open DFT datasets (ColabFit catalog) for a chemical system; downloads no configurations."
USES_OUTPUT_DIR = False
MATCH_ORDER = {"exact": 0, "subsystem": 1, "superset": 2, "overlap": 3}
# Name-prefix -> what the dataset actually contains. Filtering rows to your
# elements does not change what kind of structure they are (slabs with
# vacuum, adsorbates, MOFs), so the agent needs this to judge relevance.
FAMILY_NOTES = (
    ("OC20", "catalysis: adsorbate-on-slab cells with vacuum; within-system rows are mostly surfaces, rPBE"),
    ("OC22", "catalysis: oxide surfaces with adsorbates, vacuum, PBE+U"),
    ("Open_Catalyst_2025", "catalysis: solvent/adsorbate interfaces, RPBE+D3"),
    ("Open_Direct_Air_Capture", "MOFs with CO2/H2O adsorbates; rarely useful for inorganic bulk"),
    ("OMat24", "bulk inorganic, far-from-equilibrium (rattled, high-T AIMD); PBE(+U), MP-compatible VASP settings"),
    ("sAlex", "Alexandria structures subsampled for OMat24-compatible training; PBE(+U)"),
    ("MatPES", "bulk inorganic, near- and off-equilibrium PES sampling, curated for MLIPs; 2025.2 supersedes 2025.1"),
    ("MP-ALOE", "active-learned off-equilibrium bulk, r2SCAN"),
    ("Alexandria_geometry_optimization_paths", "relaxation trajectories: near-equilibrium, many near-duplicate frames"),
    ("Massive_Atomic_Diversity", "MAD: deliberately diverse (clusters, surfaces, rattled bulk), PBEsol/r2SCAN"),
    ("HEA25", "high-entropy transition-metal alloys (bulk, MD + rattled), PBEsol"),
    ("UNEP", "metallic alloys for a general-purpose NEP potential (bulk, MD), PBE"),
    ("23-Single-Element-DNPs", "elemental metals/semiconductors, AIMD trajectories"),
    ("mlearn", "small elemental benchmark sets (Ong group), PBE"),
)
LARGE_DATASET = 5_000_000
SCHEMA = {
    "type": "object",
    "required": ["elements"],
    "properties": {
        "elements": {"type": "array", "items": {"type": "string"}, "description": "Target chemical system, e.g. [\"Cu\", \"Zr\"]."},
        "methods": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Keep datasets containing any of these methods (exact ColabFit labels, e.g. \"DFT-PBE\"). Omit to list all and see what exists."},
        "require_forces": {"type": "boolean", "default": True},
        "periodic_only": {"type": "boolean", "default": True, "description": "Only datasets with 3D-periodic configurations."},
        "include_overlap": {"type": "boolean", "default": False},
        "limit": {"type": "integer", "default": 25},
        "refresh": {"type": "boolean", "default": False, "description": "Refetch the catalog (cached for 30 days)."},
    },
}


def add_arguments(parser) -> None:
    parser.add_argument("--elements", type=lambda s: [x.strip() for x in s.split(",") if x.strip()], default=None)
    parser.add_argument("--methods", type=lambda s: [x.strip() for x in s.split(",") if x.strip()], default=None)
    parser.add_argument("--no-require-forces", dest="require_forces", action="store_false", default=True)
    parser.add_argument("--include-nonperiodic", dest="periodic_only", action="store_false", default=True)
    parser.add_argument("--include-overlap", action="store_true", default=False)
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--refresh", action="store_true", default=False)


def classify(dataset_elements, target) -> str | None:
    d, t = set(dataset_elements or []), set(target)
    if not d & t:
        return None
    if d == t:
        return "exact"
    if d < t:
        return "subsystem"
    if d > t:
        return "superset"
    return "overlap"


def _is_periodic(entry) -> bool:
    dims = entry.get("dimension_types") or []
    if dims:
        return any(list(x) == [1, 1, 1] for x in dims)
    return 3 in (entry.get("nperiodic_dimensions") or [])


def _summarize(entry, match, target) -> dict:
    n = entry.get("nconfigurations") or 0
    notes = []
    methods = entry.get("methods") or []
    name = entry.get("name") or ""
    notes.extend(note for prefix, note in FAMILY_NOTES if name.startswith(prefix))
    if any(tag in name.lower() for tag in ("_val", "_test", "validation")):
        notes.append("validation/test split of a larger dataset; keep it out of training if you also use the train split")
    if len(methods) > 1:
        notes.append("mixed levels of theory inside this dataset: fetch one method at a time")
    if match in ("superset", "overlap"):
        notes.append("general dataset: row count for your system unknown until `data-fetch --count-only`")
    if n > LARGE_DATASET:
        notes.append(f"very large ({n:,} configurations): scanning it is slow; use --max-scan-files")
    if not entry.get("energy_count"):
        notes.append("no energies: forces-only fitting")
    bad = colabfit.known_unreadable().get(entry["repo_id"])
    if bad:
        notes.insert(0, f"UNREADABLE: {len(bad)} data file(s) failed to parse on a previous fetch (corrupt at the source)")
    return {
        "source": f"colabfit:{entry['repo_id']}",
        "name": entry.get("name"),
        "match": match,
        "n_extra_elements": len(set(entry.get("elements") or []) - set(target)),
        "elements": entry.get("elements"),
        "nconfigurations": n,
        "nsites": entry.get("nsites"),
        "methods": methods,
        "software": entry.get("software"),
        "has_stress": bool(entry.get("cauchy_stress_count")),
        "license": entry.get("license"),
        "doi": entry.get("doi"),
        "links": entry.get("links"),
        "year": entry.get("publication_year"),
        "description": (entry.get("description") or "")[:300],
        "notes": notes,
    }


def run(args) -> dict:
    target = getattr(args, "elements", None)
    if not target:
        raise ValueError("data-search requires --elements")
    methods = set(getattr(args, "methods", None) or [])
    catalog = colabfit.load_catalog(refresh=bool(getattr(args, "refresh", False)))
    usable = [e for e in catalog if "catalog_error" not in e]

    hits = []
    for entry in usable:
        match = classify(entry.get("elements"), target)
        if match is None or (match == "overlap" and not getattr(args, "include_overlap", False)):
            continue
        if getattr(args, "require_forces", True) and not entry.get("atomic_forces_count"):
            continue
        if getattr(args, "periodic_only", True) and not _is_periodic(entry):
            continue
        if methods and not methods & set(entry.get("methods") or []):
            continue
        hits.append((match, entry))

    # Fewer foreign elements = more likely on-topic (a Cu-Zr-Al set beats an
    # 89-element database); then larger first.
    tset = set(target)
    hits.sort(key=lambda h: (MATCH_ORDER[h[0]], len(set(h[1].get("elements") or []) - tset),
                             -(h[1].get("nconfigurations") or 0)))
    limit = getattr(args, "limit", 25) or 25
    methods_available = sorted({m for _, e in hits for m in (e.get("methods") or [])})
    return {
        "elements": sorted(target),
        "n_matches": len(hits),
        "results": [_summarize(e, m, target) for m, e in hits[:limit]],
        "methods_available": methods_available,
        "catalog": {"source": "ColabFit on Hugging Face", "datasets": len(catalog), "unreadable": len(catalog) - len(usable)},
    }
