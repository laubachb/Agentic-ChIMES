"""ColabFit Exchange datasets as mirrored on Hugging Face (`colabfit/*`).

Every mirror shares one schema: `ds.parquet` (one row of dataset-level
metadata: elements, sizes, DFT methods, license, DOI) and `co/*.parquet`
(one row per configuration: cell, positions, pbc, atomic_numbers, energy in
eV, atomic_forces in eV/Angstrom, method, software, configuration_id). That
uniformity is why this is the primary open-data backend: one connector
reaches MatPES, OMat24 subsets, Alexandria relaxation paths and hundreds of
system-specific sets.

The catalog (every `ds.parquet` row) is small and cached locally, so
searching never downloads configurations. Configuration files are read over
plain HTTP range requests, in two passes: the first reads only the `elements`
and `method` columns to find matching rows, the second reads coordinates for
just the rows that were sampled.

Hugging Face rate-limits anonymous `/api` calls per IP, and a lab network
shares one outbound IP. Everything here therefore goes through `resolve/`
download URLs where possible, sends the user's token when one is configured
(`HF_TOKEN` / `huggingface-cli login`), and caches the few API results needed
(the dataset list, each dataset's file list).
"""

from __future__ import annotations

import io
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from ..io import atomic
from ..io import fs

HF_AUTHOR = "colabfit"
CATALOG_COLUMNS = (
    "id", "name", "nconfigurations", "nsites", "nelements", "elements",
    "nperiodic_dimensions", "dimension_types", "energy_count", "atomic_forces_count",
    "cauchy_stress_count", "authors", "description", "license", "links",
    "publication_year", "doi", "methods", "software", "equilibrium",
)
CONFIG_COLUMNS = (
    "configuration_id", "method", "software", "energy", "atomic_forces",
    "cell", "positions", "pbc", "atomic_numbers", "cauchy_stress", "cauchy_stress_volume_normalized",
)


def _cache_dir() -> Path:
    from .. import config

    d = config.CACHE_DIR / "colabfit"
    fs.ensure_dir(d)
    return d


def _auth_headers() -> dict:
    from huggingface_hub import get_token

    token = get_token()
    return {"Authorization": f"Bearer {token}"} if token else {}


def _resolve_url(repo_id: str, rel: str) -> str:
    return f"https://huggingface.co/datasets/{repo_id}/resolve/main/{rel}"


def _rate_limit_hint(exc) -> str:
    if "429" in str(exc):
        return (" Hugging Face is rate-limiting this IP; set HF_TOKEN (or run "
                "`huggingface-cli login`) and retry.")
    return ""


def _jsonable(v):
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return v


def _get(url: str, retries: int = 5):
    import requests

    delay = 2.0
    for attempt in range(retries):
        resp = requests.get(url, headers=_auth_headers(), timeout=120)
        if resp.status_code == 429 and attempt < retries - 1:
            time.sleep(delay)
            delay *= 2
            continue
        resp.raise_for_status()
        return resp
    raise RuntimeError("unreachable")


def _fetch_catalog_entry(repo_id: str):
    import pyarrow.parquet as pq

    try:
        table = pq.read_table(io.BytesIO(_get(_resolve_url(repo_id, "ds.parquet")).content))
    except Exception as exc:  # noqa: BLE001 - one broken mirror must not sink the catalog
        return {"repo_id": repo_id, "catalog_error": str(exc)[:300]}
    if table.num_rows == 0:
        return {"repo_id": repo_id, "catalog_error": "empty ds.parquet"}
    row = table.to_pylist()[0]
    entry = {k: _jsonable(row.get(k)) for k in CATALOG_COLUMNS if k in row}
    entry["repo_id"] = repo_id
    return entry


def load_catalog(refresh: bool = False, max_age_days: float = 30.0, workers: int = 4) -> list:
    """All ColabFit dataset metadata, from the local cache unless stale or
    `refresh`. A refresh keeps entries that already succeeded and fetches only
    new or failed ones, so repeated refreshes converge under rate limiting."""
    path = _cache_dir() / "catalog.json"
    cached = json.loads(path.read_text()) if path.is_file() else {}
    fresh = time.time() - cached.get("fetched_at", 0) < max_age_days * 86400
    if cached and fresh and not refresh:
        return cached["datasets"]

    from huggingface_hub import HfApi

    good = {e["repo_id"]: e for e in cached.get("datasets", []) if "catalog_error" not in e} if fresh else {}
    try:
        repo_ids = sorted(d.id for d in HfApi().list_datasets(author=HF_AUTHOR))
    except Exception as exc:  # noqa: BLE001
        repo_ids = sorted(e["repo_id"] for e in cached.get("datasets", []))
        if not repo_ids:
            raise RuntimeError(f"could not list ColabFit datasets on Hugging Face ({exc}).{_rate_limit_hint(exc)}") from exc
    todo = [r for r in repo_ids if r not in good]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for entry in pool.map(_fetch_catalog_entry, todo):
            good[entry["repo_id"]] = entry
    entries = [good[r] for r in repo_ids]
    atomic.write_json(path, {"fetched_at": time.time(), "datasets": entries})
    return entries


def catalog_entry(repo_id: str) -> dict:
    for entry in load_catalog():
        if entry["repo_id"] == repo_id:
            return entry
    return _fetch_catalog_entry(repo_id)


def config_files(repo_id: str) -> list:
    """`co/*.parquet` paths for a dataset (one cached API call per dataset)."""
    path = _cache_dir() / "files" / (repo_id.replace("/", "__") + ".json")
    if path.is_file():
        return json.loads(path.read_text())
    from huggingface_hub import HfApi

    try:
        files = sorted(f for f in HfApi().list_repo_files(repo_id, repo_type="dataset")
                       if f.startswith("co/") and f.endswith(".parquet"))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"could not list files of {repo_id} ({exc}).{_rate_limit_hint(exc)}") from exc
    fs.ensure_dir(path.parent)
    atomic.write_json(path, files)
    return files


def _open_parquet(repo_id: str, rel: str):
    import fsspec
    import pyarrow.parquet as pq

    fs = fsspec.filesystem("http", client_kwargs={"headers": _auth_headers()}, block_size=8 * 2**20)
    return pq.ParquetFile(fs.open(_resolve_url(repo_id, rel), "rb"))


def _unreadable_path() -> Path:
    return _cache_dir() / "unreadable.json"


def known_unreadable() -> dict:
    """repo_id -> {file: error} for mirror files that failed to parse. Some
    ColabFit parquet files are corrupt at the source (bytes match the Hub's
    sha256, yet pyarrow cannot decode their page headers)."""
    path = _unreadable_path()
    return json.loads(path.read_text()) if path.is_file() else {}


def _record_unreadable(repo_id: str, rel: str, err: str) -> None:
    data = known_unreadable()
    data.setdefault(repo_id, {})[rel] = err[:200]
    atomic.write_json(_unreadable_path(), data, indent=1)


def scan(repo_id: str, row_filter, files=None):
    """Pass 1: locate matching rows using only the `elements`/`method`
    columns. Returns (locations, method_counts, unreadable_files); a location
    is (file, row_group, row_index). Unreadable files are skipped and
    remembered, not fatal."""
    from collections import Counter

    locations, methods_seen, unreadable = [], Counter(), {}
    for rel in files if files is not None else config_files(repo_id):
        try:
            pf = _open_parquet(repo_id, rel)
            names = set(pf.schema_arrow.names)
            cols = [c for c in ("elements", "method") if c in names]
            found = []
            for rg in range(pf.metadata.num_row_groups):
                t = pf.read_row_group(rg, columns=cols)
                elems = t.column("elements").to_pylist()
                meths = t.column("method").to_pylist() if "method" in names else [None] * len(elems)
                found.extend((rg, i, m) for i, (e, m) in enumerate(zip(elems, meths)) if row_filter(e or [], m))
        except OSError as exc:
            unreadable[rel] = str(exc)
            _record_unreadable(repo_id, rel, str(exc))
            continue
        for rg, i, m in found:
            locations.append((rel, rg, i))
            methods_seen[m] += 1
    return locations, dict(methods_seen), unreadable


def sample(locations: list, max_rows, seed: int) -> list:
    if max_rows is None or len(locations) <= max_rows:
        return list(locations)
    return sorted(random.Random(seed).sample(locations, max_rows))


def read_rows(repo_id: str, locations: list):
    """Pass 2: yield full configuration dicts for the given locations."""
    from collections import defaultdict

    by_group = defaultdict(list)
    for rel, rg, i in locations:
        by_group[(rel, rg)].append(i)
    open_files = {}
    for (rel, rg), idxs in sorted(by_group.items()):
        pf = open_files.get(rel) or open_files.setdefault(rel, _open_parquet(repo_id, rel))
        names = set(pf.schema_arrow.names)
        table = pf.read_row_group(rg, columns=[c for c in CONFIG_COLUMNS if c in names]).take(sorted(idxs))
        for row in table.to_pylist():
            yield row
