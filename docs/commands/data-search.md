# `chimes-agent data-search`

**Status: implemented.** Searches open DFT datasets for a chemical system
without downloading any configurations.

The backend is the [ColabFit Exchange](https://materials.colabfit.org) mirror
on Hugging Face (`colabfit/*`, ~500 datasets, one shared schema), which
includes MatPES, OMat24 subsets, Alexandria relaxation paths and many
system-specific sets. The first call builds a catalog from each dataset's
small `ds.parquet` metadata file and caches it for 30 days in
`deps/cache/colabfit/` (override with `AGENTIC_CHIMES_CACHE`).

## Usage

```bash
chimes-agent data-search --elements Cu,Zr                      # what exists, all methods
chimes-agent data-search --elements Cu,Zr --methods DFT-PBE    # one level of theory
```

## How results are classified and ranked

For target elements T and a dataset's element set D:

| match | meaning | example for Cu-Zr |
|---|---|---|
| `exact` | D = T | a Cu-Zr metallic-glass set |
| `subsystem` | D ⊂ T | pure-Cu data |
| `superset` | D ⊃ T | MatPES, OMat24 (only rows within T are fetched) |
| `overlap` | partial | listed only with `--include-overlap` |

Order: `exact`, `subsystem`, then `superset` by fewest foreign elements (a
Cu-Zr-Al set before an 89-element database), then size.

Each result has `notes`, which matter more than the rank: what kind of data
a known dataset family actually holds (catalysis slabs with vacuum, MOFs,
near-duplicate relaxation paths, validation splits), mixed methods inside
one dataset, very large datasets, and datasets whose files a previous fetch
found corrupt at the source.

A `superset` listing does not mean it has useful rows for your system:
`data-fetch --count-only` gives the actual count.

## Flags

- `--elements A,B` (required)
- `--methods M1,M2` — exact ColabFit labels, e.g. `DFT-PBE`, `DFT-PBE+U`,
  `DFT-R2SCAN`; the output's `methods_available` lists what exists
- `--no-require-forces`, `--include-nonperiodic`, `--include-overlap`
- `--limit N` (default 25), `--refresh` (refetch the catalog)

## Rate limits

Hugging Face rate-limits anonymous requests per IP address, and a lab network
shares one. A refresh keeps entries already fetched and retries only failures,
but if you see HTTP 429 set `HF_TOKEN` (any free account; read access is
enough) or run `huggingface-cli login`.
