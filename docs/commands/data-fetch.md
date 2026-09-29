# `chimes-agent data-fetch`

**Status: implemented.** Pulls configurations for a chemical system from one
source into a single packed `pool.xyzf` in ChIMES units (kcal/mol,
hartree/bohr) plus `provenance.json`.

## Usage

```bash
# how many Cu/Zr-only frames does MatPES hold, per method? (reads 2 columns)
chimes-agent data-fetch --source colabfit:colabfit/MatPES-PBE-2025.2 --elements Cu,Zr --count-only

# fetch them
chimes-agent data-fetch --source colabfit:colabfit/MatPES-PBE-2025.2 --elements Cu,Zr \
  --all-frames --output-dir study/01_data/fetch_matpes

# structures only, to relabel with QE
chimes-agent data-fetch --source colabfit:colabfit/OMat24_train_rattled_500 --elements Cu,Zr \
  --method DFT-PBE+U --max-frames 2000 --label-policy relabel --output-dir study/01_data/fetch_omat

# local DFT output (anything ASE reads with energies/forces, or .xyzf)
chimes-agent data-fetch --source runs/OUTCAR_1,runs/OUTCAR_2 --elements Cu,Zr \
  --level-of-theory "VASP PBE 520 eV, PAW_PBE 54" --output-dir study/01_data/fetch_local
```

## Behaviour

- **Composition**: `--composition-rule subset` (default) keeps
  configurations whose elements are all within the target, so pure-Cu frames
  count for a Cu-Zr set; `exact` keeps only configurations containing every
  target element.
- **One level of theory per call.** If matching rows span several methods,
  the stage stops and asks for `--method`.
- **Sampling**: `--max-frames` (default 5000) draws a seeded uniform random
  sample from *all* matches, so a capped fetch is representative, not the
  first N rows. `--all-frames` removes the cap.
- **Reading ColabFit** happens in two passes over HTTP range requests: the
  first reads only the `elements`/`method` columns, the second reads
  coordinates for the sampled rows only.
- **Cells** are rotated to the lower-triangular standard form chimes_lsq and
  LAMMPS use (positions and forces rotated with them). A pool that contains
  any triclinic frame is written entirely in `NON_ORTHO` form (chimes_lsq
  segfaults on mixed headers; see `docs/concepts/units_and_conventions.md`);
  an all-orthorhombic pool is written as `Lx Ly Lz`.
- Non-3D-periodic rows and rows missing energy/forces are dropped (counted in
  `dropped`).
- A data file that cannot be parsed is skipped, reported in
  `unreadable_files`, and remembered, so `data-search` flags that dataset
  next time. Some ColabFit files are corrupt at the source: the downloaded bytes
  match the Hub's checksum, yet the parquet pages do not decode.
  `UNEP_v1_2023_train` is one.

## Flags

- `--source` (required): `colabfit:<repo_id>` or path(s), comma-separated
- `--elements A,B` (required); `--composition-rule subset|exact`
- `--method LABEL`; `--max-frames N` / `--all-frames`; `--seed`
- `--count-only`; `--max-scan-files N` (scan only the first N parquet files of
  a very large dataset)
- `--label-policy source|relabel`
- `--level-of-theory TEXT` (local sources; recorded in provenance, and needed
  to merge with other pools)

## Output

`pool.xyzf`, `provenance.json` (source, license/DOI/authors, level of theory,
label policy, units, filters, per-frame ids aligned with frame order), and:

```json
{"pool_xyzf": "...", "n_frames": 169, "n_matching": 169,
 "level_of_theory": {"methods": {"DFT-PBE": 169}, "software": {"VASP 6.4.x": 169}},
 "compositions": {"Cu-Zr": 78, "Cu": 52, "Zr": 39},
 "n_non_orthorhombic": 166, "natoms_range": [1, 34], "dropped": {}, "license": "BSD-3-Clause"}
```

Respect each dataset's license and cite its DOI; both are in
`provenance.json` and carried into `data_manifest.json`.
