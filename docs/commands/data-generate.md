# `chimes-agent data-generate`

**Status: implemented.** Generates unlabeled structures for QE labeling
when open data is missing, too small, or at the wrong level of theory.

## Usage

```bash
chimes-agent data-generate \
  --prototypes '[{"name":"Cu","crystalstructure":"fcc","a":3.61},
                 {"name":"Zr","crystalstructure":"hcp","a":3.23,"c":5.15}]' \
  --compositions '[{"Cu":0.5,"Zr":0.5},{"Cu":0.64,"Zr":0.36}]' \
  --output-dir study/01_data/generate

# or seed from existing structures (e.g. a data-fetch --label-policy relabel pool)
chimes-agent data-generate --seeds study/01_data/fetch_matpes/pool.xyzf --output-dir ...
```

Then `qe-relabel --structure-xyzf study/01_data/generate/structures.xyzf
--kspacing 0.25 ...` (curate or subsample first if the count is large).

## What it builds

For each seed, a supercell whose every perpendicular width is at least
`--min-width-ang` (default 8 Å). Prototypes are built orthorhombic, so the
result works with every stage, and an ~8 Å ChIMES outer cutoff needs no
extra periodic layers. Seeds larger than `--max-atoms` (default 128; QE cost
grows roughly as N³) are skipped and reported. Then, for each seed:

1. random substitution to each `--compositions` entry (exact counts by
   largest-remainder rounding),
2. each `--volume-strains` value (default ±4 %, ±8 %, 0): compression
   samples the short-range repulsion ChIMES needs,
3. `--n-rattle` copies at each `--rattle-std-ang` (default 0.05 and 0.15 Å).

The defaults give 20 structures per seed and composition. This samples
around a crystal. Liquids, defects and finite-temperature disorder come from
MD, which is a later phase.

## Output

`structures.xyzf` (zero forces, no energy), `provenance.json` with
`label_policy: relabel` and a descriptive id per frame, and a summary
(`n_frames`, `natoms_range`, `total_atoms`, `skipped`, `warnings`). A
warning is raised above 500 structures, since each one is a QE job and a
directory.
