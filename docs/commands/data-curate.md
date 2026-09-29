# `chimes-agent data-curate`

**Status: implemented.** Turns fetched or labeled pools into a base
training set: merge (only when provenance allows), filter, measure what
ChIMES fitting depends on, subsample, split, and write `data_manifest.json`.
That manifest is the handoff to model building.

## Usage

```bash
chimes-agent data-curate --frames study/01_data/fetch_matpes/pool.xyzf \
  --elements Cu,Zr --output-dir study/01_data/curate

# structure pool before QE: drop bad geometry, pick a diverse 300
chimes-agent data-curate --frames gen/structures.xyzf,fetch_omat/pool.xyzf --elements Cu,Zr \
  --target-size 300 --no-holdout --output-dir study/01_data/curate_structures
```

## Merging rules

Each pool's `provenance.json` (from `data-fetch`, `data-generate` or
`qe-relabel --collect`) is read. Labeled pools are refused when they come
from different datasets or levels of theory. Two "DFT-PBE" datasets can still
use different codes, pseudopotentials, cutoffs or smearing, which shifts
absolute energies. QE batches carry a settings hash, so batches labeled with
identical settings merge. Structure-only pools can always be merged.
`--allow-mixed-theory` overrides this after you have checked the settings match.

## Filters (each removal is recorded with its reason)

| Reason | Default | Why |
|---|---|---|
| `elements_outside_target` | on | |
| `duplicate_of:<id>` | on (`--no-dedupe`) | identical geometry to 1e-3 Å |
| `isolated_atom` | on | no neighbour, not even its own periodic image, within the analysis radius: single-atom reference calculations in large boxes. Periodic images count as neighbours, so a 1-atom bulk crystal is kept (an early version dropped them) |
| `min_distance` | < 0.5 Å | unphysical overlap |
| `low_density` | V/atom > 3× pool median (`--max-volume-ratio`, `--keep-vacuum`) | clusters/molecules in vacuum boxes, common in general databases, wrong for a bulk model |
| `max_force` | > 50 eV/Å | broken or extreme labels |
| `energy_outlier` | > 8 scaled MADs **and** > 1 eV/atom from the median residual | broken labels; the absolute floor protects legitimately strained frames in tight datasets |
| `too_few_atoms` / `too_many_atoms` | off | `--natoms-min/max` |
| `non_orthorhombic` | off | `--require-orthorhombic` |

The energy check first fits one reference energy per element by least
squares and compares per-atom residuals, so compositions are comparable.

## What it measures

- **Per element pair** (`pairs`): frames containing the pair, minimum and
  1st-percentile distance, and the number of distances within 1.2× the
  minimum. The minimum bounds `s_minim`; the short-range count says whether
  the repulsive wall is sampled. A pair with no or little data is flagged.
- **Thinnest perpendicular cell width**, and `fit_hints.nlayers_required`:
  the `N_LAYERS` chimes_lsq needs for 4, 6 and 8 Å outer cutoffs
  (`s_maxim ≤ (2·N_LAYERS+1)·width/2`).
- Force-magnitude percentiles and energy-residual spread.

## Subsample and split

`--target-size N` does farthest-point sampling (composition + per-atom
energy descriptor) before splitting. `--holdout-fraction` (default 0.2) does a
composition-stratified split; `--no-holdout` skips it.

## Output files

`curated.xyzf`, `train.xyzf`, `holdout.xyzf`, `curation_report.json`
(every removed frame with pool, index, id and reason; kept ids; reference
energies), and `data_manifest.json`:

```json
{
  "elements": ["Cu", "Zr"], "labeled": true,
  "level_of_theory": ["DFT-PBE (VASP 6.4.x)"],
  "units": {"energy": "kcal/mol", "forces": "hartree/bohr", "positions": "angstrom"},
  "curated_xyzf": "...", "train_xyzf": "...", "holdout_xyzf": "...",
  "summary": {"n_frames": 158, "compositions": {"Cu-Zr": 77, "Cu": 43, "Zr": 38},
              "thinnest_cell_width_ang": 2.01, "...": "..."},
  "pairs": {"Cu-Zr": {"n_frames": 72, "min_distance": 2.28, "n_within_1.2x_min": 230, "...": "..."}},
  "fit_hints": {"s_minim_upper_bound": {"Cu-Zr": 2.28},
                "nlayers_required": {"4.0": 2, "6.0": 3, "8.0": 4},
                "fitener": true, "fitstrs": false},
  "warnings": ["thinnest cell width 2.01 A: an 8 A outer cutoff needs N_LAYERS >= 4 ..."],
  "sources": [{"source": "colabfit:colabfit/MatPES-PBE-2025.2", "license": "BSD-3-Clause", "doi": "..."}]
}
```

Stress is not carried (`fitstrs: false`): ColabFit stress units and
conventions vary by dataset.
