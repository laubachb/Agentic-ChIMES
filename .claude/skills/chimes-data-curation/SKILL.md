---
name: chimes-data-curation
description: Build the base (pre-active-learning) training set for a ChIMES model from a user's description - find open DFT data for the chemical system, judge relevance and level-of-theory compatibility, fetch, generate structures for QE labeling when open data falls short, filter and analyze, and write data_manifest.json. Use when the user asks for training data, a dataset for elements X-Y, "grab data from Materials Project / OMat / MatPES / ColabFit", or when a study reaches its data phase. The chimes-data-curator subagent follows this playbook.
---

# Data selection and curation (study phase 2)

Goal: one **base dataset** that (1) has a single, reproducible level of
theory, (2) samples every element pair across the distances and conditions
the model will see, (3) is sized for fitting, and (4) comes with a holdout
split and `data_manifest.json`. Work in `<study>/01_data/`.

Stages, in the order you usually need them (`--describe` for flags):
`data-search` → `data-fetch --count-only` → `data-fetch` / `data-generate`
→ `data-curate` → (QE route) `qe-relabel` submit, collect → `data-curate`.

## 1. Turn the request into a data spec

Write `01_data/DATA_PLAN.md` before fetching anything:

- **Elements** and **what the model is for**: phases (crystal structures,
  liquid, amorphous), temperature/pressure range, defects, surfaces. A bulk
  model wants bulk data; surfaces and clusters are a different problem.
- **Label target.** Active learning (phase 5) will add frames labeled by
  some QM engine; the base set must share its level of theory. Energies from
  different codes, pseudopotentials or cutoffs have different references and
  cannot share a fit. Two defensible choices:
  - *keep open-data labels*: fastest, but only valid if the AL engine can
    reproduce that dataset's settings (e.g. VASP with MP-compatible INCAR and
    POTCARs for MatPES/OMat24);
  - *relabel with QE* (`qe-relabel`): consistent end to end with the built-in
    labeler; costs one QE job per frame.
  If the user did not say, this is a decision to put in front of them, with
  the frame counts and rough cost for each option.
- **Size.** A base set of a few hundred to a couple of thousand frames is
  typical. Diversity matters more than count; near-duplicate relaxation
  frames add little.

## 2. Survey

- `data-search --elements A,B` first without `--methods`, to see which
  levels of theory exist; then filter. Read every `notes` entry: it says
  what the dataset really contains (catalysis slabs, MOFs, relaxation paths,
  validation splits) and flags datasets whose files are corrupt at the source.
- Ranking puts system-specific (`exact`, `subsystem`) data first, then
  general databases by how few foreign elements they have. General
  databases (MatPES, OMat24, Alexandria) hold an unknown number of your
  frames: run `data-fetch --count-only` on the few plausible ones. It reads
  two columns only and is cheap except for >5M-row sets (use
  `--max-scan-files`).
- Relevant structure-only sources are also worth noting: structures from any
  dataset can be mixed freely if you relabel them.

## 3. Choose a strategy (record it in DATA_PLAN.md)

- **A. One open dataset, its own labels.** Only when it covers the spec and
  the label target allows it. Never merge labels from two datasets;
  `data-curate` refuses, and `--allow-mixed-theory` is for when you have
  verified the DFT settings are identical, not a way around the error.
- **B. Open-data structures + QE relabel** (`data-fetch --label-policy
  relabel`, any number of datasets).
- **C. Generated structures + QE** (`data-generate`): prototypes or seeds →
  supercells ≥ 8 Å wide (orthorhombic, few periodic layers), random alloy
  compositions, volume strains, rattles. Use when open data is scarce. For
  example, MatPES holds only ~150 Cu/Zr-only frames, all tiny triclinic cells.
- B and C combine naturally (curate the structure pools together, then one
  QE batch with one settings set).

## 4. Execute

- Fetches and generation are local and need no approval; keep them capped
  (`--max-frames`, `max_atoms`) and write one packed `.xyzf` per source.
- `data-curate` on structure pools before QE too: it removes duplicates,
  vacuum clusters and too-close atoms, and `--target-size` farthest-point
  sampling picks a diverse subset so you pay for fewer QE jobs.
- **QE labeling is an HPC submission: dry-run only, then stop and ask.**
  Use `--kspacing` (≈0.2-0.3 /Å for metals), never one fixed `--kpoints` grid
  for cells of different sizes, and keep one settings set for the whole study
  (later AL rounds must reuse it). Load `chimes-hpc-jobs` for the rest.
- After `qe-relabel --collect`, run `data-curate` on the `labeled.xyzf`
  (its `provenance.json` carries the settings), with `--holdout-fraction`.

## 5. Review before handing off

Read the `data-curate` result, not just its exit status:

- `removed_by_reason`: large removals mean something about the source you
  should understand (e.g. `low_density` = gas-phase frames in a general
  database). `curation_report.json` lists every removed frame and why.
- `pairs`: every pair needs frames, and `n_within_1.2x_min` short-range
  distances for its repulsive wall; a starved pair is the most common reason
  a ChIMES model is unstable in MD.
- `summary.thinnest_cell_width_ang` and `fit_hints.nlayers_required`: tiny
  cells force large `N_LAYERS` in `fm_setup.in`.
- `warnings`: triclinic frames are fine for `chimes_lsq`, but `auto-build`'s
  cutoff derivation and `lammps-run` data files are orthorhombic-only.
- Coverage vs. the spec: conditions in DATA_PLAN.md that no frame samples
  (liquid, high pressure) are gaps for the MD / active-learning phases;
  say so.

Finish DATA_PLAN.md with what was done, the numbers, open gaps, and the path
to `data_manifest.json`, which is what the model-building phase reads
(`train_xyzf`, `holdout_xyzf`, `pairs.*.min_distance` for `s_minim`,
`fit_hints`).

Reference: `docs/commands/data-search.md`, `data-fetch.md`, `data-generate.md`,
`data-curate.md`.
