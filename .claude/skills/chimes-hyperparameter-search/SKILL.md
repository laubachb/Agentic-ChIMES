---
name: chimes-hyperparameter-search
description: Choose ChIMES hyperparameters for a curated dataset - inner/outer cutoffs per body order, Morse lambdas, 2-/3-/4-body polynomial orders - by data analysis plus a staged, noise-aware holdout search. Use when a study reaches its fitting phase, when the user asks which cutoffs/orders/lambdas to use, asks to tune or sweep a ChIMES model, or has a data_manifest.json and wants a model. The chimes-hyperparameter-tuner subagent follows this playbook.
---

# Hyperparameter selection (study phase 3)

Goal: settings for `fm_setup.in` (per-pair inner cutoffs and Morse λ, 2-/3-/4-body
outer cutoffs, polynomial orders, N_LAYERS) that generalize to the holdout
set, at the lowest MD cost the data justifies, with the reasoning written
down. Work in `<study>/02_fit/`. Input: `01_data/curate/data_manifest.json`.

## 1. Analyze the data (seconds, local)

`chimes-agent hyper-analyze --data-manifest <manifest> --output-dir 02_fit/analysis`

Read, in this order:

- `notes`: an unsampled pair or sparse short-range data is a *data* problem.
  Stop and send it back to the data phase rather than tuning around it.
- Per pair `min_distance` → `suggested.s_minim` (min − 0.02 Å). These are
  not searched: higher discards data, lower extrapolates.
- `rdf_peaks[0]` → `suggested.morse_lambda`. Sanity-check against known bond
  lengths (Cu-Cu 2.55, Zr-Zr ~3.2 Å); a λ far off usually means a phase
  mix-up in the data.
- `global.first_shell_end` / `second_shell_end` and the cutoff
  `candidates`, and `nlayers_required` per candidate (tiny open-database
  cells need N_LAYERS 3-4, which makes many-body builds slow).
- `n_force_equations` + `n_energy_equations`: the data budget. A 12/7/3
  binary basis is ~1,000+ coefficients; with ~1,500 equations that is not
  identifiable. Restrict orders before searching, not after.

## 2. Plan the search

Defaults (`hyper-search --describe`) are a sensible start. Adjust from the
analysis and from the user's goal:

- **Stages**: `2b,3b,4b,lambda,refine`. Drop `4b` (or keep
  `--four-body auto`) when equations < ~3× a 4-body basis. Liquids and
  alloys usually need 3-body; 4-body only with plenty of data.
- **Cutoffs**: many-body cutoffs at the first-shell minimum carry almost no
  signal in ChIMES (cubic smoothing multiplies a (1 − r/r_c)³ factor per
  cluster distance). The candidates reach toward the second shell for that
  reason. Longer means costlier MD (pairs ∝ r³, triplets ∝ r⁶, quartets
  ∝ r⁹): if the user has an MD cost budget, cap the candidates with
  `--s-maxim-3b` / `--s-maxim-4b`.
- **Solver**: see the *Solver* section below; do not change it casually.
  Comparing bases is only fair when each gets appropriate regularization.
- **Objective**: `auto` scores force + energy when energies are fitted.
  Use `force` if the user only cares about dynamics.
- **Size and cost**: fits × build time. Estimate from one probe fit. Anything
  beyond a few minutes runs as a Slurm job (`--machine`), never on a login
  node: `--dry-run` first, show the user the job, submit on approval.
  `--max-fit-seconds` caps runaway many-body builds.

### 4-body terms and exclusions

- 4-body builds scale steeply (quartets ∝ r⁹, times N_LAYERS images on
  small cells), so the user wants 4-body sweeps kept light. Defaults are
  orders 2-3 and two cutoffs (first-shell end and midway, never above the
  3-body cutoff), one solver, and 600 s per fit. Do not widen them without
  saying what it costs. The stage only runs when 3-body improved the score by
  ≥ `min_gain` (`--four-body auto`).
- With raw `lassolars`, 4-body terms usually change nothing (the columns are
  tiny). `--four-body-solvers lassolars,blocklasso` tests whether that hides
  real signal, but doubles the builds. On Cu-Zr it did not: `blocklasso` was
  worse everywhere. No 4-body gain means the data does not support 4-body
  terms; say so and move on.
- The `exclude` stage removes each 3-/4-body cluster type in turn
  (`EXCLUDE` blocks) and greedily drops types whose removal leaves the fit
  statistically tied. Its `cluster_types` table is the exclusion analysis:
  per type, data coverage (`instances`, `min_distances`), coefficients
  saved, and the score change when removed. Large positive = essential;
  about zero = removable; types with 0 instances are unconstrained and should
  go. Report which types were dropped and why. Physically, a type mattering
  a lot means those cluster environments carry many-body physics the
  pairwise terms can't express.
- For a manual comparison (a specific exclusion set, or a 4-body cutoff
  outside the grid), use `sweep` with `exclude_3b` / `special_maxim_4b` grid
  keys.

## 3. Run

- `chimes-agent hyper-search --data-manifest ... --output-dir 02_fit/search
  --machine dane --queue debug|batch --walltime-hours H [--dry-run]`
- Results are cached per fit under `points/`, so a re-run with a widened grid
  only fits the new points. Resume instead of starting over.

## 4. Judge the result (`hyper_report.json`)

- `final` and `hyperparameters`: the chosen settings; `best/hyper_choice.json`
  is the handoff to model building.
- Each stage's `reason`: "cheapest within one standard error" means several
  models are statistically tied; the cheaper one was taken on purpose.
- `notes`, act on them:
  - *largest value in the grid* → extend that grid and re-run;
  - *refine moved below/above the grid* → same;
  - *holdout ≫ train* → overfitting: smaller orders, or more data;
  - *relative force error > 0.3* → the data is the limit (size,
    coverage, consistency), not the hyperparameters; say so;
  - *small-signal many-body terms* → their gain is suspect.
- `n_timeout`/`n_failed` per stage: failed points are excluded from the
  choice, so check they are not the promising region.
- Compare `holdout_rmse_energy_per_atom` with the energy scale of the
  problem (kcal/mol/atom; 1 kcal/mol ≈ 0.043 eV).

Write `02_fit/HYPER_REPORT.md`: the chosen settings, why (the stage
reasons), the evidence (errors ± standard error), what was tried and ruled
out, open doubts. Then have `chimes-fit-reviewer` read `best/params.txt`,
the report and the holdout before the model is recommended.

## Solver

Default: `lassolars`, α = 1e-5 (raw columns). Keep it for the search unless
equations exceed coefficients by ≳10×. Measured on Cu-Zr (122 frames),
column-normalized solvers (`nridgecv`, `nlasso`) overfit the freed many-body
terms badly (holdout relative force error 1.1-1.5 vs 0.43), while raw
`lassolars` quietly suppresses the tiny-scale 3-/4-body columns. That is
useful regularization with scarce data, and it is also why 4-body stages
usually report "no gain": it reflects the data, not a bug. With a large
dataset, a second search with `--algorithm nridgecv` is a legitimate
comparison. Never mix solvers within one search.

Reference: `docs/commands/hyper-analyze.md`, `docs/commands/hyper-search.md`,
`docs/concepts/cutoffs_and_lambdas.md`.
