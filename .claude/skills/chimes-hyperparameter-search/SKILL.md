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

### What the literature adds (see `chimes-literature` for citations)

- λ at the first RDF peak and s_minim just below the closest sampled contact
  are published practice (Lindsey 2017, 2019, 2025). Treat them as settled,
  not searched.
- Many-body cutoffs: published models use 2-body ≥ 3-body ≥ 4-body, with the
  2-body cutoff out to the third solvation shell (Lindsey 2025, Carbon 2.0).
  Per-pair 3-body cutoffs (each pair at its own shell) matched a uniform
  cutoff at lower cost for water (Lindsey 2019). The search uses one global
  3-body cutoff; mention per-pair cutoffs as a manual follow-up
  (`SPECIAL 3B S_MAXIM: SPECIFIC`) when pair shells differ a lot.
- Smoothing: the search builds with CUBIC smoothing, which shrinks 3-/4-body
  terms (Lindsey 2020 JCP). Published models with more than 3-body terms use
  `TERSOFF` with f_O 0.5-0.75. If 3-/4-body terms show "no gain" under
  CUBIC, say that TERSOFF could change the conclusion before calling the
  many-body terms unnecessary.
- **Holdout error alone picked overfit models for water.** The final choice
  was made by comparing MD with DFT (Lindsey 2019). Run `md-check` on the
  chosen model and its tied runners-up (their `params.txt` are under
  `points/<key>/`) at the temperatures that matter, and pass
  `--reference-xyzf` when DFT-MD frames exist. Say in HYPER_REPORT.md what
  it showed. `md-check` beyond a few hundred atom-ps is a Slurm job
  (`--machine`, dry run first).
- **Before active learning, err toward complexity** (Lindsey 2025, Carbon
  2.0). A sparse initial set makes cross-validation favor bases that are
  too small. For an ALC-0 model, run with `--prefer richer`: it takes the
  richest tied model and skips the exclusion stage. Keep the default
  `cheaper` for a final model after active learning.
- Smoothing: `--smoothing 'TERSOFF 0.5'` is the literature choice for
  models with 3-/4-body terms. On Cu-Zr (126 frames) it raised 3-body
  column scales 20× and made 4-body terms active. It still gave no 4-body
  gain, overfit at 3-body order 6, and had lower energy error but no better
  force error. So: CUBIC for small data; TERSOFF (compared with `md-check`)
  when data can constrain many-body terms. A CUBIC "no 4-body gain" should
  be reported with that caveat.
- Weights: `--weights-preset` (`uniform` default, `hierarchical2026` =
  F/E/S 1/0.3/100, `carbon2_large`, `lindsey2020`, `al_driver`) applies
  published weights to every fit. `energy_weight` affects only the *score*.
  Weights trade energy accuracy for force accuracy (Cu-Zr: force error
  0.430 → 0.406, energy error 0.61 → 0.82 kcal/mol/atom at E weight 0.3), so
  pick by what the user needs. Stress data is needed when the model must
  hold density or pressure (Lindsey 2019).
- Quote accuracy as `reduced_force_rmse` from `evaluate` when comparing
  with published models (water 0.24-0.31, Carbon 2.0 0.28).

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
