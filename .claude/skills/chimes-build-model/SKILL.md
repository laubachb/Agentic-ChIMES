---
name: chimes-build-model
description: Build, tune and evaluate a ChIMES model one stage at a time - split data, generate fm_setup.in, build the A-matrix, solve, evaluate on holdout, sweep polynomial order/cutoffs/alpha, validate in LAMMPS. Use for hands-on fitting, when the user wants control over cutoffs/orders, when auto-build does not apply (non-orthogonal cells, DLARS-scale basis), or asks to compare models or diagnose a bad fit.
---

# Stage-by-stage ChIMES model building

Order of stages (each takes `--output-dir`; use one directory per experiment):

1. `dataset-select` — `stratified_holdout` for a train/holdout split (holdout
   is required for everything downstream). `fps`/`random` to subsample.
2. `fm-setup-gen` — writes `fm_setup.in`. **Read the file it produced**
   before continuing; it fully determines the basis.
3. `model-build` — amat-build + solve in one call. Or `amat-build` then
   `solve` separately when you want to reuse one A-matrix across solvers.
4. `evaluate --params ... --holdout-xyzf ...` — holdout force/energy RMSE
   (kcal/mol/Angstrom, kcal/mol).
5. `sweep` — grid over `order_2b/3b/4b`, `default_s_minim/s_maxim`,
   `alpha`, `algorithm`. Reports a table and `best_by`; it does not pick.
6. `lammps-run` — single-point/MD check inside the real integrator. Should
   agree with `evaluate` to ~1e-3; disagreement means a units or cutoff bug,
   not a model problem.

Always run `<stage> --describe` for exact flags.

## Choosing parameters (documented ChIMES practice)

- **Inner cutoff `s_minim`:** 0.002-0.02 Angstrom *below* the minimum pair
  distance in the training set. Too large and evaluation hits unsampled
  short range.
- **Outer cutoff `s_maxim`:** 2-body about 8 A; 3-body around the first
  non-bonded shell; 4-body between the first and second RDF minimum. Must be
  <= half the smallest box length (times layers) or chimes_lsq errors.
- **`morse_lambda`:** location of the first RDF peak per pair.
- **Orders:** start at 12/7/3 (2b/3b/4b). Sweep 2b first, then 3b with 2b
  fixed; add 4b only if 3b plateaus and the basis size is affordable — 4b
  cost grows steeply.
- **`alpha`:** 1e-5 for normalized fits, 1e-2 for un-normalized.
- **Algorithm:** `svd`/`lassolars`/`ridge` run locally and suit small or
  medium bases. `dlars`/`dlasso` are for large bases and need
  `--machine`; load `chimes-hpc-jobs` first.

## Diagnosing results

- Holdout RMSE much worse than train: overfit. Lower order or raise `alpha`.
- Sweep table flat across orders: the data, not the basis, is limiting;
  more or more diverse data helps more than a bigger basis.
- Forces fine but energies poor (or vice versa): check `fitener`/`fitstrs`
  flags in `fm_setup.in` and the unit conventions
  (`docs/concepts/units_and_conventions.md`).
- `chimes_lsq` fails on cutoffs: half-box limit; shrink `s_maxim`.
- Before recommending a model, hand its sweep/evaluate output to
  `chimes-fit-reviewer`.

Sweep grid points that fail are recorded (`status: failed`, `error`) and do
not stop the sweep — always check `n_failed`.

Reference: `docs/tutorials/end_to_end_holdout_study.md`,
`docs/commands/sweep.md`, `docs/concepts/cutoffs_and_lambdas.md`.
