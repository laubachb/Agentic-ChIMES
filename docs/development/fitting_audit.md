# Fitting audit (2026-10-03)

*What is missing, buggy, or worth expanding in how the toolkit fits ChIMES
models. Each finding was checked with a probe on the Cu-Zr example (126
training / 32 holdout MatPES frames) or against the code. Earlier, broader
findings are in the [assessment](assessment.md).*

## Status (same day): all items addressed

| item | done |
|---|---|
| F1 | frames checked against every model before the calculator (`evaluate`, `al-select`, `committee`, `eos-check`) |
| F2 | `fm_setup.in` pre-flight in `amat-build`; native codes run without core dumps |
| F3 | `auto-build` delegates fitting to `hyper-search` |
| F4 | `solve`/`model-build`/`sweep` default to `lassolars` α = 1e-5 |
| G1 | `evaluate` `by_composition`/`by_element`/`worst_frames`; per-group guard in `hyper-search` (on Cu-Zr it kept CuZrZr; pooled 0.313 → 0.303, pure Cu 0.145 → 0.093, E 0.854 → 0.737) |
| G2 | `deploy` writes the penalty (0.02 Å, 1e5) and removes zeroed coefficients (predictions identical, 21 % faster); `md-check` runs with it (frames inside the inner cutoff at 1200 K: 20 → 2 of 81); MD cost counts nonzero coefficients |
| G3 | `alpha` stage in `hyper-search` |
| G4 | memory-capped parallel fits (measured ~20× the dense matrix per fit); note when one fit exceeds the node |
| G5 | design matrices shared across solve-only variants (`alpha`, `stress`, 4-body solvers) |
| G6 | `committee` stage (bootstrap members; force spread 0.41 on holdout vs 1.12 on MD-harvested frames) |
| G7 | `eos-check` stage (Birch-Murnaghan V0/B0/B0', clamped-ion elastic tensor, Born stability) |
| G8 | pre-flight inner-cutoff warnings; `sweep` reports bootstrap SE, per-composition errors and `tied_with_best` |

Follow-up (2026-10-03, second pass): cross-validation, smoothing and
per-pair-λ stages, sensitivity profiles, a generated search report, the
learning curve, QUESTS coverage/selection, `al-batch`/`al-status`,
`qe-converge`, triclinic QE inputs, plots throughout, and a report that
includes them. See the CHANGELOG. CV showed that the Cu-Zr 3-body model has
high variance (fragile frames), which the holdout alone could not reveal.

Not done:

- **Binary A-matrix storage (part of G4).** chimes_lsq.py, which runs the
  LASSO/SVD solves, reads only text. A binary path would mean replacing
  its solve step, which the forks own. The memory cap and DLARS advice
  cover the risk meanwhile.
- **DLARS inside `hyper-search`.** Oversized fits are flagged, not routed.

## Summary

The fitting path works and is now hard to break by accident. But four
things stop it from producing the *best* model a user's data supports:

1. **Pooled metrics hide the composition that matters.** Alloy frames are
   3× worse than pure Cu, and a "statistically tied" exclusion degraded pure
   Cu by 60% without moving the pooled score.
2. **Deployed models are not MD-ready.** No penalty parameters are written,
   so the default penalty is 10-100× weaker than upstream recommends; LASSO's
   zeroed coefficients are never removed.
3. **Regularization is never tuned.** α is fixed at 1e-5 everywhere; energy
   error varies 0.78-1.12 kcal/mol/atom across reasonable α.
4. **It does not scale.** Fits load dense text A-matrices into RAM, and
   parallel fits ignore memory; datasets beyond toy size will run out of
   memory in `hyper-search --machine`.

## Bugs

| # | Severity | Finding (evidence) | Fix |
|---|---|---|---|
| F1 | **High** | **Silent exit on an unknown element.** `evaluate` with a holdout frame containing an element the model does not describe prints nothing and exits 0: chimes_calculator calls `exit()` inside the C library. An agent reads a successful empty result. `hierarch --subtract` and `doctor` use the same path. | Check frame elements against `params.txt` (`io/params.frame_elements_check`) before calling the library in every calculator path. |
| F2 | High | **Frame-count mismatch segfaults chimes_lsq.** `NFRAMES` 200 for a 126-frame trajectory: `chimes_lsq exited -11` plus a core file in the output directory. | Pre-flight `fm_setup.in` against its trajectory before `amat-build`: frame count, elements ⊆ atom types, `S_MAXIM` vs half-box with `N_LAYERS`, `S_MINIM` vs the closest sampled contact. Run native codes with `ulimit -c 0`. |
| F3 | Medium | **`auto-build` still picks the lowest raw holdout RMSE** (`best_by["rmse_force"]`), with none of `hyper-search`'s noise-aware ties, cost model, stresses, weights or MD checks. | Make `auto-build` a thin wrapper: curate → `hyper-search` → `md-check`, or deprecate it. |
| F4 | Low | `solve`/`model-build` default to unregularized `svd` (α unused), while every search uses `lassolars` 1e-5. | One documented default; say which. |

## Gaps that affect model quality

### G1. Per-composition and per-element errors (high)

Holdout relative force error of the deployed model by group:

| group | pooled | Cu-Zr frames | Cu frames | Zr frames | Cu atoms | Zr atoms |
|---|---|---|---|---|---|---|
| error | 0.31 | **0.43** | 0.145 | 0.257 | 0.18 | 0.34 |

The same basis with and without the excluded cross 3-body types:

| exclusions | pooled | Cu-Zr | Cu | Zr |
|---|---|---|---|---|
| none | 0.306 | 0.431 | **0.090** | 0.247 |
| CuCuZr, CuZrZr (chosen) | 0.313 | 0.430 | **0.145** | 0.257 |

The exclusions did not cause the poor alloy error: mixed frames are
data-limited. But they cost pure Cu 60% while the pooled score moved within
noise, and the tie rule accepted the cheaper model.

- `evaluate`: report errors per composition class and per element (forces),
  plus the frames with the largest errors (mislabeled-data and coverage
  diagnostics).
- `hyper-search`: a group guard. A cheaper "tied" candidate must not worsen
  any composition group beyond that group's own bootstrap SE; or score by
  the worst group. Optionally a user-chosen target group (e.g. the alloy).

### G2. MD-ready deployment: penalty and reduction (high)

- **Penalty.** chimes_lsq's documentation says to add penalty parameters
  before MD (prefactor 1e5-1e6 kcal/mol/Å³, onset 0.01-0.05 Å). No stage
  writes them, so chimesFF falls back to 1e4 and 0.01 Å. `md-check` found
  Cu-Zr sampling distances *inside* the inner cutoff at 1200 K.
  `deploy` and `md-check` should write explicit penalty lines (default 1e5 /
  0.02 Å, configurable), and `md-check` can compare penalty settings.
- **Reduction.** LASSO left 94-421 of 726 coefficients nonzero in the Cu-Zr
  4-body candidates. `post_proc_chimes_lsq.py` (in the chimes_lsq fork)
  removes zeroed 3-/4-body parameters. `deploy` should run it, and
  `hyper-search`'s MD-cost model should count nonzero coefficients.

### G3. Regularization strength (medium-high)

LASSO α on the deployed basis (holdout):

| α | 1e-7 | 1e-6 | **1e-5 (used)** | 1e-4 | 1e-3 | 1e-2 | svd | nridgecv |
|---|---|---|---|---|---|---|---|---|
| relative force error | 0.315 | 0.311 | 0.313 | 0.313 | 0.321 | 0.366 | 0.301 | 0.319 |
| E (kcal/mol/atom) | 0.781 | 0.784 | 0.854 | 0.929 | 1.024 | 1.122 | 0.782 | 0.769 |

Forces are flat to 1e-4; energies are 9% worse at the default than at 1e-6.
With many-body terms α decides how many coefficients survive. Add an
`alpha` stage: a short α ladder on the chosen basis, picked by the usual
tie rule, or by BIC, which chimes_lsq already prints. It is cheap if the
A-matrix is reused (G5).

### G4. Memory-aware parallel fitting (medium-high, blocks real datasets)

Fits read `A.txt` with `np.loadtxt` (dense, text, several copies in memory),
and `hyper-search --machine` sets `workers` = cores. A Dane node has
~2.3 GB per core. For example, 2,000 frames × 100 atoms × 3 rows ×
1,000 coefficients is 4.8 GB per fit in float64 before parsing overhead,
so 112 parallel fits cannot fit.

- Estimate memory per fit (rows from `hyper-analyze`, columns from the
  basis) and cap workers to the node's memory.
- Route fits above a size threshold to `solve --algorithm dlars --machine`
  (distributed), which exists but `hyper-search` never uses.
- Store A as binary (`.npy`) after the build; parsing text is the main
  time and memory cost.

### G5. Reuse the design matrix (medium)

Each point deletes its A-matrix after fitting, so stages that only change
the solve (`stress` weights, solver comparison, a future α ladder) rebuild
it. Keep the A-matrix of the current best point (one file, compressed) and
re-solve from it.

### G6. Uncertainty for active learning (medium)

`evaluate` reports `committee_spread` across models, but nothing builds a
committee. A bootstrap committee (refit on resampled frames from the same
A-matrix, cheap with G5) would let `al-select` and `md-check` rank frames by
disagreement, which complements fingerprint novelty.

### G7. Physical validation beyond MD stability (medium)

Published ChIMES validation includes equation of state, elastic response
and dynamics. Cheap, local checks the toolkit lacks:

- E(V) and pressure along an isotropic strain path, compared with DFT data
  for the same structure when present;
- elastic constants from small strains (with stresses now available);
- vacancy / surface energies when the target application needs them.

### G8. Smaller items

- `fm_setup.in` pre-flight (F2) should also warn when `S_MINIM` sits above
  the closest training contact, or below it by much more than the
  documented 0.02 Å.
- Energy fitting uses chimes_lsq's 3 energy rows per frame. Document it
  next to the weight presets, since it triples the effective energy weight.
- `sweep` lacks the per-frame bootstrap that `hyper-search` uses, so its
  table cannot say what is within noise.

## Recommended order

1. **F1, F2** (silent failures; small fixes).
2. **G2** penalty and reduction in `deploy`/`md-check` (direct MD effect).
3. **G1** per-group metrics and the selection guard (accuracy where it
   matters).
4. **G4 + G5** memory-aware workers, binary A-matrix and reuse (unblocks
   real dataset sizes; makes G3/G6 cheap).
5. **G3** α stage, **G6** bootstrap committee, **G7** EOS/elastic checks.
6. **F3/F4** cleanup.
