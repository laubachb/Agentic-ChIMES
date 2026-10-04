# How a model is chosen

`hyper-search` makes one decision per stage. This page is the full rule
set, so a choice can be audited from `hyper_report.json` /
`HYPER_REPORT.md`.

## The score

Relative force error (force RMSE ÷ reference RMS force), plus
`energy_weight` × per-atom energy RMSE when energies are fitted. Forces
dominate: they are 3N rows per frame against one energy.

## Where the score comes from

- **Holdout** (default): the split written by `data-curate`. With ~30
  frames the bootstrap standard error is ~0.06, and most differences
  between points are inside it.
- **k-fold cross-validation** (`--cv-folds k`): every training frame is
  scored once, by a fit that did not see it. Held-out frames get row
  weight 0 in the same design matrix, so a fold is a solve, not a build.
  Folds are group-aware (correlated frames together) and stratified by
  composition; each pair's closest-contact frame stays in training. On
  Cu-Zr, 4-fold CV scores 123 frames with SE 0.035 against the holdout's
  32 frames at 0.06. The external holdout is still reported
  (`ext_holdout_*`).

  CV also exposes **variance**: a model that scores well on one holdout
  but poorly when a quarter of the data is removed is unstable. The
  Cu-Zr 3-body model at 7 Å had holdout 0.31 but CV 0.66 (pure Cu 0.89).
  Many-body columns there are 10⁻² of the 2-body scale, so their
  coefficients are large and change with every subset.

## The tie rule

A point is *tied* with the best when `score − best ≤ max(tolerance × best,
paired bootstrap SE)`. The paired SE resamples the same frames for both
models, so frame difficulty cancels, and small consistent differences are
resolved.

## The per-composition guard

A tied point is rejected if any composition class (e.g. Cu, Zr, Cu-Zr
frames) is worse than in the best point by more than
`max(tolerance × that group's error, the paired SE on that group's
frames)`. A pooled score hid a 60 % loss on pure Cu once (0.09 → 0.145 at
pooled 0.306 → 0.313). Rejections are listed as `group_regressions`.

## Among tied points

- `--prefer cheaper` (default, final models): the lowest estimated MD
  cost, (clusters per atom within each body order's cutoff) × (nonzero
  coefficients of that order). Zeroed coefficients do not count, because
  `deploy` removes them.
- `--prefer richer` (models that will go through active learning): the
  most coefficients. A sparse initial set makes cross-validation favor
  bases that are too small (Lindsey et al. 2025); prune after.

## Stages, in order

| stage | varies | accepts |
|---|---|---|
| 2b | 2-body order × 2-body cutoff | cheapest tied |
| 3b | 3-body order × cutoff | beats 2-body beyond the tie margin |
| 3b_pairs | per-pair 3-body cutoffs | tied and cheaper |
| 4b | 4-body order × cutoff (× solvers) | beats 3-body beyond the margin |
| smoothing | TERSOFF 0.5 / 0.75 vs the current smoothing | tied or better (same cost) |
| exclude | leave-one-cluster-type-out | tied with both the current model and the stage entry |
| lambda, lambda_pairs | global λ scale; one pair's λ at a time | beats 1.0 by the tolerance; tied or better |
| refine | 2-body order ± 2; cutoff midpoints | cheapest tied |
| alpha | regularization ladder | cheapest tied (sparser wins) |
| stress | stress-row weights | lowest pressure error among force/energy ties |

Stages that only change the solve (`alpha`, `stress`, 4-body solver
comparison) share one design matrix per basis.

## After the search

- **Sensitivity profiles** (`profiles` in the report): for each setting,
  every fitted point that differs from the final choice in that setting
  alone. Flat profiles mean the data cannot resolve the setting; steep
  ones mean it matters.
- **Learning curve** ([learning-curve](../commands/learning-curve.md)):
  whether more data would help the chosen basis.
- **MD validation** ([md-check](../commands/md-check.md),
  [eos-check](../commands/eos-check.md)): holdout error is necessary, not
  sufficient (Lindsey 2019).
