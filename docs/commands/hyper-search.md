# `chimes-agent hyper-search`

**Status: implemented.** Validated end to end on real curated Cu-Zr data,
submitted to Dane `pdebug` by its own `--machine` path (52 fits,
12 minutes, see *Worked example*).

Searches ChIMES outer cutoffs, Morse lambdas and 2-/3-/4-body polynomial
orders, and picks the cheapest model statistically tied with the best one
on the holdout set. Inner cutoffs and base lambdas come from
[`hyper-analyze`](hyper-analyze.md) and are not searched: an inner cutoff
above the smallest sampled distance throws data away, and one below it
extrapolates. The [`chimes-hyperparameter-tuner`](../concepts/claude_code_integration.md)
agent plans, launches (with your approval) and interprets these searches.

## Usage

```bash
# preview the Slurm job, then submit it (every fit runs on the compute node)
chimes-agent hyper-search --data-manifest study/01_data/curate/data_manifest.json \
  --machine dane --queue debug --walltime-hours 1 --dry-run --output-dir study/02_fit/search
chimes-agent hyper-search --data-manifest ... --machine dane --queue debug --walltime-hours 1 --output-dir study/02_fit/search
# -> job id; results in study/02_fit/search/search/hyper_report.json
```

Without `--machine` it runs in the current process, which is only
appropriate for small grids: every point is a full `amat-build` + solve +
evaluate. On a node, `--workers` defaults to the node's cores.

## Stages

Each stage is a small grid; later stages hold earlier choices fixed.

| Stage | Grid | Keeps the result only if |
|---|---|---|
| `2b` | `orders_2b` × `s_maxim_2b`, no many-body terms | always |
| `3b` | `orders_3b` × `s_maxim_3b` (≤ 2-body cutoff) | it beats the 2-body model beyond the tie margin |
| `4b` | `orders_4b` (2, 3) × `s_maxim_4b` (first-shell end, midway; ≤ 3-body cutoff) × `four_body_solvers` (default: the search solver only); `--four-body auto` runs it only if 3-body improved the score by ≥ `min_gain` | it beats the 3-body model beyond the tie margin |
| `exclude` | leave-one-type-out over the model's 3-/4-body cluster types (`EXCLUDE` blocks), `exclude_rounds` (2) greedy rounds | removing the type(s) leaves the fit tied with both the current model and the model that entered the stage |
| `lambda` | one scale factor on every pair's λ | it beats scale 1.0 by more than `tolerance` |
| `refine` | `order_2b` ± 2 with many-body terms fixed | cheapest tied point |

Cutoff candidates come from the RDF shells (`hyper-analyze`); many-body
candidates reach toward the second shell because at the first shell ChIMES'
cubic smoothing leaves those terms almost no signal. N_LAYERS is set per
fit from the largest cutoff and the thinnest cell.

**Smoothing.** `--smoothing` sets `FCUTTYP` for every fit. `CUBIC` is the
default and chimes_lsq's own. The alternative is `'TERSOFF <f_O>'` with
0 < f_O < 1: it leaves interactions untouched below r_c(1 − f_O) and only
smooths the outer part. Published models with 3- and 4-body terms use
TERSOFF with f_O = 0.5 to 0.75 (see [literature](../concepts/literature.md)),
because cubic smoothing shrinks many-body contributions. When 3-/4-body
stages run with CUBIC, the report carries a note saying so, since a "no gain"
result may come from the smoothing rather than the physics. The value is
part of each point's cache key only when it is not CUBIC, so older CUBIC
caches stay valid. It is recorded as `fcuttyp` in `hyper_choice.json`.

**Measured on Cu-Zr (126 MatPES frames), same grids, stages 2b/3b/4b:**

| | CUBIC | TERSOFF 0.5 |
|---|---|---|
| 3-body column scale vs 2-body | 0.015-0.044 | 0.16-0.76 |
| 4-body column scale | 1e-10 to 3e-7 (inert) | 0.001-0.035 |
| 4-body vs 3-body | identical to 1e-10 | changes the fit, no gain (0.381 vs 0.370) |
| best 3-body model, relative force error | 0.309 ± 0.071 (8/4, 7.0 Å) | 0.370 ± 0.078 (8/4, 5.2 Å, 2-body 6 Å) |
| energy error, kcal/mol/atom | 0.85 | 0.67 |
| 3-body order 6 | tied | overfits (0.47 ± 0.13) |

TERSOFF does make many-body terms real, as the literature says. On this
small dataset it adds capacity that overfits, and the forces-plus-energy
score does not favor it. Force errors are tied within the standard error,
and the comparison is confounded by the 2-body stage choosing 6 Å. The
earlier "4-body adds nothing" finding therefore reflects the data, not the
smoothing.

CUBIC stays the default. Use TERSOFF when there is enough data to
constrain 3-/4-body terms (several hundred diverse frames or more), or
when energies matter more than forces, and compare the two with
`md-check`.

## How a point is chosen

- **Score**: holdout force RMSE ÷ holdout reference-force RMS
  (`relative_force_error`), plus `energy_weight` × per-atom energy RMSE
  (kcal/mol/atom) when energies are fitted (`--objective auto`).
- **Tie margin**: for each point, max(`tolerance` × best score, standard
  error of the *paired* difference to the best). The paired SE resamples
  holdout frames jointly for both models, so frame-to-frame difficulty
  cancels; it separates consistent small differences that each model's own
  SE (~0.07-0.09 with 30 holdout frames) would hide.
- **Choice**: the cheapest point within its tie margin, by estimated MD
  cost per atom: for each body order n, (clusters within its cutoff) =
  (ρ·4/3·π·r³)^(n−1) with ρ the training set's median number density, times
  that body order's coefficient count. A slightly longer cutoff with far
  fewer coefficients can be cheaper (3-body order 4 at 7.0 Å costs about half
  of order 6 at 6.33 Å on Cu-Zr). Each point reports `md_cost`.
- **Budget**: points with more than `max_param_ratio` (0.5) coefficients per
  equation are reported, never chosen.
- **Signal flag**: each fit reports the 3-/4-body column scale relative to
  2-body (`signal_3b`/`signal_4b`). Below `min_signal` (1e-9) the terms can
  only act through huge coefficients; this is flagged in the notes
  (`--exclude-inert` to exclude them).
- **Runaway builds**: `--max-fit-seconds` (600) abandons a design-matrix
  build (many-body cutoffs on 1-2 Å-wide cells can take very long); the
  point is reported as `timeout`.

## Cheaper or richer (`--prefer`)

- `cheaper` (default) takes the tied point with the lowest estimated MD
  cost. This is right for a final model.
- `richer` takes the tied point with the most coefficients (within
  `max_param_ratio`) and skips the exclusion stage. Use it for a model that
  will go through active learning. A sparse initial set makes
  cross-validation favor bases that are too small, so the literature errs
  toward complexity before active learning and prunes once the data is
  final (Lindsey et al. 2025, see [literature](../concepts/literature.md)).

## Fitting weights (`--weights-preset`)

Every fit uses the named [weights](weights.md) preset: `uniform` (default),
`al_driver`, `lindsey2020`, `carbon2_large` or `hierarchical2026`. Weights
matter only when energies (or stresses) are fitted with the forces. The
score is still the holdout error, so compare presets with separate
searches. On Cu-Zr, energy weight 0.3 instead of 1 moved one model's
relative force error from 0.430 to 0.406 and its energy error from 0.61 to
0.82 kcal/mol/atom. Like `--smoothing`, a non-default preset enters the
cache key, and `hyper_choice.json` records it as `weights`.

## Validate the finalists in MD

Holdout error is necessary but not sufficient. For water it favored bases
that were over-structured or unstable in MD. Before finalizing, run
[md-check](md-check.md) on the chosen model and its tied runners-up (the
stage tables list them), at the temperatures that matter.

## 4-body sweeps are deliberately light

4-body design-matrix builds scale steeply: quartets grow as r⁹, multiplied
by N_LAYERS periodic images on small cells (a 7 Å 4-body build on 2 Å-wide
cells ran for over 5 minutes before being stopped). Defaults are therefore
orders 2-3, two cutoffs, one solver, 600 s per fit: 4 fits for a binary. On
Cu-Zr, 4-body terms from 56 up to 3,172 extra coefficients never changed
the result under `lassolars`, and `blocklasso` (which lets them in) was worse
everywhere, so the light default loses nothing there. Widen it only with a
reason and a time budget.

## Cluster-type exclusions

The `exclude` stage reads each 3-/4-body type's coverage from chimes_lsq's
own log (`instances`, `min_distances`), then refits with each type removed.
Its `cluster_types` table is the exclusion analysis:

| column | meaning |
|---|---|
| `instances` | contributing clusters (periodic images included; 0 = absent from the data, unconstrained if kept) |
| `coefficients_saved` | basis reduction from removing the type |
| `score_change_when_removed` | per round; large positive = essential, ≈0 = removable |
| `excluded` | final decision |

With a small holdout the stage can be aggressive: losses below the holdout's
resolution are accepted. When exclusions raise the *training* error by more
than 10 %, a note says so. That loss is real even though the holdout can't
measure it.

## Solver

Default `lassolars`, α = 1e-5, on raw columns. Measured on the same Cu-Zr
data, column-normalized solvers overfit the many-body terms they free
(holdout 1.1-1.5 vs 0.43 for a 3+4-body basis; `docs/commands/solve.md`).
A consequence: with little data, 4-body terms rarely enter, and the `4b`
stage reports "no gain". With far more equations than coefficients, a
separate search with `--algorithm nridgecv` is a fair comparison; never mix
solvers within one search.

## Output

- `hyper_report.json`: every stage's full table (score, paired tie margin,
  coefficients, N_LAYERS, train vs holdout, signal), the chosen and best
  points, the reason for each choice, and notes (grid edges, overfitting,
  small-signal terms, data-limited error).
- `best/params.txt`, `best/fm_setup.in`, `best/hyper_choice.json`: the
  chosen hyperparameters in `fm-setup-gen` form, the handoff to model
  building.
- `points/<hash>/`: one directory per fit. A re-run reuses them, so a
  widened grid only fits new points and an interrupted search resumes.

## Worked example: Cu-Zr from MatPES

126 training / 32 holdout frames (1-34 atoms, mostly triclinic, thinnest
cell 2.0 Å), 3-body cutoffs extended to 4.09-7.0 Å, submitted with
`--machine dane --queue debug`: 61 fits in total across resumed runs, no
timeouts.

| Stage | Result | Why |
|---|---|---|
| 2b | order 8 @ 8.0 Å | lowest score and cheapest |
| 3b | order 4 @ 7.0 Å | tied with the best; ~half its MD cost. 3-body at the 4.1 Å first shell is clearly worse |
| 4b | none | 4 light fits; 4-body terms change nothing |
| exclude | drop Cu-Cu-Zr, Cu-Zr-Zr | removal costs: Zr-Zr-Zr +0.098 (essential), Cu-Zr-Zr +0.037, Cu-Cu-Cu −0.003, Cu-Cu-Zr −0.001 |
| lambda | scale 1.0 | no scale beat the RDF peaks |
| refine | 2-body order 6 | tied, cheaper |

Final: 52 coefficients, N_LAYERS 4, holdout relative force error
0.31 ± 0.06, energy 0.85 kcal/mol/atom, estimated MD cost ~1/3 of the
model before exclusions (and ~1/5 of 3-body order 6 @ 6.33 Å). The notes flag the 2-/3-body cutoffs at the top of
their grids, refine going below the 2-body grid, a 28 % training-error rise
from the exclusions that the 32-frame holdout cannot resolve, and the error
level as a data limit. The last two are the case for more data (active
learning) before trusting the exclusions.
