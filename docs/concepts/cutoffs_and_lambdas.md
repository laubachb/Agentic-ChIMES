# Cutoffs and lambdas: documented guidance vs. defaults

`stages/_cutoffs.py` and `io/rdf.py` implement ChIMES' own documented
practice for choosing per-pair cutoffs and the Morse λ parameter,
data-driven from the training set — used by
[`auto-build`](../commands/auto-build.md). This page traces every number
back to where it's documented (or says plainly when it isn't).

## S_MINIM (inner cutoff): documented, with an exact number

> "Generally taken as slightly less than smallest distance sampled in
> DFT-MD trajectory." — `chimes_lsq-LLfork/doc/source/lsq_input_file.rst:139`

> "Inner cutoffs are typically set to the lowest sampled distance for each
> given pair type in the training trajectory" — `lsq_input_file.rst:422`

> "Have you set reasonable inner cutoffs (i.e. 0.02 to 0.002 less than the
> smallest observed distance, set individually for each pair type)?" —
> `lsq_input_file.rst:654`

**Implementation**: `_cutoffs.derive_pair_params` computes the true
minimum observed distance per element pair across the training pool
(`io/rdf.pair_min_distance`, minimum-image PBC, orthorhombic boxes) and
subtracts `s_minim_delta` (default **0.02 Å**, the specific example value
`quick_start.rst:121` gives; the documented range is 0.002–0.02, tune with
`--s-minim-delta`).

## S_MAXIM (outer cutoff): documented qualitatively, RDF-derived here

> 2-body "usually set to about 8 Å" (2nd non-bonded solvation shell);
> 3-body "usually set to encompass the 1st non-bonded solvation shell";
> 4-body "set to between the first or second RDF minimum" —
> `lsq_input_file.rst:425-427`, `quick_start.rst:123-127`

No exact formula is given for 3-/4-body — but "solvation shell boundary"
is exactly what an RDF minimum *is*, and that's directly computable from
the training data. **Implementation**:
`io/rdf.pair_rdf` bins pairwise distances into a shape signal (raw counts
÷ r², canceling the r² shell-volume growth that would otherwise hide real
peak/dip structure — not a physically normalized g(r), since only peak/
minimum *locations* are needed) per pair, then:

- **3-body S_MAXIM** = the first RDF minimum after the bonding peak
  (`RDF.first_minimum_after_peak()`).
- **2-body S_MAXIM** = the second RDF minimum (`RDF.second_minimum()`),
  falling back to the documented **8 Å** default when no clear second
  shell is found (small/dilute systems).
- **4-body S_MAXIM** = defaults to the same (shorter) first minimum as
  3-body, matching 4-body's higher computational cost — the documented
  "first *or* second minimum" leaves this a judgment call;
  `s_maxim_4b_use_second=True` switches to the second minimum instead.

Validated against a real physical test case (a simple cubic lattice with
exactly known shell distances via geometry, not just hand-placed numbers)
in `tests/unit/test_rdf.py` and `tests/unit/test_cutoffs.py`.

**Known simplification**: ChIMES' `fm_setup.in` grammar only supports a
*single* global 3-body/4-body S_MAXIM override (`SPECIAL 3B/4B S_MAXIM:
ALL <value>`, not a per-pair-type one, for a simple scalar) — `auto-build`
uses the **minimum** (most conservative) derived value across all pairs
as that global override, so no pair's shell gets over-extended. A future
per-cluster `SPECIFIC` block (`io/fm_setup.py` already round-trips the
grammar for this) could give each pair type its own 3-/4-body cutoff; not
implemented.

### Hard safety bound (found in code, not docs)

`codes/chimes_lsq-LLfork/src/ClassDefs.C:2593-2630`
(`BOXDIM.IS_RCUT_SAFE(S_MAXIM, N_LAYERS)`) requires **S_MAXIM ≤ half the
(layered) box length** in every dimension, or `chimes_lsq` errors
("Outer cutoff greater than half of at least one layered cell vector").
Every S_MAXIM value `_cutoffs.py` derives is capped to this bound
(computed directly from the training frames' own box dimensions ×
`nlayers` — see `docs/commands/auto-build.md`'s `nlayers` flag); the
output reports `<label>_capped`/`<label>_cap_reason` whenever the data-
driven value got overridden, so a small DFT-sized training cell forcing a
short cutoff is visible, not silent.

## MORSE_LAMBDA: documented, RDF-derived here too

> "Generally set to location of first radial distribution peak for each
> pair type." — `lsq_input_file.rst:142,429`

**Implementation**: `RDF.first_peak()` — the same RDF computation used
for S_MAXIM. No formula relating MORSE_LAMBDA to a Morse dissociation
energy/force constant exists anywhere in the docs — only this RDF-peak
(≈ bond-length) heuristic.

## Regularization (`solve --alpha`): documented defaults

> "For LARS or LASSO-based solvers, regularization of 1.0e-2 or 1.0e-5 are
> reasonable starting points for un-normalized and normalized fits,
> respectively." — `lsq_input_file.rst:455,623`

> al_driver's `REGRESS_VAR` defaults to `1.0E-5` when unset —
> `al_driver-LLfork/src/verify_config.py:1191`

`auto-build` defaults `alpha=1e-5` (the normalized-fit value) and does
**not** sweep it — the sweep dimension you asked for is polynomial order
only; regularization stays a fixed, documented value unless you override
`--alpha` explicitly. (`sweep` itself, used standalone, can still sweep
`alpha` — see `docs/commands/sweep.md`.)

## Polynomial order: documented starting point, holdout-validated here

> "There is no hard-and-fast rule for setting the 2/3/4-body polynomial
> orders, but 12/7/3 is generally a good starting point." —
> `lsq_input_file.rst:430` (quick_start.rst's parenthetical CLI example
> says "12 8 4" instead — an inconsistency in ChIMES' own docs, not this
> repo's)

> "Are you using an appropriate bodiedness/polynomial order? Holdout
> cross-validation on the original training set can give you an idea of
> what to use." — `lsq_input_file.rst:672-674`

`auto-build`'s default `order_grid` (`{"2":[10,12,14], "3":[5,7,9],
"4":[null,2,3]}`) is centered on the documented 12/7/3 starting point;
the winning point is picked by **lowest holdout force RMSE** — directly
the documented method, not an invented one.

## Why iterative refinement (`al-run`) is the documented next step

> "\[Complex models\] will not generally be stable for dynamics the first
> time through — in these cases, iterative model fitting is usually
> required." — `lsq_input_file.rst:687`, citing
> [10.1063/5.0021965](https://doi.org/10.1063/5.0021965)

This is exactly what `auto-build`'s optional `stabilize` step (staging the
winning fit as ALC-0, then [`al-run`](../commands/al-run.md)) is for.
