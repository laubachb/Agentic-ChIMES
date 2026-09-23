# Tutorial: an end-to-end holdout study

This walks through the same shape of study as prior hand-rolled ChIMES
campaigns with this toolchain (holdout split → fit → solve → evaluate →
LAMMPS validate), using `chimes-agent` end to end, one stage at a time so
you can see and adjust each intermediate result. If you'd rather run the
whole thing in one call with cutoffs/λ/order chosen automatically from
your data, see [`auto-build`](../commands/auto-build.md) instead — this
tutorial is for when you want to drive (or understand) each step yourself.

Every stage described here (and in the CLI generally) is implemented,
including [`al-select`](../commands/al-select.md) (standalone diversity
selection outside a full AL cycle).

Assumes `chimes-agent setup --machine dane --component all` has already
been run (or at least `chimes_lsq` + `chimes_calculator`).

## 1. Prepare labeled data

You need a `.xyzf` file (ChIMES training format — see `io/xyzf.py`) with
reference forces (and, if fitting energy, per-frame energies). From raw
VASP output, that conversion is `contrib/vasp2xyzf.py` (vendored, Python
2); from Quantum ESPRESSO, `chimes-agent qe-relabel` (submit) + `--collect`
— see `docs/commands/qe-relabel.md`.

## 2. Split into train/holdout

```bash
chimes-agent dataset-select --frames pool.xyzf --method stratified_holdout \
  --holdout-fraction 0.2 --output-dir ./study/split
```

See `docs/commands/dataset-select.md` for `fps`/`random` alternatives.

## 3. Generate `fm_setup.in`

```bash
chimes-agent fm-setup-gen \
  --trjfile /abs/path/to/train.xyzf --nframes <N_TRAIN> \
  --elements C,H,O \
  --order '{"2":12,"3":5,"4":2}' \
  --pair-cutoffs '{"C-C":[1.0,6.0], "C-H":[0.9,6.0], "...": "..."}' \
  --fitener true --fitstrs false \
  --output-dir ./study/fit
```

Inspect `./study/fit/fm_setup.in` before proceeding — this is the single
file that fully determines the basis (element pairs, cutoffs, Chebyshev
order).

## 4. Build the design matrix

```bash
chimes-agent amat-build --fm-setup-in ./study/fit/fm_setup.in --output-dir ./study/fit
```

For a large training set (`SPLITFI true` in `fm_setup.in`, needed for the
DLARS solve path below), add `--machine dane --queue batch --nodes 1
--ntasks-per-node 112` to submit via Slurm instead of running locally.

## 5. Solve

Local algorithms (fine for small/medium bases):

```bash
chimes-agent solve --algorithm lassolars --alpha 1e-5 \
  --A ./study/fit/A.txt --b ./study/fit/b.txt \
  --header ./study/fit/params.header --map ./study/fit/ff_groups.map \
  --output-dir ./study/fit
```

For a large basis, `--algorithm dlars`/`dlasso` (the path most prior large
studies with this toolchain actually used, including the hand-validated
"cliff" workaround now formalized as a live monitor — see
`docs/commands/solve.md`) submits via Slurm:

```bash
chimes-agent solve --algorithm dlars --alpha 1e-5 \
  --A ./study/fit/A.txt --b ./study/fit/b.txt --dim ./study/fit/dim.txt \
  --header ./study/fit/params.header --map ./study/fit/ff_groups.map \
  --machine dane --queue batch --walltime-hours 2 --nodes 1 --ntasks-per-node 112 \
  --output-dir ./study/fit
```

## 6. Evaluate against holdout

```bash
chimes-agent evaluate --params ./study/fit/params.txt \
  --holdout-xyzf /abs/path/to/holdout.xyzf
```

For a committee of several fits (e.g. different λ, or different
retention/FPS-selected subsets), pass `--params` multiple times to get
`committee_spread` alongside each model's own holdout RMSE.

## 7. Validate with LAMMPS

```bash
chimes-agent lammps-run --params ./study/fit/params.txt \
  --structure-xyzf /abs/path/to/holdout.xyzf --frame-index 0 \
  --elements C,H,O --masses '{"C":12.011,"H":1.008,"O":15.999}' \
  --mode single_point --output-dir ./study/lmp_check
```

An independent cross-check against the ctypes evaluator used in step 6 —
they're validated to agree to ~1e-3 (see `docs/commands/lammps-run.md`),
so this step is mainly a sanity check that the model behaves the same way
inside an actual MD integrator, not just single-point.

## 8. Iterate

Adjust cutoffs/order in step 3, or λ in step 5, based on what step 6 shows
— this is the human/agent judgment loop this repo is designed to make fast
to iterate, not to automate away. `chimes-agent sweep` (`docs/commands/sweep.md`)
automates running a whole grid of step 3–6 combinations and reports a
comparison table, but the choice of which point in that table to ship
stays yours — unless you use [`auto-build`](../commands/auto-build.md),
which runs steps 1–6 in one call and does pick a winner (lowest holdout
force RMSE), still reporting the full table alongside its choice.
