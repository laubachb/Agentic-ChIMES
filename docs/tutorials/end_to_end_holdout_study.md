# Tutorial: an end-to-end holdout study

This walks through the same shape of study as prior hand-rolled ChIMES
campaigns with this toolchain (holdout split → fit → solve → evaluate →
LAMMPS validate), using `chimes-agent` end to end. It mixes stages that are
implemented today with a couple that are still stubs (marked below) so the
full intended shape of a study is visible even before every piece lands —
skip the stubbed steps or substitute your own script for now.

Assumes `chimes-agent setup --machine dane --component all` has already
been run (or at least `chimes_lsq` + `chimes_calculator`).

## 1. Prepare labeled data

You need a `.xyzf` file (ChIMES training format — see `io/xyzf.py`) with
reference forces (and, if fitting energy, per-frame energies). If your
labeled data instead comes from raw VASP/QE output, that conversion is
`contrib/vasp2xyzf.py` (vendored, Python 2) today, or `chimes-agent
qe-relabel --collect ...` once implemented (**stub today** — see
`docs/commands/qe-relabel.md`).

## 2. Split into train/holdout

**Stub today** — see `docs/commands/dataset-select.md` for the planned
interface. For now, do this with your own script (as prior studies did with
`make_holdout_split.py`-style tooling), producing a train `.xyzf` and a
held-out `.xyzf` covering the same element/composition space.

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

For a large training set you'll want `SPLITFI true` in the generated
`fm_setup.in` (edit it directly, or extend `fm-setup-gen`'s input) and the
`--hpc` path once it lands (Phase 2) — for now, `amat-build` runs locally.

## 5. Solve

Local algorithms work today:

```bash
chimes-agent solve --algorithm lassolars --alpha 1e-5 \
  --A ./study/fit/A.txt --b ./study/fit/b.txt \
  --header ./study/fit/params.header --map ./study/fit/ff_groups.map \
  --output-dir ./study/fit
```

`--algorithm dlars`/`dlasso` (the path most prior large studies with this
toolchain actually used, including the hand-validated "cliff" workaround
described in `docs/commands/solve.md`) needs the HPC submission layer and
is not wired up yet — see that page for what's already built
(`stages/_cliff_monitor.py`, unit-tested) versus what's still pending.

## 6. Evaluate against holdout

```bash
chimes-agent evaluate --params ./study/fit/params.txt \
  --holdout-xyzf /abs/path/to/holdout.xyzf
```

For a committee of several fits (e.g. different λ, or different
retention/FPS-selected subsets), pass `--params` multiple times to get
`committee_spread` alongside each model's own holdout RMSE.

## 7. Validate with LAMMPS

**Stub today** — see `docs/commands/lammps-run.md`. Once implemented, this
runs a single-point or short MD check via the ChIMES-patched
`lmp_mpi_chimes` build (`chimes-agent setup --component lammps`) as an
independent cross-check against the ctypes evaluator used in step 6 (prior
studies found these two calculators agree to ~1e-9, so this step is mainly
a sanity check that the model behaves the same way inside an actual MD
integrator, not just single-point).

## 8. Iterate

Adjust cutoffs/order in step 3, or λ in step 5, based on what step 6 shows
— this is the human/agent judgment loop this repo is designed to make fast
to iterate, not to automate away. `chimes-agent sweep` (**stub today**, see
`docs/commands/sweep.md`) will eventually automate running a whole grid of
step 3–6 combinations and reporting a comparison table, but the choice of
which point in that table to ship stays yours.
