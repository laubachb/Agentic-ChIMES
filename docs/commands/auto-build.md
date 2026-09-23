# `chimes-agent auto-build`

**Status: implemented.**

The full pipeline: unlabeled configs → QE labeling → data-driven cutoff/
Morse-λ determination → a 2b/3b/4b polynomial-order sweep at those fixed
cutoffs → one "optimal" model (lowest holdout force RMSE) → optionally,
stage the winner as ALC-0 and launch al_driver to stabilize it. See
`src/agentic_chimes/stages/auto_build.py` and
[Cutoffs and lambdas](../concepts/cutoffs_and_lambdas.md) for exactly
which numbers are documented ChIMES practice vs. reasonable defaults.

Unlike [`sweep`](sweep.md) (which deliberately stays hands-off and just
reports a table), `auto-build` is explicitly meant to return one answer —
it still reports the full sweep table in `trace.sweep`, so the pick stays
inspectable.

**Blocking**: this runs the whole pipeline top to bottom in one call,
including polling a QE Slurm job to completion if you give it unlabeled
data. Background the CLI invocation yourself (`nohup chimes-agent
auto-build ... &`, or your own agent's background-process mechanism) if
you don't want to wait on it interactively — there's no self-daemonizing
here (unlike `al-run`, which launches a genuinely multi-day external
process; this pipeline's own steps are each bounded, just possibly slow).

## Usage

Data already labeled:

```bash
chimes-agent auto-build \
  --labeled-xyzf train_pool.xyzf \
  --elements C,H --masses '{"C":12.011,"H":1.008}' \
  --order-grid '{"2":[10,12,14],"3":[5,7,9],"4":[null]}' \
  --algorithm lassolars \
  --output-dir ./auto_build_run
```

Starting from unlabeled configs (labels via QE first):

```bash
chimes-agent auto-build \
  --unlabeled-xyzf unlabeled.xyzf \
  --elements C,H --masses '{"C":12.011,"H":1.008}' \
  --pseudopotentials '{"C":"/path/C.upf","H":"/path/H.upf"}' \
  --ecutwfc 60 --kpoints 2,2,2 \
  --machine dane --queue batch --walltime-hours 4 \
  --order-grid '{"2":[10,12,14],"3":[5,7,9],"4":[null]}' \
  --output-dir ./auto_build_run
```

`--dry-run` only previews the QE submission (renders `pw.in`/the job
script, doesn't submit) — it returns right after that, it does not
preview the rest of the pipeline (see `stages/auto_build.py`'s docstring).

With AL stabilization:

```bash
chimes-agent auto-build ... --stabilize '{
  "alc0_dir": "./my_al_study/ALL_BASE_FILES/ALC-0_BASEFILES",
  "al_run_work_dir": "./my_al_study",
  "cycles": [0, 1, 2, 3]
}' --output-dir ./auto_build_run
```

(`./my_al_study` needs its `config.py` + the rest of `ALL_BASE_FILES/`
already prepared — same scope al-run documents; `auto-build` only stages
the winning `fm_setup.in` + training `.xyzf` into `alc0_dir`, then calls
`al-run` for you.)

## Pipeline phases

1. **Label** (skipped if `labeled_xyzf` given): `qe-relabel` submit →
   `hpc.poll_job` (blocks) → `qe-relabel --collect`.
2. **Split** (skipped if `holdout_xyzf` given): `dataset-select
   stratified_holdout` at `holdout_fraction` (default 0.2) — the
   documented "holdout cross-validation" method.
3. **Derive cutoffs**: `stages/_cutoffs.derive_pair_params` on the
   training split (see [Cutoffs and lambdas](../concepts/cutoffs_and_lambdas.md)).
4. **Sweep**: `sweep` over `order_grid` at the derived cutoffs, fixed
   `alpha`/`algorithm`.
5. **Choose**: lowest `rmse_force_kcal_mol_ang` among the sweep's
   completed points.
6. **Stabilize** (only if `stabilize` given): stage ALC-0 + `al-run`.

## Flags

- `--labeled-xyzf PATH` or `--unlabeled-xyzf PATH` (+ QE flags: one required)
- `--elements`, `--masses` (required)
- QE flags (only used with `--unlabeled-xyzf`): `--pseudopotentials`,
  `--ecutwfc`, `--ecutrho`, `--kpoints`, `--smearing`, `--degauss`,
  `--conv-thr`, `--machine`, `--queue`, `--walltime-hours`, `--nodes`,
  `--ntasks-per-node`
- `--holdout-xyzf PATH` (optional, else auto-split), `--holdout-fraction`
  (default 0.2), `--split-seed` (default 42)
- `--s-minim-delta` (default 0.02, documented range 0.002–0.02),
  `--s-maxim-2b-default` (default 8.0, documented typical value),
  `--nlayers` (default 1, for the box-safety bound)
- `--order-grid '{"2":[...],"3":[...],"4":[null or int,...]}'` (default
  centered on the documented 12/7/3 starting point)
- `--algorithm` (default `lassolars`, runs locally; set `dlars`/`dlasso`
  to solve via the same `--machine` used for QE labeling instead),
  `--alpha` (default 1e-5, documented normalized default — fixed, not
  swept), `--fitener`, `--fitstrs`, `--max-frames`
- `--stabilize '{"alc0_dir","al_run_work_dir","config_py","cycles"}'`

`--queue`/`--walltime-hours`/`--nodes`/`--ntasks-per-node` are shared
between the QE labeling step and a `dlars`/`dlasso` solve step (the common
single-cluster-campaign case) — use `sweep`/`solve` directly if you need
different HPC settings for each.

## Output

```json
{
  "trace": {
    "qe_submit": { "...": "..." }, "qe_collect": { "...": "..." },
    "split": { "...": "..." },
    "cutoffs": { "pairs": { "C-C": {"s_minim": 1.28, "s_maxim_2b": 8.0, "s_maxim_3b": 4.1, "morse_lambda": 1.52, "...": "..."}, "...": "..." }, "box_safety_bound": 29.5 },
    "sweep": { "n_points": 9, "n_done": 9, "results": [ "..." ], "table_csv": "..." }
  },
  "train_xyzf": "./auto_build_run/split/selected.xyzf",
  "holdout_xyzf": "./auto_build_run/split/holdout.xyzf",
  "cutoffs": { "...": "same as trace.cutoffs" },
  "chosen": {"index": 4, "overrides": {"order_2b": 12, "order_3b": 7, "order_4b": null}, "params": "./auto_build_run/sweep/point_0004/params.txt", "rmse_force_kcal_mol_ang": 3.1, "status": "done"},
  "params": "./auto_build_run/sweep/point_0004/params.txt",
  "stabilize": {"status": "launched", "pid": 123456, "log": "..."}
}
```

## Known limitations

- Orthorhombic training boxes only (`io/rdf.py`'s scope, same as
  `io/lammps_data.py`).
- The 3-/4-body outer cutoff is one global value across all pairs (the
  minimum of each pair's own derived value) — see
  [Cutoffs and lambdas](../concepts/cutoffs_and_lambdas.md#s_maxim-outer-cutoff-documented-qualitatively-rdf-derived-here)
  for why and the future per-pair extension.
- `SPLITFI`-true / split-file A-matrices aren't produced by `fm-setup-gen` here (auto-build always generates a single-file basis) -- for a basis large enough to need DLARS' split-file path, build `fm_setup.in`/`amat-build` manually and use `sweep`/`solve` directly instead of `auto-build`.
