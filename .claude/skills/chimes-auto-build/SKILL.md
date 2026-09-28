---
name: chimes-auto-build
description: Run the one-shot ChIMES pipeline (configs -> optional QE labeling -> data-driven cutoffs/lambdas -> polynomial-order sweep -> one optimal params.txt -> optional active-learning stabilization). Use when the user has a pool of configurations and wants "a model" without hand-tuning, or says auto-build, "build me a ChIMES model from these configs", or "label these with QE and fit".
---

# Auto-build a ChIMES model

`auto-build` is the only stage that makes a judgment call for the user: it
picks the sweep point with the lowest holdout force RMSE. Everything it
decided stays inspectable in the returned `trace`.

## Decide first

1. **Labeled or unlabeled?** Labeled `.xyzf` -> `--labeled-xyzf`. Unlabeled
   -> `--unlabeled-xyzf` plus QE settings; the pipeline submits QE, blocks
   on the Slurm job, then collects. For QE you need pseudopotential paths,
   `ecutwfc`, `kpoints` and a `--machine`. Ask the user for any you cannot
   infer; do not invent cutoffs or pseudopotentials.
2. **Orthorhombic boxes only.** If the configs have non-orthogonal cells,
   auto-build cannot derive cutoffs; use the manual `chimes-build-model`
   skill with explicit `--pair-cutoffs`.
3. **Large basis (SPLITFI / DLARS needed)?** auto-build always emits a
   single-file basis. For DLARS-scale fits, use `chimes-build-model`.

## Run it

1. `chimes-agent auto-build --describe` for the current flag list.
2. If labeling with QE: first `--dry-run` (previews only the QE submission)
   and read the rendered job script; check `ntasks-per-node`, walltime, and
   that the output dir is under `/p/lustre2/$USER` (see `chimes-hpc-jobs`).
3. Write inputs to a JSON file and run with `--json-in`, in the background
   (`run_in_background`) — it blocks through QE and every sweep point.
4. A modest default `--order-grid` is centered on the documented 12/7/3
   starting point. Widen it only if the first result sits at a grid edge.

## Read the result

- `chosen` — winning grid point, its `params`, and its holdout RMSE.
- `trace.cutoffs.pairs.<pair>` — derived `s_minim`, `s_maxim_2b/3b`,
  `morse_lambda`. **Check the `capped` flags**: `capped: true` means the
  data-driven outer cutoff was overridden by the half-box-length safety
  limit (common with small DFT cells). Tell the user; it usually means the
  cell is too small for the physics they want.
- `trace.sweep.results` — the full table. Report the chosen point *and*
  whether neighbours are close; a flat table means the choice is not
  meaningful and the simpler (lower order) model is preferable.
- Hand the finished `params.txt` to `chimes-fit-reviewer` before
  recommending it.

## Stabilize (optional)

`--stabilize '{"alc0_dir","al_run_work_dir","cycles"}'` stages the winner as
ALC-0 and launches al_driver. It requires an already-prepared al_driver study
(`config.py`, `ALL_BASE_FILES/`); see `chimes-active-learning`. If the user
has none, say so rather than fabricating one.

Reference: `docs/commands/auto-build.md`,
`docs/concepts/cutoffs_and_lambdas.md`.
