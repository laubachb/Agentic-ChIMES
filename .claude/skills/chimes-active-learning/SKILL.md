---
name: chimes-active-learning
description: Select diverse configurations for relabeling and run ChIMES active learning with al_driver. Use when the user wants to improve or stabilize a model with more data, asks about al-select or al-run, wants the next batch of configs to send to QE, or mentions active learning / ALC cycles / unstable MD.
---

# Active learning

Two different tools; pick by scope.

## `al-select` — one diverse batch (standalone)

Given a candidate pool and a current `params.txt`, picks `n_select` frames
whose predicted per-atom energies flatten the selected set's energy
histogram (al_driver's own Metropolis-MC selector). This is **coverage
selection, not uncertainty**: it does not know where the model is wrong, only
that the batch spans the energy range.

Typical loop, each step a separate stage call:

1. Generate candidates (e.g. `lammps-run` MD with the current model, dump
   frames into one `.xyzf`).
2. `al-select --candidate-frames pool.xyzf --params params.txt --n-select N
   --output-dir sel_k [--central-repo <previous central_repo_out>]`
3. `qe-relabel` the `selected_xyzf` (submit, then `--collect`); see
   `chimes-hpc-jobs`.
4. Append labels to the training set, refit (`chimes-build-model`),
   `evaluate` on the *unchanged* holdout.
5. Pass the returned `central_repo_out` as `--central-repo` next round so
   later batches cover what earlier ones did not.

Requires the `al-select` extra (`pip install -e ".[al-select]"`, matplotlib +
cycler). If `gen_subset` dies with a bare `exit()`, the pool is degenerate
(too few candidates per histogram bin): lower `--histogram-bins` or enlarge
the pool.

## `al-run` — al_driver's full loop (detached)

Launches al_driver's `main.py` and returns a PID immediately; it then runs
for hours to days and submits its own Slurm jobs. It needs an already-prepared
study: `ALL_BASE_FILES/` (ALC-0 data, MD and QM templates) plus `config.py`.
Templates are system-specific and **not** generated here. The fastest smoke
test is `codes/al_driver-LLfork/examples/simple_iter_single_statepoint-lmp-test/`
(ChIMES-LAMMPS as both MD engine and "QM"), which needs no DFT.

- Before launching, confirm with the user: it is long-lived and spends
  allocation.
- Monitor with `al-run --status-of <pid>` and `tail driver.log`; delegate
  waiting to `chimes-job-monitor`. Stop with `al-run --stop <pid>` only when
  asked.

## Choosing

- Want the next batch of frames to label, under your own control: `al-select`.
- Want an unattended stabilization campaign and have a prepared study, or
  `auto-build --stabilize`: `al-run`.
- Uncertainty-based (committee) selection is not implemented; `evaluate`
  with several `--params` returns `committee_spread` if the user wants to
  rank candidates by disagreement themselves.

Reference: `docs/commands/al-select.md`, `docs/commands/al-run.md`.
