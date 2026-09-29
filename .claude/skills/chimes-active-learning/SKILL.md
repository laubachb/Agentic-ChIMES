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

## What the literature says (see `chimes-literature`)

- al_driver's selector is Lindsey et al., JCP 153, 134117 (2020). MD frames
  are split into molecule-like clusters, and a Monte Carlo search flattens
  the histogram of ChIMES cluster energies. Memory mode matters little;
  partial memory with ~40 bins worked best and converged in about 8 ALCs.
- Carbon 2.0 (Lindsey 2025) and the hierarchical C/N work (Lindsey 2026)
  ran MD at many state points in parallel. Per state point they took up to
  20 frames with **close contacts** (r just above s_minim) plus up to 20
  others. Close-contact frames are what fix MD instabilities.
- **Weight decay.** Weight each cycle's new frames by n_cycles / I (I = the
  cycle index) so early unphysical frames cannot pull the fit away from
  ground-truth data (Lindsey 2025, 2026). al_driver does not do this
  unprompted; check `config.py` or weight manually when refitting.
- Refit after each ALC at the *same* hyperparameters. Once the data is
  final, a smaller basis may be justified (Lindsey 2025: err complex before
  AL, prune after).
- **When to stop.** Published practice is stable MD plus RDF, equation of
  state and dynamics consistent with DFT. Quantitatively, stop when
  ChIMES-sampled configurations are indistinguishable from DFT ones by
  cluster-graph fingerprint (Laubach 2026 JCIM). The fingerprint tool
  ships in chimes_calculator (`chimesFF/src/FP`) but is not wrapped here.
- Multi-element systems can reuse fitted single-element blocks and fit only
  the cross terms (hierarchical transfer learning, Lindsey 2026, npj Comput.
  Mater. 12, 18). al_driver 2.0 supports this (`src/hierarch.py`); the
  toolkit does not expose it yet.

## Choosing

- Want the next batch of frames to label, under your own control: `al-select`.
- Want an unattended stabilization campaign and have a prepared study, or
  `auto-build --stabilize`: `al-run`.
- Uncertainty-based (committee) selection is not implemented; `evaluate`
  with several `--params` returns `committee_spread` if the user wants to
  rank candidates by disagreement themselves.

Reference: `docs/commands/al-select.md`, `docs/commands/al-run.md`.
