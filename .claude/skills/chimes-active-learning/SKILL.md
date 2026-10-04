---
name: chimes-active-learning
description: Improve or stabilize a ChIMES model with active learning - toolkit-managed rounds (md-check harvest, al-batch selection from close contacts + QUESTS + fingerprint + committee, qe-relabel, al-merge, refit at the chosen hyperparameters with n/I weight decay, re-validate, al-status stopping verdict) or al_driver's unattended loop (al-run). Use when the user wants to improve or stabilize a model with more data, asks about al-batch/al-select/al-run/al-merge/al-status, wants the next batch of configs to label, or mentions active learning / ALC cycles / unstable MD. The chimes-active-learner subagent follows this playbook.
---

# Active learning

## Toolkit-managed rounds (the default)

One round, in `<study>/03_al/round<k>/`, each step a stage call:

1. **Candidates.** Run `md-check` with the current model at the target
   temperatures (Slurm; dry run first). `run/harvest.xyzf` holds up to 20
   close-contact and 20 other frames per run.
2. **Prioritize.** Run `al-batch --candidates-xyzf harvest.xyzf --train-xyzf
   <train> --params <model> --fm-setup-in <basis> --budget N`. It combines
   close contacts (first), QUESTS dH, fingerprint D_j² and committee spread,
   drops duplicates, and writes `batch.xyzf` + `batch.json`. For alloys add
   `--structure-weight 0.25` (element-aware fingerprint; see
   `docs/commands/fingerprint.md`). Run `quests`
   on the harvest too (`round<k>/quests/`): `al-status` uses its novel
   fraction for the stopping rule. `al-select` (energy-histogram
   diversity) remains for very large pools.
   For model disagreement as well as structural novelty, run `committee
   --fm-setup-in <best fm_setup.in> --candidates-xyzf harvest.xyzf`.
   `uncertain.xyzf` holds the frames the bootstrap members disagree on
   most. Label the union of novel and uncertain frames first.
3. **Label.** Run `qe-relabel` with **exactly the base set's QE settings**
   (from `provenance.json`). This is a Slurm submission: dry run, approval,
   then `--collect`. Optionally `data-curate --no-holdout` on the labeled
   pool to drop broken frames.
4. **Merge.** Run `al-merge --data-manifest <current manifest> --new-xyzf
   <labeled.xyzf> --cycle k`. It checks the level of theory, keeps the
   holdout fixed, and writes `train.xyzf`, `frame_cycles.json` and a new
   `data_manifest.json`.
5. **Refit at the same hyperparameters.** Refit with:
   - `fm-setup-gen --hyper-choice 02_fit/search/best/hyper_choice.json
     --trjfile <merged train.xyzf> --nframes N`;
   - `amat-build`;
   - `weights --preset <study preset> [--stress-method '["A",[w]]']
     --frame-cycles frame_cycles.json --decay-cycles <n>`;
   - `solve --weights weights.dat`.

   Re-search hyperparameters only when the data has grown a lot. Before
   active learning the literature errs toward complexity (`--prefer richer`)
   and prunes at the end.
6. **Re-validate.** `evaluate` on the unchanged holdout, then `md-check` and
   `fingerprint` again.

7. **Status.** `al-status --study <study>` scores every round's refit on the
   fixed holdout and gives CONVERGED / CONTINUE with reasons
   (`03_al/AL_STATUS.md`, `al_progress.png`). Before the first round, run
   `learning-curve` on the chosen basis: at a plateau, new *conditions*
   (coverage) help and more of the same frames do not.

**Stop** when all of these hold (`al-status` checks them), and say which held:

- stable at every target temperature;
- `below_inner_cutoff_frames` = 0;
- the harvest is no longer distinguishable from the training set by
  fingerprint, or ChIMES-MD is indistinguishable from DFT-MD at a state
  point (Laubach 2026);
- or the user's budget is reached.

Published runs converged in about 8 cycles (Lindsey 2020).

## `al-select` — one diverse batch (standalone)

Given a candidate pool and a current `params.txt`, picks `n_select` frames
whose predicted per-atom energies flatten the selected set's energy
histogram (al_driver's own Metropolis-MC selector). This is **coverage
selection, not uncertainty**: it does not know where the model is wrong, only
that the batch spans the energy range.

Typical loop, each step a separate stage call:

1. Generate candidates: `md-check` with the current model at the target
   temperatures writes `harvest.xyzf` (close contacts first). Pass it as the
   pool, or straight to `qe-relabel` if it is already small.
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
  ground-truth data (Lindsey 2025, 2026). For al-select rounds, use the
  `weights` stage: `--frame-cycles cycles.json --decay-cycles n` with a
  preset (`hierarchical2026`). In al_driver, use `WEIGHTS_*` method `B` in
  `config.py`.
- **Candidates.** `md-check` runs the current model at several state points
  in one job. It writes `harvest.xyzf` with up to 20 close-contact and 20
  other frames per run (the published recipe), ready for `qe-relabel`.
- Refit after each ALC at the *same* hyperparameters. Once the data is
  final, a smaller basis may be justified (Lindsey 2025: err complex before
  AL, prune after).
- **When to stop.** Published practice is stable MD plus RDF, equation of
  state and dynamics consistent with DFT. Quantitatively, stop when
  ChIMES-sampled configurations are indistinguishable from DFT ones by
  cluster-graph fingerprint (Laubach 2026 JCIM): the `fingerprint` stage.
- Multi-element systems can reuse fitted single-element blocks and fit only
  the cross terms (hierarchical transfer learning, Lindsey 2026, npj Comput.
  Mater. 12, 18). The `hierarch` stage does this (`--subtract`, fit the
  cross terms, `--combine`).

## Choosing

- Want the next batch of frames to label, under your own control:
  `al-batch` (close contacts + novelty + committee uncertainty, within a
  budget); `al-select` only for very large pools where energy-histogram
  diversity is enough.
- Want an unattended stabilization campaign and have a prepared study, or
  `auto-build --stabilize`: `al-run`.
- Want to know whether to continue: `al-status` (rounds scored on the fixed
  holdout) and `learning-curve` (is more data worth it at all).

Reference: `docs/commands/al-batch.md`, `docs/commands/al-status.md`,
`docs/commands/al-select.md`, `docs/commands/al-run.md`.
