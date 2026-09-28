---
name: chimes-study
description: Run a complete ChIMES model-building study from a user's high-level request - plan, gather and curate data, search hyperparameters, build the model, stabilize with active learning. Use when the user describes an end goal ("I need a ChIMES potential for molten Cu-Zr", "build and stabilize a model for X") rather than a single step, or asks to start or resume a study.
---

# ChIMES study orchestration

You (the main conversation) are the orchestrator. You own the plan, the
user conversation and every approval; specialist agents do phases and
report back. Each phase ends with a file the next phase reads, so a study
can stop and resume.

## Study layout

```
<study>/                      under /p/lustre2/$USER/... (HPC jobs need it)
  STUDY.md                    goal, decisions log, phase status (you keep this)
  01_data/                    DATA_PLAN.md, fetch_*/ generate/ qe_*/ curate/
    curate/data_manifest.json -> handoff to phase 3
  02_fit/                     sweeps, chosen params.txt, fit report
  03_al/                      al_driver study / al-select rounds
```

## Phases

| # | Phase | Who | Hands off |
|---|---|---|---|
| 1 | Plan | you, with the user | `STUDY.md` |
| 2 | Data selection and curation | **`chimes-data-curator`** subagent | `01_data/curate/data_manifest.json` |
| 3 | Hyperparameter search | not yet an agent: `chimes-build-model` / `chimes-auto-build` skills | sweep table, chosen settings |
| 4 | Build model (default weighting) | not yet an agent: `chimes-build-model` | `02_fit/params.txt` + `chimes-fit-reviewer` verdict |
| 5 | Active learning | not yet an agent: `chimes-active-learning` | stabilized model |

An MD agent (candidate generation, stability checks) is planned. Until
then, `lammps-run` covers MD.

## 1. Plan

From the request, settle and write to `STUDY.md`: elements; what the model
must describe (phases, T/P range, defects/surfaces); accuracy or stability
needs; the **label target** (which QM engine and settings, since active
learning must reuse them: QE via `qe-relabel` is built in); compute budget
and machine. Ask the user only what changes the plan; state the defaults
you chose for the rest. Create the study directory on lustre.

## 2. Data

Delegate to `chimes-data-curator` with a self-contained brief: the user's
request, the study path, and the decisions from `STUDY.md`. It returns a
fixed-format report. Then:

- `NEEDS_DECISION`: put its questions to the user with its numbers and
  recommendation. Resume the same agent (SendMessage) with the answers
  rather than starting a new one.
- QE commands awaiting approval: show the dry-run summary (frames, nodes,
  walltime), get a yes, run the command yourself in the background, hand
  waiting to `chimes-job-monitor`, then resume the curator to collect and
  curate the labeled output.
- `DONE`: check `data_manifest.json` exists and read its `warnings` and
  `pairs`; log the result in `STUDY.md`.

## 3-4. Fit

Read `data_manifest.json`: `train_xyzf`/`holdout_xyzf`, `level_of_theory`,
`pairs.*.min_distance` (inner cutoffs sit just below these),
`fit_hints.nlayers_required` (`N_LAYERS` for the outer cutoff you choose),
`fit_hints.fitener`. Weighting: use the default (uniform) weights until
custom schemes exist; record that in `STUDY.md`. Note `auto-build` requires
orthorhombic frames; with triclinic data use the stage-by-stage route.
Before recommending a model, get `chimes-fit-reviewer`'s verdict.

## 5. Active learning

New frames must be labeled with exactly the level of theory in
`data_manifest.json` (same QE settings hash in `provenance.json`); curate
each round's labels with the base data using the same checks.

## Always

- HPC submissions only after the user approves (dry-run first).
- Keep `STUDY.md` current: decisions, paths, numbers, open gaps.
