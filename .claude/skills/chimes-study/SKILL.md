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

Create it with `chimes-agent study --init <dir> --name ... --goal "<user's request>" --elements ...`
(on /p/lustre2 for HPC work). `study.json` is the registry every phase writes
its artifacts into (`study --study <dir> --register key=path`), and marks the
directory so login-node CPU time is recorded automatically.

```
<study>/study.json  STUDY.md         goal, decisions log, phase status (you keep STUDY.md)
  01_data/     DATA_PLAN.md, fetch_*/ generate/ qe_*/ curate/data_manifest.json
  02_fit/      search/hyper_report.json, search/best/{params.txt,hyper_choice.json}, HYPER_REPORT.md
  03_al/       al_driver study / al-select rounds
  04_md/       lammps-run validation runs
  05_bench/    benchmark.json, BENCHMARK.md
  06_deploy/   params.txt, in.lammps.example, MODEL_CARD.md
  usage/       usage_report.json, local.jsonl (CPU-hour ledger)
  REPORT.md    REPORT_FACTS.json
```

## Phases

| # | Phase | Who | Hands off |
|---|---|---|---|
| 1 | Plan | you, with the user | `study.json`, `STUDY.md` |
| 2 | Data selection and curation | **`chimes-data-curator`** | `01_data/curate/data_manifest.json` |
| 3 | Hyperparameter search (cutoffs, λ, orders, 4-body, exclusions) | **`chimes-hyperparameter-tuner`** | `02_fit/search/best/` + `HYPER_REPORT.md` |
| 4 | Build/check the model (default weighting) | you + `chimes-fit-reviewer` | final `params.txt` registered |
| 5 | Active learning | not yet an agent: `chimes-active-learning` skill | stabilized model |
| 6 | MD validation | not yet an agent: `lammps-run` (orthorhombic cells) | `04_md/*` registered as `md_runs` |
| 7 | Benchmark + compute accounting | **`chimes-benchmark`** | `05_bench/benchmark.json`, `usage/usage_report.json` |
| 8 | Deploy + final report | `deploy`, then **`chimes-report-writer`** | `06_deploy/MODEL_CARD.md`, `REPORT.md` |

An MD agent (candidate generation, stability checks) is planned.

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

## 3. Hyperparameters

Delegate to `chimes-hyperparameter-tuner` with the manifest path, study
path, and the user's MD cost limits / compute budget. `NEEDS_JOB`: show the
user the dry-run job (fits, node, walltime), submit on approval, delegate
waiting to `chimes-job-monitor`, then resume the tuner to interpret
`hyper_report.json`. Log the chosen settings and the reasons in `STUDY.md`.

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

## 6-8. Validate, benchmark, deploy, report

- MD: run `lammps-run` (or LAMMPS directly) in `04_md/<name>/` at the
  conditions the user cares about; register each run
  (`--register md_runs=<dir>`). Unstable runs are the signal for active
  learning.
- Delegate to `chimes-benchmark` with the user's intended production runs.
  It returns the scaling job for approval, then the sizing recipe and the
  study's CPU-hours.
- `chimes-agent deploy --study <dir>` packages the model and its card.
- Delegate to `chimes-report-writer` for `REPORT.md`. Offer to publish it.

## Always

- HPC submissions only after the user approves (dry-run first).
- Keep `STUDY.md` current: decisions, paths, numbers, open gaps.
