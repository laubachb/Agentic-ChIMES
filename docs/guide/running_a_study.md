# Running a study

A *study* is one model's development from a request to a deployed,
documented potential. You talk to Claude Code, which acts as the
orchestrator (skill `chimes-study`). It plans with you, hands each phase to
a specialist agent, and brings decisions and cluster submissions back to you.

## Starting

Describe the goal in your own words. Useful things to say up front:

- **Elements and the physics**: phases (crystal, liquid, amorphous),
  temperature and pressure range, defects or surfaces.
- **Label source**: "Quantum ESPRESSO, PBE, these pseudopotentials", or
  "reuse MatPES labels". Active learning later must use the same settings.
- **Compute**: machine, bank/account, rough budget.
- **Downstream use**: system sizes and simulated times you plan to run. The
  benchmark sizes its advice to them.
- **Where**: a study directory on shared scratch (`/p/lustre2/...`).

> Build a ChIMES potential for Cu-Zr metallic glasses, liquid and amorphous,
> 300-2000 K, for ~50,000-atom quenches. Label with Quantum ESPRESSO on
> Dane, bank pls2. Study dir /p/lustre2/me/studies/cuzr.

Claude creates the study (`chimes-agent study --init`), writes the plan into
`STUDY.md`, and starts the data phase.

## What each phase brings back to you

| Phase | Agent | You're typically asked | You receive |
|---|---|---|---|
| Data | `chimes-data-curator` | keep open-data labels or relabel with QE? approve the QE job | `01_data/DATA_PLAN.md`, curated train/holdout, coverage per element pair |
| Hyperparameters | `chimes-hyperparameter-tuner` | approve the search job (fits, cores, walltime) | chosen cutoffs/λ/orders/smoothing/α with reasons; cross-validated errors by composition; sensitivity profiles; a learning-curve verdict (more data would / would not help); `HYPER_REPORT.md` |
| Model check | `chimes-fit-reviewer` | nothing | an independent verdict: trustworthy / caveats / do not use |
| MD validation | `chimes-md-validator` | which conditions to test; a DFT reference if you have one | stability, close contacts, RDFs, equation of state and elastic constants, coverage of the MD configurations (fingerprint, QUESTS), `MD_REPORT.md` |
| Active learning | `chimes-active-learner` | approve each round's QE job; the labeling budget | per round: the batch and why (`batch.json`), errors on the fixed holdout, `AL_STATUS.md` with a CONVERGED/CONTINUE verdict |
| Benchmark | `chimes-benchmark` | approve the scaling job; your production sizes | cost model, sizing table, CPU-hours used so far |
| Deploy + report | `deploy`, `chimes-report-writer` | audience of the report | `06_deploy/MODEL_CARD.md`, `REPORT.md` |

Agents return a fixed-format status: `DONE`, `NEEDS_DECISION` (a question
with options, numbers and a recommendation), `NEEDS_JOB` (a dry-run job for
you to approve) or `BLOCKED`.

## Approvals

Anything that runs on the cluster is shown as a dry run first: the job
script, cores, queue, walltime, and roughly what it is for. Claude submits
only after you say yes, and `.claude/settings.json` enforces the same rule
at the tool level. Waiting is delegated to a cheap monitoring agent, so the
conversation stays usable while jobs run.

## Steering

You can intervene at any point, for example:

- "Keep 4-body sweeps light."
- "Extend the 3-body cutoffs to 7 Å."
- "Don't exclude cluster types; the holdout is too small."
- "Use force-only scoring."

Hyperparameter searches cache every fit, so re-running with a wider grid
only fits the new points.

## Resuming

Everything lives in the study directory: `study.json` (registry),
`STUDY.md` (decisions), and each phase's outputs. Ask Claude to "resume the
study in /p/lustre2/me/studies/cuzr". It reads the registry, reports which
phases are complete (`chimes-agent study --study DIR`), and continues.

## Reading the results

Start with `REPORT.md`, then `06_deploy/MODEL_CARD.md`. Three numbers to
understand:

- **Holdout relative force error** is force RMSE on held-out frames divided
  by the typical force. 0.1 is very good, 0.3 is a rough model, and 1.0
  predicts nothing. It always comes with a ± from bootstrapping the holdout
  set.
- **Minimum sampled distance per pair** marks the edge of validity. Do not
  run configurations with closer contacts.
- **core-seconds per atom-step** is the runtime cost unit. CPU-hours ≈ unit
  × atoms × steps / 3600.
