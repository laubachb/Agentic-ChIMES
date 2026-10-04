# `chimes-agent study`

**Status: implemented.** Creates a study directory, registers what each phase
produced, and shows a study's status.

```bash
chimes-agent study --init /p/lustre2/$USER/studies/cuzr --name cuzr \
  --elements Cu,Zr --goal "ChIMES potential for Cu-Zr metallic glasses"
chimes-agent study --study /p/lustre2/$USER/studies/cuzr \
  --register data_manifest=.../01_data/curate/data_manifest.json
chimes-agent study --study /p/lustre2/$USER/studies/cuzr     # registry: what is present, what is missing
chimes-agent study --study /p/lustre2/$USER/studies/cuzr --status   # one-screen dashboard, also written to STATUS.md
```

## Dashboard (`--status`)

`dashboard_text` (and `<study>/STATUS.md`) show, in one screen:

- **Phases**: done / partial / not started for data, hyperparameters,
  model, MD validation, active learning, benchmark, deploy and report, each
  with one headline number read from its artifact (frames, cross-validated
  force error, MD stability, the `al-status` verdict, cost per atom-step),
  and the next phase with who runs it.
- **Jobs**: every submission recorded under the study (and its extra
  roots), with one `squeue` call: queued, running, finished, or left the
  queue without its result files.
- **Waiting on you**: job scripts rendered by a dry run and never
  submitted, and jobs that ended without results (with the `job-status`
  command that says why).
- **CPU-hours**: Slurm charged and used (from the last `usage` run) and
  login-node CPU time.

`dashboard` carries the same as JSON. It reads small files only and does
not call `sacct`; run [`usage`](usage.md) to refresh the Slurm totals.

`--init` creates the standard layout (`01_data` … `06_deploy`, `usage/`),
`study.json` and a `STUDY.md` for the decision log. Keep studies on a
shared filesystem (`/p/lustre2`) so Slurm jobs can read and write them.

**Registry keys** (`--register key=path`, repeatable): `data_manifest`,
`hyper_report`, `params`, `fm_setup`, `al_run`, `md_runs` (appends),
`benchmark`, `usage`, `deploy`, and the validation artifacts `md_check`,
`eos_check`, `fingerprint`, `quests`, `committee`, `learning_curve`,
`evaluate` (a directory from `evaluate --plot`) and `al_status`. Each may be
the JSON file or the stage's output directory; `study-report` finds the
JSON and every PNG beneath it. Unregistered keys fall back to the standard
layout (e.g. `02_fit/search/hyper_report.json`). `study --study DIR` with
no other flag lists what is present and what is `missing`.

**Extra roots** (`--extra-root DIR`): directories outside the study whose
Slurm jobs belong to it, for [`usage`](usage.md).

A `study.json` also switches on CPU-hour bookkeeping: any stage whose
`--output-dir` is inside the study appends its login-node CPU time to
`usage/local.jsonl` (not inside Slurm jobs, which `sacct` accounts for).
