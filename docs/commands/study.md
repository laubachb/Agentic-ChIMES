# `chimes-agent study`

**Status: implemented.** Creates a study directory, registers what each phase
produced, and shows a study's status.

```bash
chimes-agent study --init /p/lustre2/$USER/studies/cuzr --name cuzr \
  --elements Cu,Zr --goal "ChIMES potential for Cu-Zr metallic glasses"
chimes-agent study --study /p/lustre2/$USER/studies/cuzr \
  --register data_manifest=.../01_data/curate/data_manifest.json
chimes-agent study --study /p/lustre2/$USER/studies/cuzr     # status
```

`--init` creates the standard layout (`01_data` … `06_deploy`, `usage/`),
`study.json` and a `STUDY.md` for the decision log. Keep studies on a
shared filesystem (`/p/lustre2`) so Slurm jobs can read and write them.

**Registry keys** (`--register key=path`, repeatable): `data_manifest`,
`hyper_report`, `params`, `fm_setup`, `al_run`, `md_runs` (appends),
`benchmark`, `usage`, `deploy`. Unregistered keys fall back to the standard
layout (e.g. `02_fit/search/hyper_report.json`). `--status` output lists
what is `missing`.

**Extra roots** (`--extra-root DIR`): directories outside the study whose
Slurm jobs belong to it, for [`usage`](usage.md).

A `study.json` also switches on CPU-hour bookkeeping: any stage whose
`--output-dir` is inside the study appends its login-node CPU time to
`usage/local.jsonl` (not inside Slurm jobs, which `sacct` accounts for).
