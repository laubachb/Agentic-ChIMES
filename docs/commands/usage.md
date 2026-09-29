# `chimes-agent usage`

**Status: implemented.** How much compute a study used, by phase and job.

```bash
chimes-agent usage --study /p/lustre2/$USER/studies/cuzr
```

- **Slurm jobs** come from `sacct`, selected by working directory: every job
  that ran inside the study (or a registered extra root) counts, including
  jobs al_driver submits itself. No registration needed.
- **Login-node work** comes from `usage/local.jsonl`, written automatically
  by stages that ran inside the study.
- **Phases** are read from the path (`01_data` → data, `02_fit` → fit,
  `03_al`, `04_md`, `05_bench`, `06_deploy`); `--phase-hints
  '{"/path/prefix": "fit"}'` covers other layouts.

## Output

`usage/usage_report.json`, `usage/usage_jobs.csv`, and:

```json
{"total_cpu_hours": 77.7, "total_used_cpu_hours": 1.31, "allocation_efficiency": 0.017,
 "total_local_cpu_hours": 0.0, "n_jobs": 9,
 "by_phase": {"fit": {"jobs": 7, "cpu_hours": 77.5, "used_cpu_hours": 1.31, "node_hours": 0.69}}}
```

- `cpu_hours`: allocated cores × elapsed (sacct `CPUTimeRAW`), which is what
  the allocation is charged.
- `used_cpu_hours`: CPU time the processes actually consumed (sacct
  `TotalCPU` on the job steps).
- `allocation_efficiency` = used ÷ charged. The example above is real: the
  Cu-Zr development study held full 112-core nodes for hyperparameter
  searches whose fits used a few cores' worth of time. `hyper-search
  --machine` now requests only the cores its largest stage can use.
