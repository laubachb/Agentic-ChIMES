# `chimes-agent submit`

**Status: implemented.**

Generic Slurm submit/status/cancel/dry-run on top of `hpc/slurm.py` (which
wraps al_driver's own `create_and_launch_job`/`wait_for_job(s)`). Other
HPC-submitting stages call `hpc.submit_job(...)` directly rather than
shelling out to this subcommand; it's also exposed standalone for ad hoc
use.

## Usage

```bash
# preview an sbatch script without submitting (spends no allocation)
chimes-agent submit --machine dane --job-name test \
  --command "srun -n 112 my_program" \
  --nodes 1 --walltime-hours 1 --queue debug \
  --output-dir ./job1 --dry-run

# actually submit
chimes-agent submit --machine dane --job-name test \
  --command "srun -n 112 my_program" \
  --nodes 1 --walltime-hours 1 --queue debug \
  --output-dir ./job1

# check on it
chimes-agent submit --status-of 1234567

# cancel it
chimes-agent submit --cancel 1234567
```

## Flags

- `--machine {dane,stampede3,<path>}` (required unless `--status-of`/`--cancel`)
- `--job-name NAME`
- `--command CMD` (repeatable, in order — the job body)
- `--nodes N` (default 1)
- `--ntasks-per-node N` (default: the machine profile's
  `default_ntasks_per_node` — see the Dane gotcha in
  `docs/concepts/machine_profiles.md`)
- `--walltime-hours H` (default 1.0)
- `--queue {debug,batch}` (default `debug`; translated to the machine's
  real partition string)
- `--output-dir DIR` (required for a real/dry-run submission — this is the
  job's `work_dir`)
- `--dry-run` (generic flag) — render the sbatch script, don't submit
- `--status-of JOB_ID` — check queue status instead of submitting
- `--cancel JOB_ID` — cancel instead of submitting

## Output

```json
{
  "job_id": "1234567",
  "job_name": "test",
  "work_dir": "./job1",
  "job_file": "./job1/run.cmd",
  "dry_run": false,
  "machine": "dane",
  "queue": "pdebug"
}
```
