---
name: chimes-hpc-jobs
description: Safely submit and manage Slurm work for ChIMES - QE labeling, chimes_lsq A-matrix builds, DLARS/DLASSO solves, generic sbatch. Use before any --machine / --dry-run / qe-relabel / dlars / submit action, when a job seems stuck or reports COMPLETED with no output, or when lustre file quota or cleanup comes up.
---

# HPC jobs for ChIMES (Slurm, LC machines)

## Before submitting

1. **Shared filesystem.** `--output-dir` must be under `/p/lustre2/$USER/...`.
   `/tmp` and login-node scratch are invisible to compute nodes: the job ends
   `COMPLETED 0:0` with no output. Nothing warns you; check the path.
2. **`--dry-run` first**, read the rendered script: full node
   (`ntasks-per-node 112` on Dane), queue (`debug` = pdebug, short tests;
   `batch` = pbatch), walltime, account. The CLI converts walltime hours to
   `HH:MM:SS`; never hand-write `-t 1.5`.
3. **Tell the user** what will be submitted (nodes, queue, walltime,
   rough cost) before a real submission. Submitting is visible to shared
   resources and consumes allocation.
4. Real submissions block. Run in the background and delegate the waiting
   to the `chimes-job-monitor` subagent rather than polling yourself.

## DLARS / DLASSO solves (`solve`, `model-build`, `sweep` with `--machine`)

- Needs `--machine`, `--dim` (dim.txt), and A/b either as split files or
  named A.txt/b.txt (the stage symlinks canonical names for you).
- Leave `--normalize` false (default). True fails instantly with
  `Intel MKL ERROR: Parameter 7 ... cblas_dgemv` and floods the disk.
- **Cliff monitor:** DLARS can hit an ill-conditioning cliff near the end
  of its path. The stage watches `dlars.log`; on a detected cliff it cancels
  the job and re-runs a short capped `dlars --iterations=<n>` to finalize
  from the restart checkpoint. The result JSON says whether that happened.
  This is expected behavior, not a failure — report it, don't alarm.
- A solve that never produces `params.txt` raises; read `log_tail`.

## If a job looks wrong

Start with `chimes-agent job-status --work-dir <dir>`. Every submission
records `job.json` with the result files it should produce, and the stage
returns `SUCCEEDED`, `FAILED` (with reason and fix) or
`COMPLETED_WITHOUT_RESULTS`.

- `squeue -u $USER`, `sacct -j <id> --format=JobID,State,Elapsed,ExitCode`.
- COMPLETED but no files: filesystem issue above.
- Stuck at iteration 0 with a huge `dlars.log`: `scancel` it and delete the
  log immediately (file count / disk), then check `--normalize`.
- Cancel only jobs you launched this session, and say so. Never `scancel -u`.

## Lustre etiquette (quota is file COUNT, ~1.05M on lustre2)

- One packed `.xyzf` per dataset, not a file per frame; QE relabeling
  creates one work dir per frame, so collect and then delete or tar them.
- After a study: delete `traj.txt`, `dlars.log`, restart files you do not
  need; move finished archives to `/p/lustre3/$USER`.
- `lfs quota -u $USER /p/lustre2` shows the count.

## Testing HPC changes

Use pdebug with a trivial job first (`submit`), on `/p/lustre2/$USER/...`,
and remove the test directory afterwards.

Reference: `docs/concepts/machine_profiles.md`, `docs/commands/solve.md`,
`docs/commands/submit.md`, `docs/commands/qe-relabel.md`.
