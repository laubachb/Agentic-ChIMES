# `chimes-agent job-status`

**Status: implemented.** Read-only.

One verdict for a submitted job: whether it finished, whether it worked,
and if not, why and what to do.

```bash
chimes-agent job-status --work-dir 02_fit/search      # the directory a stage submitted from
chimes-agent job-status --job-id 7850407 --expect path/to/result.json
```

Every submission writes `job.json` into its work directory: the job id,
walltime, and the result files a successful run leaves behind.

| stage | expected results |
|---|---|
| `hyper-search --machine` | `search/hyper_report.json` |
| `md-check --machine` | `run/md_check.json` |
| `fingerprint --machine` | `run/fingerprint.json` |
| `benchmark` | each case's `log.lammps` |

`job-status` combines that record with `squeue` / `sacct`, checks that each
expected result exists, is non-empty and (for JSON) parses, and tails
`stdoutmsg` and `*.out`. Re-submitting into the same directory keeps the
earlier job ids under `history`.

| verdict | meaning | typical fix |
|---|---|---|
| `QUEUED` / `RUNNING` | in the scheduler (`elapsed`, `time_left`, pending `reason`) | wait |
| `SUCCEEDED` | finished; every expected result exists and parses | read the results |
| `COMPLETED_WITHOUT_RESULTS` | Slurm says COMPLETED, results missing or truncated | read `log_tail`; a job directory under `/tmp` writes to the compute node's own disk |
| `FAILED` | `reason`: TIMEOUT, OUT_OF_MEMORY, NODE_FAIL, CANCELLED, or a nonzero exit | `fix` says what to change (e.g. more walltime: hyper-search resumes from its point cache) |
| `UNKNOWN` | no accounting record and no results | check the directory by hand |
