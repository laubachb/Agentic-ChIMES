# `chimes-agent amat-build`

**Status: implemented**, local or via `--machine`.

Builds `A.txt`/`b.txt`/`dim.txt` (and friends) by subprocessing the
`chimes_lsq` C++ binary against an `fm_setup.in` whose `TRJFILE` already
points at a labeled `.xyzf`. The binary's CLI is exactly `chimes_lsq
<fm_setup.in>` (confirmed against `chimes_lsq.C`'s `argc != 2` check); all
output lands in the process's working directory.

## Usage

```bash
# local (fine for small/medium bases)
chimes-agent amat-build --fm-setup-in ./run1/fm_setup.in --output-dir ./run1

# via Slurm (chimes_lsq is MPI-capable; needed for SPLITFI-true / DLARS-bound runs)
chimes-agent amat-build --fm-setup-in ./run1/fm_setup.in \
  --machine dane --queue batch --walltime-hours 1 --nodes 1 --ntasks-per-node 112 \
  --output-dir ./run1
```

If `--output-dir` is omitted, output lands next to the `fm_setup.in` file.
**When submitting to HPC, `--output-dir` must be on a shared filesystem**
(e.g. `/p/lustre2/...`), never a login-node-local path like `/tmp` -- see
[Machine profiles: shared filesystem required for real jobs](../concepts/machine_profiles.md#shared-filesystem-required-for-real-hpc-submissions).

## Flags

- `--fm-setup-in PATH` (required)
- `--chimes-lsq-bin PATH` — override the resolved `chimes_lsq` binary
  (default: resolved from `deps/installed.json` / `AGENTIC_CHIMES_LSQ_BIN`,
  see `chimes-agent setup`)
- `--machine {dane,stampede3,<path>}` — if given, submits via
  `<launcher> [-n <cores>] chimes_lsq <fm_setup.in>` and blocks until it
  completes (`hpc.poll_job`); omit for local execution
- `--queue`, `--walltime-hours`, `--nodes`, `--ntasks-per-node` (HPC only)
- `--dry-run` (generic flag, HPC only) — render the sbatch script, don't submit

## Output

```json
{
  "work_dir": "./run1",
  "log": "./run1/fm_setup.log",
  "split": false,
  "missing_outputs": [],
  "A": "./run1/A.txt",
  "b": "./run1/b.txt",
  "b_labeled": "./run1/b-labeled.txt",
  "dim": "./run1/dim.txt",
  "natoms": "./run1/natoms.txt",
  "params_header": "./run1/params.header",
  "ff_groups_map": "./run1/ff_groups.map"
}
```

`"split": true` means `fm_setup.in` had `SPLITFI true` and output landed as
`A.0000.txt`/`b.0000.txt`/... instead (required for the DLARS solve path);
in that case the named keys above won't all be present.
