# Machine profiles

Every stage that submits Slurm work takes `--machine <profile>`. That
includes:

- `setup`, `submit`, `qe-relabel`
- `amat-build`, `solve` (DLARS), `model-build`, `sweep`
- `hyper-search`, `md-check`, `benchmark`

A profile is a small YAML file describing one cluster: its account,
partitions, modules and defaults. Built-in profiles live in
`src/agentic_chimes/machines/profiles/*.yaml`. `--machine` takes a built-in
name (`dane`, `stampede3`) or a path to your own YAML file, so adding a
cluster never requires touching package code.

## Accounts and paths come from your environment

Every string in a profile may use `${VAR}` or `${VAR:-default}`, expanded
when the profile loads. The shipped profiles use this so they carry no
personal settings:

```bash
export CHIMES_ACCOUNT=<your Slurm bank / TACC allocation>
```

| profile | `account` | `scratch_root` |
|---|---|---|
| `dane` | `${CHIMES_ACCOUNT:-pls2}` | `/p/lustre2/${USER}` |
| `stampede3` | `${CHIMES_ACCOUNT}` (required) | `${SCRATCH}` |
| `generic` | `${CHIMES_ACCOUNT}` (required) | `${CHIMES_SCRATCH}` |

`generic` is for any Slurm cluster without a bundled profile, and for CI.
It builds the forks with the compilers and MPI already on your PATH
(`hosttype: none`: the forks' `install.sh` scripts run without a
`modfiles/*.mod`). Partitions and cores come from `CHIMES_DEBUG_PARTITION`,
`CHIMES_BATCH_PARTITION` and `CHIMES_CORES_PER_NODE` (defaults debug,
batch, 16). Load a compiler + MPI stack, then run
`chimes-agent setup --machine generic` and `chimes-agent doctor --machine generic`.

A profile whose account is empty refuses to render a job and tells you to
set `CHIMES_ACCOUNT`.

## Schema

```yaml
name: dane                     # required
job_system: slurm              # required: slurm | TACC | UM-ARC (sbatch) or torque (qsub)
launcher: srun                 # required: srun | ibrun | ...
account: ${CHIMES_ACCOUNT:-pls2}   # required: passed as sbatch -A
hosttype: LLNL-LC              # required: which codes/*/modfiles/<hosttype>.mod the forks' install scripts use
partitions:                    # required: logical name -> real partition
  debug: pdebug
  batch: pbatch
default_ntasks_per_node: 112   # required: see "the Dane gotcha" below
require_ntasks_per_node_or_exclusive: true
poll_interval_s: 60
modules:                       # `module load` line in every job script
  - intel-classic/2021.6.0
  - mvapich2/2.3.7
# conda_env: <name>            # optional, see "Which Python a job runs"
filesystem:
  scratch_root: /p/lustre2/${USER}
  max_small_files_per_dir: 5000
  bulk_archive_root: /p/lustre3/${USER}
qe:
  configure_extra_args: []     # extra flags for QE's ./configure at `chimes-agent setup`
```

## Adding a new machine

1. Let the toolkit write it:
   `chimes-agent setup --init-profile ./my_cluster.yaml --scratch <shared dir> --modules <compiler>,<mpi>`.
   It reads partitions, cores per node and your accounts from the scheduler
   and lists what it could not determine under `todo`. (Or copy `dane.yaml`
   or `stampede3.yaml` by hand.)
2. Check the required keys; the account is `${CHIMES_ACCOUNT:-…}` so the
   file can be shared.
3. Pass the path: `chimes-agent setup --machine ./my_cluster.yaml`, then
   `--machine ./my_cluster.yaml` on any submitting stage.
4. To use a short name, add the file under
   `src/agentic_chimes/machines/profiles/<name>.yaml`.

## What a submission does

`hpc/slurm.py` has **one renderer** (`hpc/dry_run.render_sbatch_script`)
for both modes:

- **`--dry-run`** writes the script and stops.
- **A real submission** writes the same script and runs `sbatch` on it
  (`qsub` for torque).

What you approve in the dry run is byte-for-byte what runs
(`tests/unit/test_dry_run.py` checks this). Waiting for jobs still uses
al_driver's `helpers.wait_for_job(s)`.

The script always carries:

- `--ntasks-per-node` (the profile default unless the stage right-sizes
  it, e.g. `hyper-search`, `md-check`);
- `-A <account>`, `-p <partition>`, `-V`, `-o stdoutmsg`;
- the profile's `module load` line;
- `export PATH=<bin dir of the Python that submitted the job>:$PATH`.

### Which Python a job runs

Jobs run the interpreter that submitted them, so compute-node stages see
the same `agentic_chimes` install and packages as the login-node ones. A
profile may still name a `conda_env`. It is activated only if `conda` is
available in the batch shell, before the PATH line, so the submitting
interpreter still comes first.

!!! note "Earlier versions"
    Real submissions used to go through al_driver's `create_and_launch_job`,
    which writes its own script. Dry runs showed a `conda activate <env>`
    line that real jobs never ran, and on Dane the named environment was
    not the one the CLI ran in.

## Shared filesystem required

Job directories must be on storage the compute nodes can see
(`/p/lustre2/...`), never a login node's `/tmp`. A job submitted from `/tmp`
reports COMPLETED in `sacct` but reads and writes the compute node's own
`/tmp`, so nothing comes back. Every submission (dry run included) now
refuses `/tmp`, `/var/tmp`, `/dev/shm` and `$TMPDIR`, with a message
pointing at the profile's `scratch_root`.

## The Dane gotcha

A bare `sbatch -N 1` on Dane allocates **one CPU and about 2.3 GB**, not a
full node. The renderer always writes `--ntasks-per-node`, and
`require_ntasks_per_node_or_exclusive` refuses a script without one.

## Walltime format

`sbatch -t` needs `HH:MM:SS`; a bare `1.5` is read as minutes and the
decimal is rejected. Every `--walltime-hours` goes through
`hpc.dry_run.hours_to_slurm_time()`.
