# Machine profiles

Every HPC-touching stage (`setup`, `submit`, and later `amat-build --hpc`,
`solve --algorithm dlars`, `qe-relabel`, `lammps-run`) takes a machine
profile: a declarative YAML describing one cluster's account, partitions,
module set, and defaults. Built-in profiles live in
`src/agentic_chimes/machines/profiles/*.yaml`; `chimes-agent <stage> --hpc
<name>` resolves a built-in by name first, then falls back to treating the
argument as a path to your own YAML file — so adding a new cluster never
requires touching package code.

## Schema

```yaml
name: dane                     # required
job_system: slurm              # required: slurm | conda-slurm | TACC | UM-ARC | torque
                                #   (torque submission works upstream; completion polling does not -- see helpers.wait_for_job)
launcher: srun                 # required: srun | ibrun | ...  (informational; stages that shell out to
                                #   an MPI program read this to build their own command line)
account: pls2                  # required: passed as sbatch -A
hosttype: LLNL-LC              # required: which codes/*/modfiles/<hosttype>.mod the vendored
                                #   forks' own install.sh scripts source (see `chimes-agent setup`)
partitions:                    # required: logical name -> real partition/queue string
  debug: pdebug
  batch: pbatch
default_ntasks_per_node: 112   # required: see "the Dane gotcha" below
require_ntasks_per_node_or_exclusive: true   # default true; refuses to submit/render without one
poll_interval_s: 60            # default 60, matches al_driver's own squeue-poll cadence
modules:                       # module load list, sourced before any build/submit
  - intel-classic/2021.6.0
  - mvapich2/2.3.7
conda_env: mat_mcts            # optional; `conda activate <env>` added to rendered job scripts
filesystem:                    # optional, informational for now (docs/tooling, not enforced)
  scratch_root: /p/lustre2/${USER}
  max_small_files_per_dir: 5000
  bulk_archive_root: /p/lustre3/${USER}
qe:                             # optional, Quantum ESPRESSO build/run settings for this machine
  configure_extra_args: []      # extra flags for QE's own ./configure at `chimes-agent setup`
```

## The Dane gotcha this profile exists to close

A bare `sbatch -N 1` on Dane allocates **one CPU and about 2.3GB**, not a
full node — every parallel job (multiprocessing, `xargs -P`, `srun`)
submitted that way silently crawls or gets OOM-killed. `hpc/slurm.py` and
`hpc/dry_run.py` always fill in `default_ntasks_per_node` when a caller
doesn't override it explicitly, and `require_ntasks_per_node_or_exclusive`
refuses to render or submit a job without one — this is enforced by
default, not something you have to remember to pass. See
`tests/unit/test_dry_run.py` for the regression test.

## Adding a new machine

1. Copy `src/agentic_chimes/machines/profiles/dane.yaml` (or `stampede3.yaml`
   for a TACC-style system) to a new file, anywhere — it doesn't need to
   live inside the package.
2. Fill in the required keys above for your cluster.
3. Use it with `--hpc /path/to/your_cluster.yaml` on any stage that takes
   `--hpc`, or `chimes-agent setup --machine /path/to/your_cluster.yaml`.

If you use it often enough to want a short name, add it under
`src/agentic_chimes/machines/profiles/` as `<name>.yaml` and it becomes a
built-in (`--hpc <name>`).

## What `hpc/slurm.py` actually does

`submit_job(profile, ...)` is a thin wrapper around al_driver's own,
already-working `create_and_launch_job`/`wait_for_job(s)`
(`codes/al_driver-LLfork/src/helpers.py`), imported from its vendored
location rather than copied. The wrapper's only added value: filling in
`ntasks_per_node` from the profile default, translating a logical queue
name (`"debug"`/`"batch"`) to the machine's real partition string via
`profile.queue_for(...)`, and `dry_run=True` rendering the sbatch script
(via `hpc/dry_run.py`) instead of calling `sbatch`. It does not
reimplement Slurm submission or job polling.

## Shared filesystem required for real HPC submissions

`--output-dir` (or `work_dir`) for any stage that submits a **real**
(non-dry-run) Slurm job must be on a filesystem the compute nodes can see
— `/p/lustre2/...`, not a login-node-local path like `/tmp` or a
per-session scratch directory under it. This isn't a `chimes-agent`
restriction, it's how the cluster is built: a compute node has its own
local `/tmp`, physically separate from the login node's. A job submitted
with `--output-dir` under `/tmp` will still show as `COMPLETED` in
`sacct` (Slurm itself doesn't know or care what the job's commands did to
files), but every file the job reads or writes lands on the *compute
node's* local disk, invisible from wherever you're checking — the job
silently runs against missing input and produces no visible output.

This was found the hard way validating `solve --algorithm dlars`'s real
HPC path (see `docs/commands/solve.md`): a real submission with
`--output-dir` under a `/tmp` scratchpad path came back `COMPLETED`
with zero output files anywhere reachable; the same submission against a
`/p/lustre2/...` path produced `stdoutmsg`, module-load output, and
command output exactly as expected. `--dry-run` previews render correctly
either way (no filesystem access needed to just write a script file
locally), so this only bites on a real submission — always point any
stage's `--output-dir` at shared storage before dropping `--dry-run`.

## A Slurm walltime gotcha already closed here

`sbatch -t` wants `HH:MM:SS` (or a similarly qualified format) — a bare
decimal hour count like `1.5` is not valid Slurm time syntax (a bare
number is parsed as *minutes*, and the decimal point is rejected
outright). Every `walltime_hours` value passed through this repo's
`hpc/slurm.py`/`hpc/dry_run.py` is converted via
`hpc.dry_run.hours_to_slurm_time()` before it ever reaches `sbatch` — this
was a real, previously-undetected bug (every earlier `--dry-run` preview
looked fine since nothing validated the `-t` value's actual Slurm
validity) until the first real submission surfaced it; see
`tests/unit/test_dry_run.py::test_rendered_script_never_has_a_bare_decimal_walltime`
for the regression test. You don't need to do anything to get this right
— every stage's `--walltime-hours` flag already goes through the fixed
path — this section exists so a future direct caller of `hpc.submit_job`
knows not to bypass it.
