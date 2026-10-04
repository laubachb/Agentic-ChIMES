# Stages and contracts

## What makes a "stage"

Every `chimes-agent <name>` subcommand is backed by a module in
`src/agentic_chimes/stages/<name>.py` exposing three things:

- `NAME` / `SUMMARY` — the subcommand name and one-line description shown in
  `chimes-agent --help` and `chimes-agent <name> --describe`.
- `SCHEMA` — a JSON Schema describing the stage's input (and, loosely, its
  output) shape. This is what `--describe` prints, and what a coding agent
  should read before constructing a `--json-in` payload.
- `add_arguments(parser)` / `run(args) -> dict` — the argparse wiring and
  the actual work function. `run` takes the parsed `argparse.Namespace`
  (after any `--json-in` overrides have been applied) and returns a
  JSON-serializable dict, which `cli.py` prints to stdout or `--json-out`.

`cli.py` (`build_parser`/`main`) is the only place that knows about the
list of stages (`STAGE_MODULE_NAMES`) and the generic flags every stage
gets for free: `--json-in`, `--json-out`, `--describe`, `--force`,
`--dry-run`, and (unless a module sets `USES_OUTPUT_DIR = False`)
`--output-dir`.

### stdout carries exactly one JSON object

`cli.main` runs each stage with file descriptor 1 redirected to
`<output-dir>/<stage>.log` (a temp file when there is no `--output-dir`),
restoring it only to print the result. This has to happen at the
descriptor level, not by reassigning `sys.stdout`: chimes_calculator's C++
library and al_driver's `print`s write to fd 1 directly, and one
`al-select` call otherwise emitted ~1,600 lines ahead of the JSON, which a
tool-calling agent cannot parse. Consequences for callers:

- Success: the normal result, plus `stage_log` *only if* the stage
  produced output (quiet stages leave no log).
- Failure: `{"error": ..., "log": <path>, "log_tail": <last 20 lines>}`,
  exit code 1. The tail is usually enough to diagnose without opening the log.
- Stderr is untouched (Python warnings, etc.).

Covered by `tests/unit/test_cli_stdout_contract.py`.

## Why JSON in, JSON out

The point is that a stage is equally usable by a human typing flags and by
a coding agent constructing a payload programmatically:

```bash
# human
chimes-agent fm-setup-gen --elements C,H --order '{"2":12,"3":5}' ...

# agent / script
echo '{"elements": ["C","H"], "order": {"2":12,"3":5}, ...}' > in.json
chimes-agent fm-setup-gen --json-in in.json --json-out out.json
```

`--json-in` keys must match the argparse `dest` names for that stage
(visible via `--describe` or by reading the stage module's
`add_arguments`); when both are given, JSON wins.

## Primitive stages do not know about each other

`fm-setup-gen`'s output includes `fm_setup_in`; `amat-build` takes
`fm_setup_in` as input; `solve` takes `amat-build`'s `A`/`b`/`header`/`map`
outputs; `evaluate` takes `solve`'s `params` output. Deciding whether to
re-run `fm-setup-gen` with different cutoffs before building is the
caller's job (human or agent). *Composite* stages exist for the recurring
pipelines (`model-build`, `sweep`, `hyper-search`, `learning-curve`,
`committee`, `al-batch`, `auto-build`): they call the primitives' `run()`
functions in-process (`stages/_compose.py`), never the CLI as a subprocess,
and they report every intermediate result rather than hiding it. Only
`auto-build` makes a choice for the caller. This is a deliberate contrast
with al_driver's `main.py`, which walks a fixed build→solve→MD→QM→post-
process sequence as one long blocking process; see the "Why not al_driver's
loop directly" section below.

## Idempotency: the manifest, not `restart.dat`

al_driver checkpoints progress in a single `restart.dat` flag file per
study, appended to and parsed by substring search
(`codes/al_driver-LLfork/src/restart.py`) — there's no way to ask it "what
were the inputs to the last completed stage" or "did this stage actually
finish, or just get far enough to write some files." Every stage here that
writes files (`USES_OUTPUT_DIR = True`, the default) instead writes
`<output-dir>/manifest.json` (via `stages/_manifest.py`):

```json
{
  "stage": "solve",
  "input_hash": "…sha256 of the JSON-serialized input dict…",
  "status": "done",
  "finished_at": 1234567890.1,
  "outputs": { "...": "..." }
}
```

Re-invoking a stage against the same `--output-dir`:

- **same inputs, `status: done`** → short-circuits, reprints the prior
  `outputs` dict, does no work.
- **different inputs** → refused with a clear error naming the mismatch;
  pass `--force` to overwrite, or use a different `--output-dir`. This is
  the property `restart.dat` doesn't have: it has no notion of "these are
  different parameters," so re-running al_driver's `main.py` at the same
  study path with a changed config silently mixes state from two different
  parameterizations.
- **`--force`** → always re-runs regardless of prior state.
- **`--dry-run`** → records nothing, so the real run that follows with the
  same inputs executes instead of short-circuiting to the preview's result.
- **inputs include file contents**: each input file's size and mtime is
  part of the hash, so regenerating a file in place is noticed (and named).
- **a failed run** never blocks a retry, even with changed inputs.
- **a run still in progress** blocks a second one in the same directory.
  On the same host the recorded PID is checked. A run on another host
  counts as live for 6 hours. `--force` overrides either.
- **a truncated manifest** (a job killed mid-write) counts as absent.

## Crash safety

- Every JSON result and cache is written to a temporary file and renamed
  into place (`io/atomic.py`), so a killed job never leaves half a file.
  Readers treat an unreadable file as missing: a hyper-search point whose
  cache was cut short is refitted, not fatal.
- Directories are created one level at a time, with retries on Lustre's
  transient `EREMOTE` (`io/fs.py`).
- Anything that fails before a stage starts (creating the output
  directory, reading `--json-in`) still returns the one-JSON-object error.
- Submissions record `job.json`; `job-status` turns scheduler state plus the
  expected result files into one verdict.

## Machine-callable discovery

`chimes-agent <stage> --describe` prints `{"stage", "summary",
"uses_output_dir", "schema"}` without running anything — this is how an
agent (or a human) figures out a stage's contract without reading source.

## Phasing

This repo was built incrementally. While a stage was unimplemented it
still registered a real subcommand (`--describe` worked, flags parsed,
`--json-in`/`--json-out` worked) whose `run()` echoed its parsed input
back (`stages/_stub.py`, kept for new stages). The CLI surface a caller
depends on has therefore been stable since Phase 0; every stage is now
implemented.

Build order, and why:

1. **Package scaffold + `chimes-agent setup`** — every later stage that
   shells out to a vendored binary depends on `deps/installed.json`
   existing, so the install/bootstrap subsystem had to come first.
2. **Walking skeleton, no HPC** (`fm-setup-gen` → `amat-build` → `solve`
   [local algorithms] → `evaluate`) — the highest-value, lowest-risk slice:
   it exercises all three vendored repos' core pieces, and is checkable
   against real golden fixtures already in the repos
   (`test_suite-lsq/*/fm_setup.in`, `serial_interface/tests/`) with zero
   HPC dependency. Proving the stage/manifest/JSON-contract abstraction
   here, before writing any Slurm-touching code, meant a wrong abstraction
   would be cheap to fix.
3. **HPC submission layer** (`hpc/slurm.py`, `submit`) — the
   highest-blast-radius part (wrong flags waste allocation), done once the
   surrounding stage contract was already proven stable. The DLARS
   cliff-detection log parser (`stages/_cliff_monitor.py`) is deliberately
   pure log-parsing with zero Slurm dependency, unit-tested against
   synthetic log fixtures — then wired into a *live* `solve --algorithm
   dlars` submission (`stages/_dlars_hpc.py`, `amat-build --machine`).
   Validated against a real Slurm job on Dane, which also caught two real
   bugs a mocked-only test never would have: a `walltime_hours` float
   reaching `sbatch -t` unconverted (invalid Slurm time syntax), and a
   `normalize=true` default that `dlars` itself warns against and that
   caused an immediate real failure the cliff monitor correctly cancelled
   — see `docs/commands/solve.md` and
   `docs/concepts/machine_profiles.md#shared-filesystem-required-for-real-hpc-submissions`.
4. **Dataset tooling + LAMMPS + sweep + model-build** — `dataset-select`
   (FPS/random/stratified), `lammps-run` (validated three ways: standalone
   `chimescalc` binary, ctypes evaluator, and LAMMPS all agree on the same
   published reference to ~1e-3), `sweep` (2b/3b/4b order × cutoffs ×
   alpha/algorithm grid), and `model-build` (amat-build+solve composed).
5. **QE relabeling** — `qe-relabel` + `converters/qe2xyzf.py`, implemented
   directly rather than through an abstract QM-driver registry (QE was the
   only new code being added; see `docs/concepts/qm_driver_plugins.md` for
   the registry design kept on file for *if* a second QM code joins later).
6. **`al-run`** — launches al_driver's own `main.py` as a detached
   background process (see below), rather than reimplementing its
   orchestration. **`al-select`** — a standalone wrapper around al_driver's
   own `gen_subset` Metropolis-MC diversity selector, imported and called
   in-process (not reimplemented), for use outside a full driver cycle;
   see `docs/commands/al-select.md`. This was the last remaining stub and
   is now implemented.
7. **Data phase** — `data-search`, `data-fetch`, `data-generate`,
   `data-curate` (+ `qe-relabel --kspacing` and provenance), driven by the
   `chimes-data-curator` subagent; see `docs/concepts/data_curation.md`.
8. **Hyperparameter phase** — `hyper-analyze` and `hyper-search`, driven by
   the `chimes-hyperparameter-tuner` subagent. Building it surfaced bugs in
   earlier stages, all fixed:
   - `evaluate` mixed force units;
   - the calculator mishandled cells thinner than the cutoff;
   - chimes_lsq crashed on mixed cell headers;
   - pair analysis dropped self-image neighbours, so 1-atom crystals looked
     like isolated atoms;
   - local MPI binaries crashed inside Slurm steps;
   - a dry run blocked the real run's idempotency check.
9. **Study, benchmark and report phase**: `study` (registry), `usage`
   (CPU-hours via sacct + login-node ledger), `benchmark` (LAMMPS strong/weak
   scaling → cost model), `deploy` (model card), `study-report`, driven by the
   `chimes-benchmark` and `chimes-report-writer` subagents. Documentation was
   reorganized around using the toolkit (User guide).
10. **Validation and active-learning tooling** — `md-check`, `eos-check`,
    `fingerprint` (native, with the element-aware metric), `quests`,
    `committee`, `learning-curve`, `weights`, `hierarch`, `al-batch`,
    `al-merge`, `al-status`, `qe-converge`, `job-status`, `doctor`; cross-
    validation, smoothing/α/stress stages and a generated report in
    `hyper-search`; plots throughout and a fuller `study-report`. Driven by
    the `chimes-md-validator` and `chimes-active-learner` subagents. See the
    CHANGELOG for what each pass found.

## Why not al_driver's loop directly — and where `al-run` fits

al_driver's active-learning loop is real, working orchestration logic —
`gen_ff.py`, `qm_driver.py`, `run_md.py`, `gen_selections.py` are reused
(wrapped, not rewritten) by `hpc/slurm.py` and (via `al-run`) `main.py`
itself. What it isn't, on its own, is agent-callable: `main.py` is a
single ~1300-line function that blocks for days, polls `squeue` with
`time.sleep(60)`, and has no JSON output anywhere — only prints and flat
files. The discrete, structured-I/O stages in this repo are what let a
coding agent drive "build a ChIMES model" one decision at a time, and
that's still the primary interface — this repo does not build a *second*
autonomous loop layered on top of them.

`al-run` is not that second loop. It's a thin, correct wrapper for the
case where a human or agent has already decided to run al_driver's real
loop as-is (an explicit, deliberate choice, not something any other stage
triggers automatically) — it launches `main.py` detached and returns
immediately, rather than blocking the caller for days. The discrete stages
remain how you'd build and iterate on a single model; `al-run` is how you
kick off a full multi-cycle active-learning campaign once you've decided
that's what you want, using al_driver's own tested orchestration for it
rather than a reimplementation.
