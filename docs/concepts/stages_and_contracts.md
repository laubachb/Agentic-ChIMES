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

## No stage calls another stage

`fm-setup-gen`'s output includes `fm_setup_in`; `amat-build` takes
`fm_setup_in` as input; `solve` takes `amat-build`'s `A`/`b`/`header`/`map`
outputs; `evaluate` takes `solve`'s `params` output. Composing them —
deciding whether to re-run `fm-setup-gen` with different cutoffs before
building, or to run `sweep` instead of a single `solve` — is the caller's
job (human or agent), not baked into the stages themselves. This is a
deliberate contrast with al_driver's `main.py`, which walks a fixed
build→solve→MD→QM→post-process sequence as one long blocking process; see
the "Why not al_driver's loop directly" section below.

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

## Machine-callable discovery

`chimes-agent <stage> --describe` prints `{"stage", "summary",
"uses_output_dir", "schema"}` without running anything — this is how an
agent (or a human) figures out a stage's contract without reading source.

## Phasing

This repo was built incrementally. Stages not yet implemented still
register a real subcommand (`--describe` works, flags parse,
`--json-in`/`--json-out` work) whose `run()` just echoes its parsed input
back — see `stages/_stub.py`. This means the CLI surface a caller (human
or agent) depends on has been stable since Phase 0; swapping a stub's
`run()` for real logic never changes how the stage is invoked.

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
   is now implemented, closing out this phasing list.

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
