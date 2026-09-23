# Changelog

## Unreleased — dataset/QE/sweep/LAMMPS/active-learning iteration

- **`model-build`** (new): amat-build then solve composed into one call.
- **`dataset-select`**: real FPS (greedy farthest-point sampling, composition
  and/or per-atom-energy descriptor, z-score standardized) / random /
  stratified-holdout (by composition class, every class represented in
  both splits) implementations, replacing the echo stub. 8 unit tests
  against a synthetic pool (disjointness/completeness, determinism,
  descriptor coverage, error handling) -- no `codes/` dependency, runs in CI.
- **`sweep`**: real grid-sweep engine composing fm-setup-gen + model-build +
  evaluate, with independently sweepable `order_2b`/`order_3b`/`order_4b`
  (a `null`/`0` 4-body entry omits the term, matching fm-setup-gen's own
  convention), `default_s_minim`/`default_s_maxim`, `alpha`, and
  `algorithm`. Writes a CSV + `best_by` pointers (not auto-tuning -- a
  table to look at). Validated with a real 4-point grid against the same
  bundled fixture the quickstart uses.
- **`lammps-run`**: real single-point/MD implementation
  (`io/lammps_data.py`'s data-file writer + LAMMPS input templating +
  dump/thermo-log parsing). Cross-validated three ways against a published
  reference force field: the standalone `chimescalc` binary, the ctypes
  evaluator, and LAMMPS itself all agree to ~1e-3 on the same
  energy/forces -- a genuine correctness check, not just "it ran."
- **`qe-relabel`** (new, was a stub): `converters/qe2xyzf.py` parses real
  `pw.x` stdout (energy, forces) with correct Rydberg->kcal/mol and
  Ry/bohr->hartree/bohr unit conversions (new `RY_TO_EV`/`RY_TO_HARTREE`
  constants in `converters/units.py`), validated against a hand-crafted
  realistic fixture with independently hand-computed expected values.
  `stages/qe_relabel.py` generates `pw.in`, submits one combined Slurm job
  across selected frames (dry-run validated, including the Dane
  `--ntasks-per-node` guardrail), and `--collect` merges finished output
  back onto the original structure into a labeled `.xyzf`, reporting
  converged/not-converged/missing/parse-failed per frame. Scope: 
  single-point SCF only (no relax/vc-relax geometry re-parsing).
- **`al-run`** (new): launches al_driver's own `main.py` as a properly
  detached (session-leader, nohup-style) background process, with
  `--status-of`/`--stop` PID management -- does not reimplement AL
  orchestration, correctly calls the existing driver. Deliberately minimal
  scope: expects an already-prepared al_driver study (`ALL_BASE_FILES/` +
  `config.py`, as al_driver's own examples show); does not generate
  `config.py` from scratch. `docs/concepts/qm_driver_plugins.md` updated
  to reflect that the originally-planned QM-driver registry was not needed
  for QE (implemented directly instead) and is kept on file only for if a
  second QM code is added later.
- 28 new unit tests across `test_dataset_select.py`, `test_lammps_run.py`,
  `test_qe2xyzf.py`, `test_qe_relabel.py`, `test_al_run.py` (52 total, up
  from 19) -- all pass, including two real end-to-end integration tests
  (evaluate + LAMMPS) against a published reference force field, and one
  real dry-run + synthetic-collect round trip for qe-relabel.
- Found and fixed two real bugs during this pass: `qe-relabel`'s dry-run
  path required `pw.x` to already be built (defeats the point of
  dry-run) and duplicated the machine profile's `module load` line
  (`hpc.submit_job` already adds it) -- both fixed and now regression-tested.

## 0.1.0 — initial scaffold (Phases 0–1, partial 2)

- Repo scaffold: `src/agentic_chimes/` package, `chimes-agent` CLI entry
  point, per-stage subcommand registry with a uniform JSON-in/JSON-out
  contract, `--describe`, and a per-stage `manifest.json` idempotency
  mechanism (`stages/_manifest.py`) replacing al_driver's `restart.dat`.
- **`codes/` is gitignored, cloned fresh by `chimes-agent setup`**
  (`setup/clone_codes.py`), never pushed to this repo: `al_driver-LLfork`,
  `chimes_lsq-LLfork`, `chimes_calculator-LLfork` are cloned from their real
  `LindseyLab-umich` GitHub remotes at pinned commits, kept pristine (no
  modifications). (Superseded an earlier attempt to track them as git
  submodules — that would have pushed gitlinks referencing the lab's
  private forks to this repo's own remote, which is exactly what should not
  happen.) See `docs/concepts/vendored_forks.md`.
- **`chimes-agent setup`**: clones the three vendored forks (always, as its
  first action, regardless of requested component), then builds/fetches
  `chimes_lsq` + `dlars`,
  `chimes_calculator`'s serial/ctypes evaluator, the ChIMES-patched LAMMPS
  build, and a from-scratch Quantum ESPRESSO clone+build (QE had zero
  support anywhere in the vendored repos before this). Machine-profile
  aware, idempotent, records resolved paths in `deps/installed.json`.
  Validated with real builds on LLNL Dane (`chimes_lsq`, `chimes_calculator`).
- **Machine profiles** (`dane`, `stampede3`): declarative YAML, pluggable
  via `--hpc <name-or-path>`; `hpc/slurm.py` wraps al_driver's own
  `create_and_launch_job`/`wait_for_job(s)` and always enforces an explicit
  `--ntasks-per-node` (closing the Dane "-N 1 gives 1 CPU" gotcha by
  default); `hpc/dry_run.py` renders sbatch scripts without submitting.
- **`fm-setup-gen`**: generates `fm_setup.in` from typed parameters;
  `io/fm_setup.py`'s parser/renderer round-trips the real golden fixtures
  in `codes/chimes_lsq-LLfork/test_suite-lsq/`.
- **`amat-build`**: subprocesses the `chimes_lsq` binary (local only so far).
- **`solve`**: subprocesses `chimes_lsq.py` for local algorithms
  (svd/ridge/lassolars/...); `dlars`/`dlasso` raise a clear
  not-implemented-yet error pending the HPC layer.
- **`evaluate`**: in-process holdout force/energy RMSE via
  `chimescalc_serial_py.py`'s ctypes API, with multi-model committee-spread
  support; validated against `chimes_calculator`'s own published
  force-field test fixtures to ~1e-3.
- **`submit`**: generic Slurm submit/status/cancel/dry-run.
- **`stages/_cliff_monitor.py`**: formalizes the hand-validated DLARS
  ill-conditioning "cliff" detection + finalize-from-restart trick as a
  pure, unit-tested log parser, ahead of being wired into a live
  `solve --algorithm dlars` HPC submission.
- Stubs (real subcommand + schema, echo-only `run()`) for `dataset-select`,
  `qe-relabel`, `sweep`, `lammps-run`, `al-select` — planned in later
  phases; see `docs/concepts/stages_and_contracts.md`.
- Full documentation set: top-level README, `docs/concepts/*`,
  `docs/commands/*` (one page per subcommand), and a worked
  `docs/tutorials/end_to_end_holdout_study.md`.
- **Documentation site**: `mkdocs.yml` (MkDocs + Material theme) compiles
  `docs/` into a real, searchable static site — validated locally with
  `mkdocs build --strict` (clean, zero warnings) and `mkdocs serve` (dev
  server, confirmed serving the homepage, a command page, and the search
  index). `.github/workflows/docs.yml` deploys it to GitHub Pages on push
  to `master` via `actions/deploy-pages` (needs a one-time
  Settings → Pages → Source → GitHub Actions toggle in the repo).
- `tests/unit/`: 19 tests, no HPC/allocation required (fm_setup round-trip,
  cliff-monitor synthetic logs, dry-run sbatch guardrails, manifest
  collision/short-circuit/force semantics, evaluate against real
  published fixtures — the last two skip cleanly if the relevant
  component hasn't been built yet).
- `.github/workflows/ci.yml`: runs the local test tier on push.

### Known gaps (see README's command status table and docs/concepts/stages_and_contracts.md#phasing)

- `dlars`/`dlasso` solve path not yet wired to HPC submission.
- `amat-build`/`solve` have no `--hpc` path yet (local execution only).
- QE driver (`qm_drivers/`), `qe-relabel`, dataset selection, LAMMPS
  running, hyperparameter sweep, and AL batch selection are stubs.
