# Changelog

## Unreleased — `al-select` implemented, closing out the last stub

- **`al-select`** now really wraps al_driver's own Metropolis-MC
  energy-histogram selector (`codes/al_driver-LLfork/src/gen_selections.py:gen_subset`),
  imported and called in-process rather than reimplemented, instead of
  echoing its parsed input. Candidate per-frame energies are predicted
  via the same ctypes evaluator `evaluate` uses and normalized per atom
  (`energy / natoms`), confirmed to match al_driver's own on-disk
  convention against `utilities/new-get_dumb_ener_subjob.sh`'s
  `paste xyzlist.dat xyzlist.energies | awk '{print $NF/$1}'`.
  Deliberately standalone (not a full `ALC-<n>`/`CENTRAL_REPO`
  integration, same scope reasoning as `al-run` not generating
  `config.py`): a new `--central-repo`/`central_repo_out` plain-energies-
  file pair lets you chain diversity across repeated calls yourself. New
  `[al-select]` extra (`matplotlib`/`cycler`, which `gen_selections.py`
  imports unconditionally for its diagnostic plots) added to
  `pyproject.toml` since neither was previously a project dependency.
  Validated for real: a synthetic candidate pool (jittered copies of the
  known-good CHON fixture `test_evaluate.py` already uses) run through
  both the Python API and the actual `chimes-agent al-select` CLI
  entrypoint, confirming a real `gen_subset` Metropolis-MC selection
  (always keeping the observed min/max-energy frames, e.g. `[2, 4, 10,
  11, 13]` out of 20 candidates in one real run) — not just a schema
  round-trip. A separately-tried real ChIMES fixture
  (`test_suite-lsq/h2o-invr`'s `INVRSE_R`-basis `params.txt`) was rejected
  by chimes_calculator's serial C++ parser (`"Incorrect input in
  line...Expect 7 or 8 entries"`), which is why the test fixture uses the
  MORSE-basis CHON params already proven compatible rather than that one.
  This was the one remaining planned-but-stubbed stage; every
  `chimes-agent` subcommand is now implemented.

## Unreleased — DLARS/HPC solve path, validated against a real Slurm job

- **`solve --algorithm dlars/dlasso`** now actually submits to HPC
  (`--machine`), closing the one gap flagged since Phase 2 as "not wired
  up." New `stages/_dlars_hpc.py`: submits `chimes_lsq.py --algorithm
  dlars|dlasso` as a Slurm job, polls the live `dlars.log` into
  `stages/_cliff_monitor.py`'s `CliffMonitor` every `--poll-interval-s`,
  and on a detected cliff cancels the job and runs a short fresh `dlars
  --iterations=<target>` finalize + `chimes_lsq.py --read_output true`,
  exactly formalizing the hand-validated workaround this repo has
  documented since early in the project.
- **`amat-build --machine`**: submits `chimes_lsq` via Slurm (MPI-capable
  binary) for SPLITFI/DLARS-scale runs, instead of local-only.
- **`model-build`/`sweep`/`auto-build`** now pass `machine`/HPC settings
  through to their internal `solve` calls, so a `dlars`/`dlasso` solve is
  reachable from every level of the stack, not just standalone `solve`.
- **Validated against a real Slurm job on Dane pdebug**, not just unit
  tests — and found two real bugs neither the extensive mocked test suite
  nor `--dry-run` previews could have caught, since both only bite a real
  submission:
  - `walltime_hours` (a float, e.g. `1.5`) was stringified directly into
    `sbatch -t`, which is not valid Slurm time syntax (a bare number
    parses as *minutes*, and the decimal point is rejected outright) —
    every real submission before this fix would have gotten a walltime
    far shorter than requested (or rejected outright). Fixed with
    `hpc.dry_run.hours_to_slurm_time()`, used everywhere `-t` is rendered.
  - `--output-dir` pointed at this session's `/tmp` scratchpad: Slurm
    reported the job `COMPLETED` with no visible error, but zero output
    files existed anywhere reachable -- the compute node's local `/tmp` is
    physically separate from the login node's, so the job silently ran
    against missing input in an unreachable directory. Not a code bug, but
    now documented prominently (`docs/concepts/machine_profiles.md`) since
    nothing about `--dry-run` or a mocked test would surface it.
  - Also found (via the real job's actual log output, not guesswork):
    `dlars` itself prints `"Warning: normalize should not be used with
    chimes_lsq"`, and `--normalize true` (the prior default) caused an
    immediate, real MKL error loop that never advanced past iteration 0 --
    the cliff monitor's failure-signature detection correctly caught and
    cancelled it. Default flipped to `--normalize false`; confirmed the
    same input then solved cleanly end to end on a second real submission
    (valid `ENDFILE`-terminated `params.txt`, sane holdout RMSE via
    `evaluate`).
- 9 new unit tests (`test_dlars_hpc.py`, `test_solve_dlars_dispatch.py`,
  `test_amat_build_hpc.py`, plus 3 in `test_dry_run.py`) — 85 total, up
  from 71. The cliff-detection control flow (submit → poll → detect →
  cancel → finalize) is tested against a mocked HPC boundary; the
  detection/cancel half was *also* confirmed against the real
  pathological run above.

## Unreleased — auto-build: the full documented-cutoff-driven pipeline

- **`auto-build`** (new): unlabeled configs → QE labeling → data-driven
  cutoff/Morse-λ determination → 2b/3b/4b polynomial-order sweep at those
  fixed cutoffs → one "optimal" model (lowest holdout force RMSE) →
  optional AL stabilization handoff. Explicitly allowed to pick a winner
  (unlike `model-build`/`sweep`, which stay hands-off), per user request —
  confirmed blocking/synchronous execution, the RMSE selection criterion,
  and the AL-stabilization scope (mechanical ALC-0 staging + `al-run`
  launch, not `config.py` auto-generation) with the user beforehand.
- Every non-trivial number `auto-build` derives is traced to actual
  documented ChIMES practice (`codes/chimes_lsq-LLfork/doc/source/
  {lsq_input_file,quick_start}.rst`), not invented heuristics -- see new
  `docs/concepts/cutoffs_and_lambdas.md` for the citations:
  - **S_MINIM**: min observed pair distance minus 0.002-0.02 Å (documented
    range; default 0.02).
  - **S_MAXIM**: RDF-shell-derived (2-body = 2nd RDF minimum or the
    documented ~8 Å default; 3-/4-body = 1st RDF minimum), capped by the
    box-safety bound `chimes_lsq` itself enforces
    (`ClassDefs.C:2593-2630`'s `BOXDIM.IS_RCUT_SAFE`) -- capping is
    reported per pair, not silent.
  - **MORSE_LAMBDA**: location of the first RDF peak.
  - **Regularization**: fixed at the documented normalized-fit default
    (1e-5), not swept (only polynomial order is, per the user's request).
  - **Order sweep**: centered on the documented 12/7/3 starting point;
    selection by holdout cross-validation is the documented method for
    choosing order, not an invented criterion.
- New **`io/rdf.py`**: PBC-aware (minimum-image, orthorhombic) per-pair
  minimum distance and RDF shape computation, with peak/minimum
  detection. Validated against a real physical test case (a simple cubic
  lattice with exactly known shell distances via geometry -- 1st/2nd/3rd
  shell at a, a√2, a√3) in `tests/unit/test_rdf.py`, not just synthetic
  numbers.
- New **`stages/_cutoffs.py`**: turns the RDF computation into the actual
  S_MINIM/S_MAXIM/MORSE_LAMBDA numbers per pair, with the box-safety cap
  and per-pair `capped`/`cap_reason` reporting.
- **`fm-setup-gen`** gained `special_maxim_3b`/`special_maxim_4b`/
  `special_blocks` inputs (renders `SPECIAL 3B/4B S_MAXIM` blocks --
  `io/fm_setup.py` already round-tripped this grammar, just wasn't
  exposed as an input) -- needed so a shorter documented 3-/4-body outer
  cutoff can actually be requested; `sweep`'s per-point fm-setup-gen calls
  updated to pass these through too.
- 20 new unit tests (`test_rdf.py`, `test_cutoffs.py`, `test_auto_build.py`
  -- 71 total, up from 52), including a real end-to-end pipeline run
  against the bundled fixture (label→split→cutoffs→sweep→choose in ~33s)
  and a real `stabilize`-phase test exercising `auto_build.py`'s own
  ALC-0-staging code path against a fake al_driver fixture (mirrors
  `test_al_run.py`'s pattern) -- not a hand-copied replica of the logic.
- Full suite still passes in a simulated fresh-CI checkout (no `codes/`):
  61 passed / 10 skipped, no failures.

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
