# Changelog

## Unreleased — sweep detail, reporting, QUESTS, active-learning tooling

- **`lammps-run --machine`**: one Slurm job that re-runs the stage on the
  compute node (inputs, types and masses checked before submission; ranks
  sized to the system; `srun` inside the allocation; `thermo_last` for MD;
  `job-status` verdicts). Checked on Dane: 1 and 4 ranks give identical
  thermo output.
- **`setup --ref REPO=COMMIT`**: a vendored fork at another commit for one
  call, without editing the pin. **`setup --init-profile PATH`**: writes a
  machine profile from `sinfo`/`sacctmgr` (on Dane it reproduces the built-in
  profile) and lists what it could not determine.
- **`study --status`**: a one-screen dashboard (phases with headline
  numbers, jobs, what waits on the user, CPU-hours), also `STATUS.md`.
- **Docs and workflow audit**: a generated [All options](docs/commands/options.md)
  reference (every stage's flags, types, defaults; `tools/gen_options_doc.py`,
  kept fresh by a test); the study layout, playbooks and agent descriptions
  now describe the active-learning round directory, `al-batch`/`al-status`,
  the validation registry keys and the composite stages; stale statements
  removed (`study --status`, "committee selection not implemented", the
  planned packed-frames module, stub stages); the Cu-Zr example gained the
  cross-validated re-search; install extras `plots` and `quests` documented.
- **Element-aware fingerprints** (`fingerprint --structure-weight α`,
  `--structure-weights` sweep, `--descriptor mass|number`; `al-batch
  --structure-weight`): the paper's hybrid metric (α · structure +
  (1 − α) · composition, centrality-ordered element descriptors for
  triplets/quadruplets) implemented natively on top of the type-agnostic
  fingerprint, which stays the default and still matches the shipped tool.
  A sweep reports D²/critical per α and says whether composition or
  structure separates two sets (the paper's alloy vs molecular regimes).
- **Small-input guards**: `learning-curve` rejects fractions outside (0, 1]
  before building anything; `quests` says when a reference is too small for
  its self-dH threshold (and when that threshold is ≤ 0, so everything would
  count as novel); `al-batch` notes a rank-deficient fingerprint reference
  and a single-element composition term.
- **Worker pools spawn instead of fork** (`io/pool.py`; `al-batch`, `fingerprint`,
  `md-check`, `hyper-search`). Forking after QUESTS started numba's threads
  deadlocked inside `fork()` on Dane compute nodes: three `al-batch` jobs wrote
  nothing and timed out. Native codes are likewise started without a
  `preexec_fn` (the core-dump limit is set on the parent), which removed a
  second deadlock when the committee launched `chimes_lsq`. `al-batch` on the
  Cu-Zr harvest now finishes in 81 s. Thread and worker counts also follow the CPU affinity
  the job was given, not the node's core count.
- **Cross-validation in `hyper-search`** (`--cv-folds k`): every point scored
  on all training frames via zero row weights in one design matrix
  (group-aware, stratified folds; closest contacts protected). The external
  holdout stays reported. On Cu-Zr the SE fell from 0.06 to 0.035, and CV
  exposed variance the holdout hid (3-body model: holdout 0.31, CV 0.66,
  with four "fragile" frames mispredicted when held out). CV reports the
  median per-frame error and the fragile frames.
- **More detailed sweep**: `smoothing` stage (TERSOFF 0.5/0.75 vs CUBIC),
  `lambda_pairs` (per-pair λ), `refine` now also tries cutoff midpoints;
  one-dimensional sensitivity `profiles`; a generated
  `search/HYPER_REPORT.md`.
- **`learning-curve`** (new): holdout error vs training size from one
  design matrix; data-limited vs plateau verdict with extrapolation. Cu-Zr:
  plateau from 63 of 126 frames (energies still improving).
- **QUESTS** (`quests` stage, extra `[quests]`): entropy, diversity,
  entropy curve/saturation, per-environment dH novelty, greedy
  entropy-maximizing selection; `dataset-select --method quests` and
  `data-curate --selection quests`.
- **Active learning**: `al-batch` (close contacts + QUESTS + fingerprint +
  committee → one deduplicated batch within a budget, every score
  recorded), `al-status` (per-round holdout errors on the fixed holdout,
  stability, novelty, CONVERGED/CONTINUE verdict, `AL_STATUS.md`).
  `al-merge` refuses holdout duplicates and drops training duplicates.
- **Labeling**: `qe-converge` (ecutwfc and k-spacing ladders on one frame,
  `--collect` recommends the cheapest converged setting). `qe-relabel`
  accepts triclinic cells (ibrav 0 with the full cell), which most
  open-database frames are.
- **Reporting**: plots (`evaluate --plot` parity and per-composition bars;
  `md-check` temperature/energy/RDF; `eos-check` E(V); `learning-curve`;
  `quests` dH histograms; `al-status` progress), new registry keys
  (`md_check`, `eos_check`, `fingerprint`, `quests`, `committee`,
  `learning_curve`, `evaluate`, `al_status`), and `study-report` sections
  for data sufficiency, coverage/uncertainty, MD validation, equation of
  state, active-learning rounds, and a Figures section.
- `evaluate.evaluate_frames()` scores in-memory frames for the other stages.
- Docs: concepts/model_selection.md, guide/troubleshooting.md, pages for
  every new stage.

## Fitting audit items (docs/development/fitting_audit.md)

- **Silent failures fixed:** frames with an element a model does not know
  are refused before chimes_calculator (which exited the process with
  status 0). `amat-build` checks `fm_setup.in` against its trajectory
  (NFRAMES, elements, cutoff vs box with NLAYERS, stresses/energies, inner
  cutoff vs closest contact); chimes_lsq segfaulted on a frame-count
  mismatch. Core dumps are disabled for native codes.
- **Per-composition accuracy:** `evaluate` reports `by_composition`,
  `by_element` and `worst_frames`. `hyper-search` refuses a "tied" cheaper
  model that regresses any composition beyond its paired noise. On Cu-Zr
  this kept a cross 3-body type and improved the model (pooled 0.313 →
  0.303, pure Cu 0.145 → 0.093, energy 0.854 → 0.737 kcal/mol/atom).
- **MD-ready deployment:** `deploy` writes the repulsive penalty explicitly
  (0.02 Å, 1e5 kcal/mol/Å³; chimesFF's implicit 1e4/0.01 let 20 of 81
  frames at 1200 K sample inside the inner cutoff, the new default 2) and
  removes zeroed coefficients (identical predictions, 21 % faster on a
  4-body model). `md-check` validates with the same penalty;
  `hyper-search`'s MD cost counts nonzero coefficients.
- **`alpha` stage** in `hyper-search`; **memory-capped parallel fits**
  (one fit peaks at ~20× the dense matrix); **shared design matrices** for
  solve-only variants (alpha, stress, 4-body solvers).
- **`committee`** (new): bootstrap committee of one basis, candidates
  ranked by force/energy spread → `uncertain.xyzf`.
- **`eos-check`** (new): Birch-Murnaghan equation of state, clamped-ion
  elastic tensor, Born stability, optional DFT pressure comparison.
- `auto-build` now fits through `hyper-search`. `solve`/`model-build`/
  `sweep` default to `lassolars` α = 1e-5. `sweep` reports bootstrap SE,
  per-composition errors and `tied_with_best`.

## Robustness: failure modes found by injection

- **Masses from the model.** `lammps-run`, `md-check` and `benchmark` take
  element types and masses from `params.txt` and refuse disagreeing inputs.
  LAMMPS matches types by mass: Cu typed as 63.5 instead of 63.546 silently
  gave -183.6 instead of -272.3 kcal/mol. `--elements`/`--masses` are now
  optional cross-checks.
- **Crash-safe files.** All JSON results, caches and manifests are written
  atomically (`io/atomic.py`). A truncated manifest (killed job) is treated
  as absent; it used to crash the CLI with a raw traceback. A truncated
  hyper-search point cache is refitted.
- **`job-status`** (new, read-only): every submission records `job.json`
  (job id, expected result files). The stage returns SUCCEEDED, FAILED
  (TIMEOUT / OUT_OF_MEMORY / ... with a fix), COMPLETED_WITHOUT_RESULTS, or
  QUEUED/RUNNING. The job-monitor agent uses it first.
- **No concurrent runs in one output dir:** a live run (PID checked on the
  same host; 6 h on another host) blocks a second; `--force` overrides.
- **Lustre EREMOTE:** directories are created one level at a time with
  retries (`io/fs.py`), across all 38 former `mkdir -p` sites.
- **`doctor`** checks the Lustre file-count quota on the scratch root
  (warn at 85%, fail at 98%).
- Any failure before a stage starts still returns one JSON error object.

## Stresses, per-pair cutoffs, fingerprints, hierarchical fits, MD and AL agents

- **Stresses end to end.** `data-fetch` keeps stresses from ASE files
  (Cauchy sign) and ColabFit. It verifies the ColabFit sign per dataset
  (pressure must fall with volume per composition); MatPES stores the
  pressure sign despite the column name. `qe-relabel --collect` parses QE's
  stress. `data-curate` sets `fit_hints.fitstrs: ALL`, `evaluate` reports
  stress and pressure RMSE (GPa), and `hyper-search` fits stresses with a
  final `stress` stage that measures the stress weight. On 2-atom Cu-Zr cells
  the published weight (100) wrecked the fit; the stage chose 3, and holdout
  pressure error fell 4.43 → 2.88 GPa at unchanged force error.
- **Per-pair 3-body cutoffs:** `fm-setup-gen special_maxim_3b_pairs`
  (SPECIFIC rows) and a `3b_pairs` stage in `hyper-search`. chimes_lsq,
  chimes_calculator and LAMMPS agree to 5e-6 on such a model.
- **`fingerprint`** (new): native cluster-graph fingerprints (matches
  chimes_calculator's shipped tool on its reference), D² dataset comparison,
  per-frame novelty, `novel.xyzf`; `--machine` for large sets.
- **`hierarch`** (new): hierarchical fitting through al_driver's machinery
  (subtract element models, merge cross + element params; guarded against
  al_driver's empty 4-body block). `fm-setup-gen` writes `EXCLUDE 1B/2B` and
  `HIERARC`. Tied with fitting all at once on Cu-Zr.
- **`al-merge`** (new) and **`fm-setup-gen --hyper-choice`**: one
  active-learning round merges labels (theory check, fixed holdout, cycle
  bookkeeping for n/I decay) and refits the chosen model.
- **Agents:** `chimes-md-validator` (skill `chimes-md-validation`) and
  `chimes-active-learner`. Every study phase now has an agent.
- **`generic` machine profile** (compilers/MPI from PATH; `hosttype: none`)
  and a **CI integration workflow**: it builds the toolchain on Ubuntu, runs
  `doctor`, the unit tests and an end-to-end EMT-labeled study
  (`tests/integration`; passes on Dane).
- Fixes:
  - The holdout split keeps each pair's closest contact in training. One
    holdout frame inside the inner cutoff had pushed the Cu-Zr holdout error
    from ~0.3 to ~1.4. `evaluate` reports `n_frames_below_inner_cutoff`.
  - `fm-setup-gen` masses default to standard atomic masses (were 1.0 amu).
  - Failed stage runs can be retried without `--force`.
  - `submit` honors `work_dir`.
  - Stress rows are no longer counted as forces in training errors or CV
    grouping.

## Robustness and accuracy fixes from the assessment

Robustness:

- **Dry run = real submission.** One renderer writes the sbatch script in
  both modes, and real jobs `sbatch` that exact file (previously they went
  through al_driver's `create_and_launch_job`, whose script lacked the
  previewed `conda activate`). Jobs run the submitting interpreter (its bin
  dir first on PATH); `conda_env` is optional and guarded.
- **Node-local job directories refused** (`/tmp`, `/var/tmp`, `/dev/shm`,
  `$TMPDIR`), including in dry runs.
- **Profiles expand `${VAR}` / `${VAR:-default}`**; shipped profiles use
  `CHIMES_ACCOUNT` (and `$SCRATCH` on stampede3) instead of one user's
  account and paths. `setup --machine` accepts a profile path.
- **Stale results closed:** stage manifests fingerprint input files (size +
  mtime) and name the changed file; `hyper-search` point caches are reused
  only when data, solver, alpha and masses match. Old manifests still
  short-circuit on their path-only hash.
- **`--json-in` validated** against the stage schema (unknown keys, wrong
  types); stages that submit nothing reject `--dry-run`.
- **Approval gate hook** (`.claude/hooks/approval_gate.py`): asks before
  submissions in any form (module invocation, `machine` inside a JSON file);
  settings.json ask rules mirrored for `python3 -m agentic_chimes.cli`.

Accuracy:

- **`lammps-run` triclinic support** (restricted-triclinic box, forces
  rotated back) and **automatic replication of thin cells**: chimesFF in
  LAMMPS under-counts energy on cells thinner than 2x the cutoff (30-50
  kcal/mol on 2-atom MatPES cells); replicated results match `evaluate`
  to 1e-4.
- **`md-check`** (new stage): short NVT MD of candidate models, locally or
  as one Slurm job; stability (runaway, energy jumps, lost atoms),
  equilibration, close contacts vs inner cutoffs, partial RDFs vs a
  reference, and `harvest.xyzf` (<=20 close-contact + <=20 other frames per
  run) for active learning.
- **`weights`** (new stage): al_driver's weight methods A-G on
  `b-labeled.txt`, published presets (`al_driver`, `lindsey2020`,
  `carbon2_large`, `hierarchical2026`) and n/I active-learning decay;
  `model-build`/`hyper-search --weights-preset`.
- **`hyper-search --prefer richer`**: take the richest statistically tied
  model (and skip exclusions) for models headed into active learning.
- **Group-aware holdout split** (`dataset-select`/`data-curate
  --split-by group`, default): correlated frames (relaxation paths, close MD
  frames) are held out together.

## Literature grounding and assessment

- **ChIMES literature integrated.** The `chimes-literature` skill distills
  8 ChIMES papers into decision guidance with citations: cutoffs, λ,
  smoothing, orders, weights, active learning, validation and published
  accuracy. Every agent and playbook points to it, and `CLAUDE.md` asks
  agents to cite it. `tools/index_papers.sh` builds a searchable text index
  of a local, gitignored `chimes_papers/` folder. New page:
  `docs/concepts/literature.md`.
- **`evaluate` reports `reduced_force_rmse`** (RMSE ÷ mean |F_ref|), the
  literature's "reduced RMSE". Compare with published models using this
  number, not `relative_force_error` (÷ RMS force, which is 1.48× lower on
  Cu-Zr).
- **`hyper-search --smoothing`**: CUBIC (default) or `'TERSOFF <f_O>'` for
  every fit. With CUBIC, the report notes that many-body terms may be
  suppressed.
- **Assessment**: `docs/development/assessment.md`, which covers bugs, gaps,
  quality-of-life improvements and a roadmap for general users.
- `chimes_papers/` is gitignored and was removed from the index; the local
  files are kept.

## Benchmark and report agents; user-facing documentation

- **Studies**: `study` stage (`--init` standard layout + `study.json`
  registry, `--register key=path`, status). A study directory switches on
  automatic login-node CPU-time bookkeeping (`usage/local.jsonl`, written by
  the CLI).
- **`usage`**: CPU-hours by phase and job, from `sacct` filtered by working
  directory (catches al_driver's own jobs) plus the ledger. Reports charged
  (allocated × elapsed) and used (TotalCPU from job steps) hours and
  allocation efficiency. On the Cu-Zr study: 103 charged, 7.3 used.
- **`hyper-search --machine` right-sizes its allocation** (cores = largest
  stage's fit count) after the usage report showed ~2 % efficiency on full
  112-core nodes.
- **`benchmark`**: strong/weak LAMMPS scaling of the final model as one Slurm
  job, then `--collect`: efficiency, ns/day, core-seconds per atom-step, and
  CPU-hour estimates that use the measured cost at the packing each run would
  use (per-core cost rose 1.8× from 1 rank to a full node on Cu-Zr), with a
  note when multi-node estimates are extrapolated.
- **`deploy`**: `06_deploy/` with params, `in.lammps.example`, `MODEL_CARD.md`
  (accuracy ± SE, validity by minimum sampled distance, cost to run,
  development cost, data licenses, caveats).
- **`study-report`**: `REPORT_FACTS.json` + `REPORT.md` with every table
  filled and narrative placeholders. MD runs are summarized from LAMMPS logs,
  ensemble-aware (energy change is called drift only for NVE).
- **Agents**: `chimes-benchmark`, `chimes-report-writer` (+ skills
  `chimes-benchmarking`, `chimes-study-report`); `chimes-study` extended with
  the MD, benchmark, deploy and report phases.
- **Documentation rewritten for users**: new README; landing page; getting
  started; a User guide (running a study, study layout, the agents, compute
  and costs, the Cu-Zr worked example with real numbers); commands grouped by
  phase.
- Validated end to end on Cu-Zr: benchmark job (16 cases, 8.7 min), 10 ps MD
  at 300 and 1,200 K, deploy, and the full report.

## Unreleased — 4-body sweeps and cluster-type exclusions

- **`hyper-search` `exclude` stage**: leave-one-type-out over 3-/4-body
  cluster types (`EXCLUDE` blocks) with greedy rounds, each judged against
  both the current and the stage-entry model so losses cannot accumulate.
  Reports per-type coverage from chimes_lsq's log, coefficients saved and
  score change. Adds a note when exclusions raise training error by >10 %.
- **4-body stage**: order × cutoff × `four_body_solvers`, each extra solver
  also refitting the 3-body baseline (like-for-like). Light by default at the
  user's request (4-body builds scale steeply): orders 2-3, two cutoffs, one
  solver, `--max-fit-seconds 600`.
- **New solver `blocklasso`**: lassolars with each body-order block rescaled
  to the 2-body scale. Measured on Cu-Zr it was worse everywhere (3-body
  baseline 0.63 vs 0.32), so it is opt-in only.
- **Tie-breaking by estimated MD cost** (clusters per atom ×
  coefficients per body order, using the data's density), replacing
  "shorter cutoff first", which chose a model ~2× more expensive.
- **3-body grid extended to 7.0 Å on Cu-Zr**: order 4 @ 7.0 Å beat 6.33 Å
  (holdout 0.28 vs 0.31).
- `sweep` grid keys `special_maxim_3b/4b`, `exclude_3b/4b`; a
  `relative_force_error` column.
- Fixed: failed fits were cached and never retried (only done/timeout are
  reused now); the inner `--machine` job always re-runs (`--force`) while
  still reusing cached fits; the exclusion stage could report "every type
  needed" without testing any.

## Unreleased — hyperparameter agent

- **`chimes-hyperparameter-tuner` subagent + `chimes-hyperparameter-search`
  skill**: analyze → plan → search as an approved Slurm job → judged
  result, `hyper_choice.json` and a written report. `chimes-study` phase 3
  now delegates to it.
- **`hyper-search` finished and validated**: 52 fits on real Cu-Zr in 12 min
  on Dane `pdebug` via its own `--machine` path. Changes from the first
  draft:
  - ties are decided by the *paired*-bootstrap SE of each point's difference
    to the best;
  - cost ordering is cutoffs first, then coefficients;
  - many-body cutoff candidates reach toward the second shell;
  - the small-signal check flags instead of excluding;
  - `--max-fit-seconds` with `timeout` status.
- **Solver study** (4 bases × 5 solvers, `pdebug`): raw `lassolars` α=1e-5
  stays the default; column-normalized solvers overfit freed many-body terms
  on scarce data (holdout 1.1-1.5 vs 0.43).
- **New `solve` algorithms** `nsvd`, `nridge`, `nlasso`, `nridgecv`:
  column-normalized, written out through `chimes_lsq.py --read_output`;
  `nridgecv` picks α by CV over whole training frames, scoring force rows.
- **Fixed: 1-atom crystals treated as isolated atoms.** Pair analysis in
  `data-curate` and `hyper-analyze` kept only `i < j` neighbour pairs,
  dropping an atom's own periodic images, the only neighbours in a 1-atom
  cell. `data-curate` removed such MatPES frames as `isolated_atom`; Cu-Zr
  now keeps 158 frames instead of 152. Perfect-crystal first RDF peaks
  (exactly at the minimum distance) are now detected too.
- **Fixed: MPI binaries crashed inside Slurm steps.** `chimes_lsq` (and
  single-process LAMMPS) inherited the step's `PMI_*` variables without their
  file descriptor and segfaulted; they now start as singletons
  (`hpc/local.singleton_env`).
- **Fixed: a dry run blocked the real run.** The idempotency manifest
  ignored `--dry-run`, so a real submission after a preview short-circuited
  to the preview's result. Dry runs now record nothing.

## Unreleased — hyperparameter phase (in progress) and evaluation fixes

- **Fixed: `evaluate` compared forces in mixed units.** Reference forces in
  `.xyzf` are hartree/bohr (ChIMES `doc/source/units.rst`); predictions are
  kcal/mol/Å. The difference was taken unconverted, so "force RMSE" was
  essentially the RMS of the predicted forces. Every force RMSE from
  `evaluate`, `sweep` or `auto-build` before this fix, and any model choice
  based on it, is invalid. New outputs: `relative_force_error`,
  `rmse_energy_kcal_mol_per_atom`, `reference_force_rms_kcal_mol_ang`,
  optional `--per-frame` errors.
- **Fixed: small cells evaluated wrongly.** chimes_calculator's serial
  interface mishandles cells thinner than twice the cutoff (energies off by
  ~20 kcal/mol with `small=False`; forces summed over replicas with
  `small=True`). `evaluate` and `al-select` now evaluate an exact
  replicated supercell; this reproduced chimes_lsq's own `force.txt` to
  5×10⁻⁶ kcal/mol/Å on 122 real triclinic frames.
- **Fixed: chimes_lsq segfault on mixed cell headers.** A plain `Lx Ly Lz`
  frame after a `NON_ORTHO` frame crashes chimes_lsq. `write_xyzf` now
  writes every frame as `NON_ORTHO` when any frame is triclinic, and the
  reader turns diagonal `NON_ORTHO` cells back into orthorhombic frames.
- **New `hyper-analyze`**: per-pair minimum distances and RDF shells from
  neighbour lists (any cell shape) → inner cutoffs, Morse lambdas,
  outer-cutoff candidates, N_LAYERS per candidate, equation counts.
- **New `hyper-search` (in development)**: staged search (2b → 3b → 4b →
  lambda → refine) that chooses the cheapest model within one bootstrap
  standard error of the best. It has a coefficients-per-equation budget and a
  per-body-order column-scale diagnostic, caches and resumes fits, runs in
  parallel, and can submit itself to Slurm. Open issues are listed in
  `docs/commands/hyper-search.md`: local `lassolars` does not normalize
  columns, so 4-body terms get zeroed; the `min_signal` default is too
  aggressive; many-body cutoff candidates are too short (measured, see
  `docs/concepts/cutoffs_and_lambdas.md`); a segfault inside Slurm steps.

## Unreleased — data selection and curation agent

- **Data agent**: `chimes-data-curator` subagent + `chimes-data-curation`
  playbook skill (plan → search → fetch/generate → curate →
  `data_manifest.json`; returns decisions and QE dry-runs, never submits).
  New `chimes-study` skill lays out the five-phase study (plan, data,
  hyperparameters, build, active learning), the study directory and handoff
  files, with the data phase wired to the new agent.
- **New stages**:
  - `data-search` searches the ColabFit catalog on Hugging Face (~500 datasets,
    one schema, cached) by elements/method. It never downloads configurations,
    ranks by relevance, and attaches notes on what each dataset family really
    contains.
  - `data-fetch` pulls from `colabfit:<repo>` or local DFT files into one
    `.xyzf` (kcal/mol, hartree/bohr) plus provenance. It uses a two-pass HTTP
    range read and seeded uniform sampling, enforces one level of theory, and
    has a structure-only `relabel` mode.
  - `data-generate` builds orthorhombic ≥8 Å supercells with substitution,
    strain and rattle for QE labeling.
  - `data-curate` merges only provenance-compatible pools. It filters
    duplicates, isolated atoms, vacuum/cluster frames, overlaps, force outliers
    and energy outliers (per-element reference fit, MAD plus an absolute floor),
    reports per-pair distance coverage and `N_LAYERS` needs, does an FPS
    subsample and stratified split, and writes `data_manifest.json`.
- **`qe-relabel`**: `--kspacing` (per-frame k-grid; a fixed grid across
  different cell sizes gives inconsistent energies); collect writes
  `provenance.json` keyed on a hash of the DFT settings, so batches with
  identical settings (e.g. AL rounds) merge and others are refused.
- **Fixed**: `.xyzf` triclinic frames were written with `NON-ORTHO`;
  chimes_lsq only recognises `NON_ORTHO` (ClassDefs.C) and would have read
  the frame as orthorhombic with a garbage box. Reader accepts both; tested
  against chimes_lsq's own `nonorth2` fixture.
- Validated on real data: Cu-Zr from MatPES-PBE-2025.2 (169 frames → 152
  after removing 8 isolated atoms, 8 vacuum clusters, 1 duplicate), with
  labels checked exactly against recomputation in tests (EMT stand-in).
  Found and handled: HF anonymous rate limiting on shared lab IPs (token
  support, resolve-URL downloads, incremental catalog refresh) and a
  ColabFit dataset (`UNEP_v1_2023_train`) whose parquet is corrupt at the
  source despite a matching checksum (skipped, remembered, flagged in search).
- New `data` extra (ase, pyarrow, huggingface_hub, fsspec, aiohttp); CI now
  installs it so the data tests run there.

## Unreleased — Claude Code agent setup + clean-stdout CLI contract

- **CLI stdout is now exactly one JSON object.** Everything a stage or its
  native libraries write to fd 1 (chimes_calculator's C++ banner in
  `evaluate`/`al-select`/`lammps-run`, al_driver's progress prints) is
  diverted at the file-descriptor level to `<output-dir>/<stage>.log`, returned
  as `stage_log` when non-empty. Found because a real `al-select` call put
  ~1,600 lines ahead of its JSON, which an agent cannot parse. Errors are now
  `{"error", "log", "log_tail"}`. Regression-tested in a real subprocess
  (`tests/unit/test_cli_stdout_contract.py`).
- **Claude Code integration**: `CLAUDE.md` (rules + how to drive the CLI),
  four skills in `.claude/skills/` (`chimes-auto-build`, `chimes-build-model`,
  `chimes-hpc-jobs`, `chimes-active-learning`), two subagents in
  `.claude/agents/` (`chimes-job-monitor`, a read-only Haiku status checker;
  `chimes-fit-reviewer`, an independent result reviewer), and shared
  permissions in `.claude/settings.json` (allow read-only/local work; ask
  before anything that spends allocation; deny `scancel -u` and edits to
  `codes/`/`deps/`). `tests/unit/test_claude_setup.py` guards all of it
  against rot. New page `docs/concepts/claude_code_integration.md`.
- README: added a "Using it with Claude Code" section; fixed the stage-
  contract text that still said `--hpc` (the flag is `--machine`) and that no
  stage calls another (the composites do).

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
    `evaluate`). *[Correction, later: that RMSE was computed with
    `evaluate`'s unit bug and is not a valid accuracy figure; the
    `params.txt` validity check stands.]*
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
