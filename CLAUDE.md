# Agentic ChIMES

`chimes-agent` is a tool-calling CLI for building ChIMES machine-learned
interatomic potentials end to end. You (the agent) drive it: run a stage,
read its JSON, decide the next stage. The CLI does not chain stages or make
judgment calls for you (except `auto-build`, which is explicit about it).

## How to drive it

- One subcommand per stage. **`chimes-agent <stage> --describe` is the source
  of truth for a stage's inputs/outputs** — read it instead of guessing flags.
  Stages: `setup`, `doctor`, `study`, `usage`, `data-search`, `data-fetch`, `data-generate`,
  `data-curate`, `fingerprint`, `quests`, `dataset-select`, `qe-relabel`, `qe-converge`, `hyper-analyze`,
  `hyper-search`, `learning-curve`, `fm-setup-gen`,
  `amat-build`, `solve`, `weights`, `hierarch`, `model-build`, `sweep`, `auto-build`, `evaluate`,
  `lammps-run`, `md-check`, `eos-check`, `benchmark`, `deploy`, `study-report`, `submit`, `job-status`, `al-select`, `al-batch`, `al-merge`, `al-status`, `committee`, `al-run`.
- **stdout is exactly one JSON object** (errors are `{"error", "log",
  "log_tail"}` with exit code 1). All native-library and subprocess noise is
  diverted to `<output-dir>/<stage>.log`, whose path comes back as
  `stage_log`. Read that log only when something looks wrong.
- Give every file-writing stage an `--output-dir`. Re-invoking with identical
  inputs short-circuits via a manifest; different inputs at the same path
  fail loudly until you pass `--force`. Use one directory per experiment.
- Prefer `--json-in file.json` over long flag lists for anything with nested
  values (`--order`, `--pair-cutoffs`, `--order-grid`, `--masses`).
- Long-running stages block until done: `auto-build`, `sweep` (with
  `--machine`), `solve --algorithm dlars|dlasso`, `amat-build --machine`. Run
  those with the Bash tool's `run_in_background`, then check the output
  file — do not sleep-poll. `al-run` is the exception: it detaches itself
  and returns a PID.
- Stages with real cost accept `--dry-run`. It renders the exact sbatch
  script a real submission would send, without submitting. Do this first.
  Stages that submit nothing reject `--dry-run`. Caveat:
  `auto-build --dry-run` only previews the QE submission, not the rest of
  the pipeline.
- Prefer `hyper-search --cv-folds 4` on datasets under ~500 frames: every
  training frame is scored, the standard error drops ~2x, and model
  variance shows up (holdout and CV disagreeing is a finding, not noise).
  `search/HYPER_REPORT.md` is the search's own account; start reports from it.
- Run `learning-curve` on the chosen basis before active learning: a
  plateau means more of the same data will not help.
- One active-learning round is: `md-check` → `al-batch` (close contacts +
  QUESTS + fingerprint + committee, within a budget) → `qe-relabel` →
  `al-merge` → refit (`fm-setup-gen --hyper-choice`) → `md-check` →
  `al-status` for the verdict. QUESTS (`quests` stage) needs
  `pip install -e ".[quests]"`. For alloys give `al-batch`/`fingerprint`
  `--structure-weight 0.25` (element-aware fingerprint); `fingerprint
  --structure-weights 0,0.5,1` shows whether composition or structure
  separates MD from training.
- After a submission, `chimes-agent job-status --work-dir <dir>` says
  whether it succeeded, failed (why, fix) or completed without results.
  `chimes-agent study --study <dir> --status` is the whole study on one
  screen: phases, jobs, what waits on the user, CPU-hours.
- Long MD: `lammps-run --machine` (one job, plain run) or `md-check
  --machine` (several models/temperatures with stability analysis). A new
  cluster: `setup --init-profile <file>` writes its machine profile.
- LAMMPS stages (`lammps-run`, `md-check`, `eos-check`, `benchmark`) take element types
  and masses from `params.txt`; pass `--masses` only to double-check.
  LAMMPS matches types by mass, so a mismatch is refused. A rounded mass
  silently gave a wrong energy before.
- `--json-in` refuses unknown keys and wrong types. If it errors, fix the
  key name; don't drop the setting.

## Playbooks (skills)

Load the matching skill before starting that kind of task:

- `chimes-study` — an end-to-end goal ("I need a potential for X"): the
  orchestration plan, study layout, phase handoffs and approval gates
- `chimes-data-curation` — find, fetch, generate and curate training data
- `chimes-hyperparameter-search` — cutoffs, Morse lambdas, 2/3/4-body orders
- `chimes-benchmarking` — CPU-hours used; strong/weak scaling; sizing compute requests
- `chimes-study-report` — the final human-readable study report
- `chimes-auto-build` — unlabeled or labeled configs → one optimal model
- `chimes-build-model` — stage-by-stage fitting, sweeps, reading results
- `chimes-hpc-jobs` — anything that touches Slurm, QE, DLARS, or lustre
- `chimes-active-learning` — `al-select` / `al-run` / stabilizing a model
- `chimes-md-validation` — MD checks of candidate models, fingerprint coverage, choosing between tied models
- `chimes-literature` — published ChIMES methodology (cutoffs, λ, smoothing,
  orders, weights, AL, validation, accuracy benchmarks) with citations

Subagents: `chimes-data-curator` (the data phase: plan → search → fetch/
generate → curate → `data_manifest.json`; returns decisions it cannot make),
`chimes-hyperparameter-tuner` (the fitting phase: analysis → search plan →
judged result → `hyper_choice.json`; returns Slurm jobs for approval),
`chimes-benchmark` (CPU-hour accounting + scaling benchmark → sizing
recipe), `chimes-report-writer` (collates the study into `REPORT.md`),
`chimes-md-validator` (MD stability, close contacts, RDF, fingerprint
coverage → which model, or "needs active learning"),
`chimes-active-learner` (active-learning rounds: harvest → label → merge →
refit → re-validate, with a stopping rule), `chimes-job-monitor` (cheap Slurm/log status checks — delegate
waiting-and-checking to it) and `chimes-fit-reviewer` (independent read of a
finished sweep/evaluate result before you recommend a model).

## Ground technical choices in the literature

Before recommending a cutoff, λ, smoothing function, order grid, solver,
weight or AL setting, check `chimes-literature` and cite the paper (e.g.
"Lindsey 2019, 10.1021/acs.jctc.8b00831") in your reasoning and write-ups.
If you deviate from published practice, say so and why. Full text, when
present, is under `chimes_papers/text/` (gitignored; build with
`tools/index_papers.sh`). Compare accuracy with papers only via
`evaluate`'s `reduced_force_rmse`, never `relative_force_error`.

## Rules that prevent expensive mistakes

- **Never edit `codes/` or `deps/`.** They are gitignored clones of three LLNL
  forks and built third-party code, recreated by `chimes-agent setup`.
- **HPC job I/O must live on a shared filesystem (`/p/lustre2/$USER/...`),
  never `/tmp`** — compute nodes cannot see a login node's local disk, and
  the job will report COMPLETED with no output.
- **lustre2 limits file count, not space (~1.05M files).** Never write one
  file per frame at scale; pack into a single `.xyzf`. Delete runaway logs
  (`dlars.log`, `traj.txt`) promptly. Archive to `/p/lustre3/$USER`.
- **On Dane always request a full node** (`--ntasks-per-node 112`); a bare
  `-N 1` is 1 CPU + 2.3 GB. The profile enforces this — don't fight it.
- **DLARS: leave `--normalize` at its default (false).** `true` fails
  immediately with MKL errors (confirmed on a real job).
- **One level of theory per fit.** Never merge labels from different
  datasets, codes or DFT settings; `data-curate` refuses by default. Active
  learning must reuse the base set's QE settings (`provenance.json`).
- Hugging Face rate-limits anonymous downloads per IP (shared on lab
  networks). If `data-*` stages report HTTP 429, ask the user to set
  `HF_TOKEN` rather than retrying in a loop.
- Units are fixed: energy kcal/mol, force hartree/bohr in training files.
  QE output (Ry, Ry/bohr) is converted by `qe-relabel --collect`.
- Cutoff derivation (`auto-build`) supports **orthorhombic boxes only**.
  LAMMPS stages (`lammps-run`, `md-check`, `eos-check`, `benchmark`) handle triclinic
  cells and replicate cells thinner than 2× the cutoff (LAMMPS energies are
  wrong on them otherwise).
- A PreToolUse hook (`.claude/hooks/approval_gate.py`) asks the user before
  any submission, including `--machine` hidden in a `--json-in` file and the
  `python3 -m agentic_chimes.cli` form. Don't try to route around it.
- Set `CHIMES_ACCOUNT` to the user's Slurm bank / allocation; profiles read
  it (`stampede3` requires it).
- Don't commit, push, or open PRs unless asked.

## Working on this repo itself

- Tests: `pytest tests/unit` (no HPC needed; tests needing built components
  skip cleanly). Docs: `mkdocs build --strict` must stay warning-free.
- Every user-visible change updates `CHANGELOG.md` and the matching
  `docs/commands/<stage>.md`. Stage source is `src/agentic_chimes/stages/`;
  a stage exposes `NAME`, `SUMMARY`, `SCHEMA`, `add_arguments`, `run`.
- Machine specifics live in `src/agentic_chimes/machines/profiles/*.yaml`,
  never hard-coded in stages.
- Full docs: `docs/` (searchable site via `mkdocs serve`). Architecture:
  `docs/concepts/stages_and_contracts.md`.
