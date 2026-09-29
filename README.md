# Agentic ChIMES

Build [ChIMES](https://chimes-lsq.readthedocs.io/) machine-learned interatomic
potentials by describing what you need. Open this repository in
[Claude Code](https://claude.com/claude-code), say *"I need a ChIMES potential
for liquid Cu-Zr"*, and a team of specialist agents finds and curates
training data, chooses cutoffs and polynomial orders, fits and validates the
model, benchmarks it, and writes up the whole study. They stop for your
approval before anything is submitted to the cluster.

Every step is also a plain command-line stage (`chimes-agent <stage>`) with a
JSON contract, so everything the agents do can be run, inspected and
repeated by hand.

**Full documentation:** https://laubachb.github.io/Agentic-ChIMES/ (or `mkdocs serve`).

---

## What a study produces

| | |
|---|---|
| **A fitted model** | `params.txt` + `fm_setup.in`, ready for LAMMPS (`pair_style chimesFF`) |
| **A model card** | what it is, holdout accuracy (with uncertainty), the distances and compositions it is valid for, data provenance and licenses |
| **A cost model** | strong/weak scaling in LAMMPS and CPU-hour estimates for production runs ("100,000 atoms for 1 ns = N CPU-hours on M nodes") |
| **A development ledger** | CPU-hours charged vs. used, by phase and job |
| **A written report** | `REPORT.md`: data → hyperparameters → model → MD → performance → cost, with findings, caveats and next steps; every number traced to an artifact |

## How it works

```
 you ─► plan ─► data ─► hyperparameters ─► model ─► active learning ─► MD ─► benchmark ─► deploy + report
         │        │            │              │            │             │         │              │
       STUDY.md  data-      hyperparameter  fit-        al-select/     lammps-  benchmark     report-
                 curator     tuner          reviewer     al-run         run      agent         writer
```

- **The orchestrator** is the main Claude Code conversation (skill
  `chimes-study`). It plans the study with you, keeps `STUDY.md`, delegates
  each phase to a specialist agent, and brings every decision and cluster
  submission back to you.
- **Specialist agents** (`.claude/agents/`): data curation, hyperparameter
  search, independent fit review, benchmarking and compute accounting, the
  final report, and cheap job monitoring. Each follows a written playbook
  (`.claude/skills/`) and returns a fixed-format report.
- **Stages** do the deterministic work: fetch from open DFT databases,
  label with Quantum ESPRESSO, build and solve the ChIMES fit, evaluate,
  run LAMMPS, submit to Slurm. `chimes-agent <stage> --describe` prints any
  stage's full contract.
- **Guardrails**: nothing is submitted without a dry run and your approval;
  one level of theory per fit is enforced; login-node compute stays small;
  `codes/` (the upstream ChIMES repositories) is never edited.

## Quick start

### 1. Install

```bash
git clone https://github.com/laubachb/Agentic-ChIMES.git && cd Agentic-ChIMES
pip install -e ".[data,al-select]"          # the chimes-agent CLI + open-data and AL extras
chimes-agent setup --machine dane --component all   # clones the ChIMES forks into codes/ and builds
chimes-agent setup --status                          # chimes_lsq, chimes_calculator, LAMMPS, Quantum ESPRESSO
```

Built-in machine profiles: `dane` (LLNL LC), `stampede3` (TACC). For another
cluster, write a profile YAML and pass its path to `--machine`
([machine profiles](docs/concepts/machine_profiles.md)). Optional: set
`HF_TOKEN` (a free Hugging Face token) so open-data downloads are not
rate-limited on shared lab networks.

### 2. Start a study

Open Claude Code in the repository and describe the goal. The more you say
about what the model is for, the better the plan:

> Build a ChIMES potential for Cu-Zr metallic glasses: liquid and amorphous,
> 300-2000 K. Label with Quantum ESPRESSO on Dane, bank pls2. Study
> directory /p/lustre2/me/studies/cuzr.

You will be asked about things only you can decide: which DFT settings
labels must match (so later active learning stays consistent), how much
compute to spend, and approval for each cluster job. Everything else is
decided, justified in writing, and recorded in the study directory.

### 3. Read the results

```
/p/lustre2/me/studies/cuzr/
  REPORT.md                 the write-up (start here)
  06_deploy/MODEL_CARD.md   the model, its validity range and cost
  06_deploy/params.txt      the model
  05_bench/benchmark.json   scaling and the CPU-hour cost model
  usage/usage_report.json   compute used
  01_data/ 02_fit/ 03_al/ 04_md/   every intermediate, resumable
```

A study can be stopped and resumed at any phase; ask Claude to "resume the
study in <dir>".

## A real example: Cu-Zr from MatPES

Built while developing this toolkit (the full report is in the
[worked example](docs/guide/example_cuzr.md)):

- **Data:** the data agent searched ~500 open datasets. The best source was
  MatPES-PBE: 169 Cu/Zr-only frames, curated to 158 (9 vacuum/isolated-atom
  frames, 1 duplicate and 1 energy outlier removed).
- **Hyperparameters:** 61 fits in all. The search chose 2-body order 6 at
  8 Å and 3-body order 4 at 7 Å, with no 4-body terms (they never helped).
  It excluded the Cu-Cu-Zr and Cu-Zr-Zr 3-body types, whose removal cost
  nothing measurable, while Zr-Zr-Zr was essential. The result is 52
  coefficients, with holdout relative force error 0.31 ± 0.06.
- **Finding:** at that error level the data (tiny cells, 126 training
  frames) is the limit, not the settings. The report says so and recommends
  active learning.

## Using the stages directly

The agents are optional. Every stage runs by hand with the same contract:

```bash
chimes-agent data-search --elements Cu,Zr                        # which open datasets exist
chimes-agent data-fetch --source colabfit:colabfit/MatPES-PBE-2025.2 --elements Cu,Zr --output-dir d/fetch
chimes-agent data-curate --frames d/fetch/pool.xyzf --elements Cu,Zr --output-dir d/curate
chimes-agent hyper-search --data-manifest d/curate/data_manifest.json --machine dane --dry-run --output-dir d/fit
chimes-agent solve --describe                                    # any stage's full contract
```

Stdout is always exactly one JSON object; logs go to `<output-dir>/<stage>.log`.
A 5-minute, no-HPC example that fits a tiny model to a bundled fixture is in
[Getting started](docs/getting_started.md#try-the-stages-without-hpc).

## Commands by phase

| Phase | Commands |
|---|---|
| Study | `study` (create / register / status), `usage` (CPU-hours), `study-report` |
| Data | `data-search`, `data-fetch`, `data-generate`, `data-curate`, `dataset-select`, `qe-relabel` |
| Hyperparameters & fit | `hyper-analyze`, `hyper-search`, `fm-setup-gen`, `amat-build`, `solve`, `model-build`, `sweep`, `auto-build`, `evaluate` |
| Active learning | `al-select`, `al-run` |
| MD, performance, deployment | `lammps-run`, `benchmark`, `deploy` |
| Infrastructure | `setup`, `submit` |

Reference for each: [docs/commands/](docs/commands/index.md).

## Status and limitations

- Tested on LLNL Dane (Slurm, MVAPICH2, Intel). Other Slurm clusters need a
  machine profile; PBS/LSF are not supported.
- Quantum ESPRESSO is the built-in labeler; al_driver supports others for
  its own loop.
- Open data comes from ColabFit on Hugging Face; local DFT output (VASP,
  QE, extxyz, anything ASE reads) is supported directly.
- `lammps-run` and `benchmark` use orthorhombic cells; triclinic training
  data is fine for fitting.
- Not yet agents: MD candidate generation and active-learning orchestration
  (the `chimes-active-learning` skill and `al-run` cover it by hand).

Known gaps, bugs and the roadmap:
[docs/development/assessment.md](docs/development/assessment.md). The
methodology the agents follow, with citations:
[docs/concepts/literature.md](docs/concepts/literature.md).

## Development

```bash
pip install -e ".[dev,data,al-select,docs]"
pytest tests/unit            # no HPC needed; tests needing built components skip cleanly
mkdocs build --strict        # the documentation site
```

Architecture and conventions: [docs/concepts/stages_and_contracts.md](docs/concepts/stages_and_contracts.md)
and [CLAUDE.md](CLAUDE.md). A new stage exposes `NAME`, `SUMMARY`, `SCHEMA`,
`add_arguments`, `run`, is registered in `cli.STAGE_MODULE_NAMES`, and gets a
`docs/commands/` page; `tests/unit/test_claude_setup.py` checks that the
agent configuration mentions it. See [CHANGELOG.md](CHANGELOG.md) for history.

Training data fetched from open databases keeps its original license. The
model card and report list each source; cite them when you publish.
