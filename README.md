# Agentic ChIMES

A tool-calling CLI and Python API for building [ChIMES](https://chimes-lsq.readthedocs.io/)
machine-learned interatomic potentials end to end: dataset selection, DFT/QM
relabeling, `fm_setup.in` authoring, design-matrix generation, DLARS/LASSO
solving, LAMMPS/holdout evaluation, and Slurm submission on LLNL and TACC
HPC systems.

**What this is:** every stage of the ChIMES model-building workflow exposed
as a discrete command with a structured JSON contract — `chimes-agent
<stage> --json-in in.json --json-out out.json`. A human or a coding agent
(e.g. a Claude Code session) calls one stage at a time, inspects the result,
and decides what to do next.

**What this is not:** an autonomous closed-loop orchestrator that runs a
multi-cycle active-learning campaign by itself. Hyperparameter choices
(regularization strength, basis order, dataset curation) stay a human/agent
judgment call made *between* stage invocations — this repo makes each of
those steps fast, reliable, and inspectable, not automatic.

See [docs/](docs/) for the full documentation set; this README is the map.

---

## Repo layout

```
Agentic-ChIMES/
├── codes/                    # gitignored -- cloned fresh by `chimes-agent setup` (never pushed to this repo)
│   ├── al_driver-LLfork/       # active-learning orchestration (reference for QM-driver contract)
│   ├── chimes_lsq-LLfork/      # design-matrix generation (chimes_lsq) + DLARS/LASSO solver
│   └── chimes_calculator-LLfork/  # force-field evaluator (ctypes + LAMMPS pair style)
├── deps/                     # gitignored -- built/fetched by `chimes-agent setup`
│   ├── lammps-chimes/          # -> codes/chimes_calculator-LLfork/etc/lmp/exe/
│   └── quantum-espresso/       # cloned + built pw.x
├── src/agentic_chimes/        # this repo's own code: the CLI + orchestration layer
│   ├── cli.py                   # `chimes-agent` entry point
│   ├── config.py                 # path/binary resolution (codes/, deps/, deps/installed.json)
│   ├── setup/                    # install/bootstrap subsystem (`chimes-agent setup`)
│   ├── machines/                 # HPC machine profiles (Dane, Stampede3, + your own YAML)
│   ├── qm_drivers/               # QM-driver plugin registry (VASP/CP2K/DFTB+/Gaussian/QE)
│   ├── converters/                # unit conventions, QE->xyzf conversion
│   ├── stages/                    # one module per CLI subcommand
│   ├── io/                        # fm_setup.in and .xyzf parsers/writers
│   └── hpc/                       # Slurm submission (wraps al_driver's own helpers.py)
├── tests/unit/                # no HPC/allocation needed; run these on any machine
└── docs/                      # concepts, per-command reference, worked tutorial
```

The vendored forks under `codes/` are cloned fresh by `chimes-agent setup`
(see `setup/clone_codes.py`) from their real `LindseyLab-umich` GitHub
remotes, pinned to a specific validated commit — `codes/` is gitignored and
**never pushed to this repo** (these are the lab's own forks; this repo
should not embed or redistribute their contents/history). They are never
edited in place either — everything new lives in `src/agentic_chimes/`,
which subprocesses the vendored binaries/scripts or calls their
Python/ctypes APIs directly. A fresh clone of Agentic-ChIMES has no
`codes/` directory at all until you run `chimes-agent setup`. To move to a
newer commit of a fork, bump its `ref` in `setup/clone_codes.py:REPOS` and
run `chimes-agent setup --component codes --force`.

---

## Install

```bash
git clone <this-repo>
cd Agentic-ChIMES
pip install -e .          # installs the `chimes-agent` CLI (numpy, pyyaml, jsonschema only)
```

Then build the underlying ChIMES toolchain, LAMMPS, and Quantum ESPRESSO for
your machine:

```bash
chimes-agent setup --machine dane --component all
# or just one piece:
chimes-agent setup --machine dane --component chimes_calculator
# or just clone the vendored forks into codes/ without building anything (no --machine needed):
chimes-agent setup --component codes
```

`setup` always starts by cloning the three vendored forks
(`al_driver-LLfork`, `chimes_lsq-LLfork`, `chimes_calculator-LLfork`) into
`codes/` from their real GitHub remotes at a pinned commit — `codes/` is
gitignored, so a fresh checkout of this repo has none of that source until
you run this (see [docs/concepts/vendored_forks.md](docs/concepts/vendored_forks.md)).
It then runs each fork's own (unmodified) `install.sh` under the right
modules for your machine, plus a from-scratch clone+build of Quantum
ESPRESSO (not vendored anywhere — there was no QE support in any of these
repos before this one). Built artifact paths are recorded in
`deps/installed.json`; every stage resolves the binaries/libraries it needs
from there (or from an explicit `AGENTIC_CHIMES_*_BIN`/`AGENTIC_CHIMES_*_LIB`
env var override). Check what's installed with:

```bash
chimes-agent setup --status
```

Built-in machine profiles: `dane` (LLNL LC), `stampede3` (TACC). Add your
own by pointing `--hpc /path/to/your_machine.yaml` at a file following the
same schema — see [docs/concepts/machine_profiles.md](docs/concepts/machine_profiles.md).

---

## Quickstart (no HPC allocation needed)

Every stage prints one JSON object; chain them by hand or from a script.
This example fits a tiny model against a bundled ChIMES test fixture (kept
to a small basis — order `6 2` — on purpose: `solve --algorithm svd` does a
dense SVD, and a much larger basis, e.g. `12 5 4`, is slow enough on a
login node that you want the DLARS/HPC path instead; see
[docs/commands/solve.md](docs/commands/solve.md)):

```bash
export FM=codes/chimes_lsq-LLfork/test_suite-lsq/test_4atoms.2

chimes-agent fm-setup-gen \
  --trjfile "$(pwd)/$FM/dump2.xyzf" --nframes 250 \
  --elements C,H --order '{"2":6,"3":2}' \
  --pair-cutoffs '{"C-C":[1.29,5.0],"C-H":[1.29,5.0],"H-H":[0.9,5.0]}' \
  --output-dir /tmp/chimes-quickstart

chimes-agent amat-build --fm-setup-in /tmp/chimes-quickstart/fm_setup.in \
  --output-dir /tmp/chimes-quickstart

chimes-agent solve --algorithm svd \
  --A /tmp/chimes-quickstart/A.txt --b /tmp/chimes-quickstart/b.txt \
  --header /tmp/chimes-quickstart/params.header --map /tmp/chimes-quickstart/ff_groups.map \
  --output-dir /tmp/chimes-quickstart

chimes-agent evaluate --params /tmp/chimes-quickstart/params.txt \
  --holdout-xyzf "$FM/dump2.xyzf" --max-frames 25 --json-out /tmp/chimes-quickstart/eval.json
```

(`evaluate`'s underlying C++ library prints its own verbose init logging
straight to stdout below the Python layer — pass `--json-out` rather than
parsing raw stdout if you're doing this from a script or agent; see
[docs/commands/evaluate.md](docs/commands/evaluate.md).)

`--describe` prints any stage's full input/output schema without running it:

```bash
chimes-agent solve --describe
```

For a full real-study walkthrough (holdout split → fit → DLARS solve on
HPC → evaluate → LAMMPS validate), see
[docs/tutorials/end_to_end_holdout_study.md](docs/tutorials/end_to_end_holdout_study.md).

---

## The stage contract

Every `chimes-agent <stage>` subcommand:

- takes input via `--json-in FILE` or discrete flags (never both silently —
  `--json-in` overrides matching flags),
- writes one JSON object to stdout or `--json-out FILE`,
- supports `--describe` to print its JSON Schema + summary without running,
- if it writes files, takes `--output-dir` and is **idempotent** there: a
  second invocation with the same inputs short-circuits and reprints the
  prior result instead of re-running; different inputs at the same
  `--output-dir` are refused unless you pass `--force` (this replaces
  al_driver's append-only, substring-parsed `restart.dat` with something an
  agent can actually introspect),
- if it submits to HPC, supports `--dry-run` to render (not submit) the
  sbatch script, and `--hpc {dane,stampede3,<your-profile.yaml>}` to pick
  the machine.

No stage calls another stage internally — composing them is the caller's
job. See [docs/concepts/stages_and_contracts.md](docs/concepts/stages_and_contracts.md).

---

## Commands

| Command | Status | Purpose |
|---|---|---|
| `setup` | **implemented** | Build/fetch chimes_lsq, chimes_calculator, LAMMPS, Quantum ESPRESSO for a machine |
| `fm-setup-gen` | **implemented** | Generate `fm_setup.in` from typed parameters (elements, cutoffs, order, fit flags) |
| `amat-build` | **implemented** (local or `--machine`) | Build A.txt/b.txt/dim.txt via the `chimes_lsq` binary |
| `solve` | **implemented** (local algorithms + dlars/dlasso via `--machine`, validated on a real Slurm job) | Solve for `params.txt` |
| `model-build` | **implemented** | Complete build: amat-build then solve, sequentially |
| `auto-build` | **implemented** | Full pipeline: unlabeled configs → QE labeling → data-driven cutoffs/λ → order sweep → one optimal model → optional AL stabilization |
| `dataset-select` | **implemented** | FPS / random / stratified-holdout sampling |
| `sweep` | **implemented** (local algorithms) | Grid sweep over 2b/3b/4b order, cutoffs, alpha/algorithm + comparison table (not auto-tuning) |
| `evaluate` | **implemented** | Holdout force/energy RMSE via the ctypes evaluator; multi-model committee spread |
| `lammps-run` | **implemented** (local) | Single-point/MD via the ChIMES-patched LAMMPS build |
| `qe-relabel` | **implemented** | Submit Quantum ESPRESSO single-point jobs; `--collect` converts output to `.xyzf` |
| `submit` | **implemented** | Generic Slurm submit/status/cancel/dry-run |
| `al-run` | **implemented** | Launch al_driver's own active-learning loop (`main.py`) as a detached background process |
| `al-select` | **implemented** | Diversity-based active-learning batch selection via al_driver's own `gen_subset` (wraps `gen_selections.py`); needs the `al-select` extra (`matplotlib`/`cycler`) |

All stages are implemented now. See
[docs/concepts/stages_and_contracts.md](docs/concepts/stages_and_contracts.md#phasing)
for the build history.

---

## Documentation

**Full searchable site: https://laubachb.github.io/Agentic-ChIMES/** (built
with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) from
the same files linked below; deployed by
[`.github/workflows/docs.yml`](.github/workflows/docs.yml) on push to
`master` — requires enabling **Settings → Pages → Source → GitHub Actions**
once for the repo). Build/browse it locally:

```bash
pip install -e ".[docs]"
mkdocs serve   # live-reloading dev server at http://127.0.0.1:8000
# or: mkdocs build --strict   # static site in ./site/
```

- [docs/concepts/stages_and_contracts.md](docs/concepts/stages_and_contracts.md) — the stage abstraction, manifest/idempotency, phasing plan
- [docs/concepts/vendored_forks.md](docs/concepts/vendored_forks.md) — why `codes/` is gitignored and cloned fresh, pinned commits, bumping a pin
- [docs/concepts/machine_profiles.md](docs/concepts/machine_profiles.md) — HPC machine profiles, adding your own cluster
- [docs/concepts/qm_driver_plugins.md](docs/concepts/qm_driver_plugins.md) — the QM-driver registry, how QE slots in, how to add another code
- [docs/concepts/units_and_conventions.md](docs/concepts/units_and_conventions.md) — unit conventions and the guardrails baked in as defaults (Dane `--ntasks-per-node`, lustre2 file-count quota, hartree/bohr↔eV/Å)
- [docs/concepts/cutoffs_and_lambdas.md](docs/concepts/cutoffs_and_lambdas.md) — which cutoff/λ/order choices are documented ChIMES practice vs. reasonable defaults, and how `auto-build` derives them from training data
- [docs/commands/](docs/commands/) — one page per subcommand
- [docs/tutorials/end_to_end_holdout_study.md](docs/tutorials/end_to_end_holdout_study.md) — a full real-study walkthrough

## Testing

```bash
pip install -e ".[dev]"
pytest tests/unit          # no HPC/allocation needed
```

Tests that need a built component (e.g. `chimescalc_lib` for `evaluate`)
skip cleanly if `chimes-agent setup` hasn't been run for that component yet,
rather than failing. HPC-dependent integration checks (a real Slurm
submission, a real DLARS solve, a real QE job) are not part of this suite —
see the verification section of [docs/concepts/stages_and_contracts.md](docs/concepts/stages_and_contracts.md).
