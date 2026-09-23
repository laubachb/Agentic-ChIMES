# Getting started

## 1. Install the CLI

```bash
git clone <this-repo>
cd Agentic-ChIMES
pip install -e ".[dev]"
```

This installs `chimes-agent` plus `numpy`/`pyyaml`/`jsonschema` (and
`pytest` for the dev extra). It does **not** fetch or build any of the
vendored ChIMES toolchain — `codes/` is gitignored and doesn't exist yet
after this step (see [docs/concepts/vendored_forks.md](concepts/vendored_forks.md)),
and building involves compiling C++ against your machine's MPI/MKL modules
— both are handled by the next step.

## 2. Build the toolchain

```bash
chimes-agent setup --machine dane --component all
```

Pick `dane` (LLNL LC) or `stampede3` (TACC), or write your own machine
profile (see [docs/concepts/machine_profiles.md](concepts/machine_profiles.md)).
This runs each vendored fork's own `install.sh` under the right modules,
plus a from-scratch Quantum ESPRESSO clone+build. It can take a while
(LAMMPS-from-source is the slowest piece); build one component at a time
with `--component chimes_lsq` / `chimes_calculator` / `lammps` /
`quantum_espresso` if you don't need all of them yet.

Check what's built:

```bash
chimes-agent setup --status
```

## 3. Run the quickstart

See the [Quickstart](index.md#quickstart)
for a full `fm-setup-gen` → `amat-build` → `solve` → `evaluate` chain
against a bundled ChIMES test fixture — no HPC allocation needed.

## 4. Explore a stage's contract

```bash
chimes-agent --help                 # list every subcommand
chimes-agent solve --describe       # full input/output schema for one stage, without running it
```

## Troubleshooting

- **`chimes-agent: command not found` after `pip install -e .`** — on a
  shared/module Python where you can't write to site-packages, pip installs
  to `~/.local/bin` and warns that it's not on `PATH`; add `export
  PATH="$HOME/.local/bin:$PATH"` to your shell profile (or run via `python3
  -m agentic_chimes.cli <stage> ...` instead).
- **`ComponentNotInstalled` error from a stage** — run `chimes-agent setup
  --component <name> --machine <name>` for whatever it's asking for, or set
  the `AGENTIC_CHIMES_*_BIN`/`AGENTIC_CHIMES_*_LIB` env var it names to an
  existing binary/library path.
- **`solve` fails with an import error inside `chimes_lsq.py`** — that
  script needs `numpy`/`scipy`/`scikit-learn` in whatever Python
  environment runs `chimes-agent` (it's invoked via `sys.executable`, i.e.
  the same interpreter `chimes-agent` itself runs under) — activate your
  machine's conda env (e.g. `mat_mcts`-style) before running `chimes-agent`.
- **A stage's manifest refuses to re-run** ("was written for different
  inputs") — you changed a parameter at the same `--output-dir`; pass
  `--force` to overwrite, or pick a new `--output-dir`.
- **Something about a bare `-N 1` / an under-allocated Dane job** — see the
  guardrail described in
  [docs/concepts/machine_profiles.md](concepts/machine_profiles.md#the-dane-gotcha-this-profile-exists-to-close);
  it should not be possible to hit this via `chimes-agent submit`/`hpc.slurm`,
  but if you're calling `create_and_launch_job` directly it's on you to set
  `job_ppn`.
