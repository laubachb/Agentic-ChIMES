# Agentic ChIMES

A tool-calling CLI and Python API for building
[ChIMES](https://chimes-lsq.readthedocs.io/) machine-learned interatomic
potentials end to end: dataset selection, DFT/QM relabeling, `fm_setup.in`
authoring, design-matrix generation, DLARS/LASSO solving, LAMMPS/holdout
evaluation, and Slurm submission on LLNL and TACC HPC systems.

**What this is:** every stage of the ChIMES model-building workflow exposed
as a discrete command with a structured JSON contract — `chimes-agent
<stage> --json-in in.json --json-out out.json`. A human or a coding agent
(e.g. a Claude Code session) calls one stage at a time, inspects the
result, and decides what to do next.

**What this is not:** an autonomous closed-loop orchestrator that runs a
multi-cycle active-learning campaign by itself. Hyperparameter choices
(regularization strength, basis order, dataset curation) stay a human/agent
judgment call made *between* stage invocations — this repo makes each of
those steps fast, reliable, and inspectable, not automatic.

[Source on GitHub :fontawesome-brands-github:](https://github.com/laubachb/Agentic-ChIMES){ .md-button }
[Get started →](getting_started.md){ .md-button .md-button--primary }

---

## Install

```bash
git clone git@github.com:laubachb/Agentic-ChIMES.git
cd Agentic-ChIMES
pip install -e .
chimes-agent setup --machine dane --component all
```

`setup` clones the three vendored ChIMES forks into `codes/` (gitignored —
never pushed to this repo, see [The vendored forks](concepts/vendored_forks.md))
and builds them, plus a from-scratch Quantum ESPRESSO clone+build, for
your machine. See [Getting started](getting_started.md) for the full walkthrough.

## Quickstart

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

Every stage supports `--describe` to print its full input/output schema
without running it:

```bash
chimes-agent solve --describe
```

## Where to go next

- **[Getting started](getting_started.md)** — install, build the toolchain, troubleshooting
- **[Stages and contracts](concepts/stages_and_contracts.md)** — the stage abstraction, manifest/idempotency, why it's built this way, and the phasing plan
- **[The vendored forks](concepts/vendored_forks.md)** — why `codes/` is gitignored and cloned fresh, pinned commits
- **[Machine profiles](concepts/machine_profiles.md)** — HPC machine profiles (Dane, Stampede3), adding your own cluster
- **[QM-driver plugins](concepts/qm_driver_plugins.md)** — the QM-driver registry design, how Quantum ESPRESSO slots in
- **[Units and conventions](concepts/units_and_conventions.md)** — unit conventions and the guardrails baked in as defaults
- **[Cutoffs and lambdas](concepts/cutoffs_and_lambdas.md)** — documented ChIMES cutoff/λ/order guidance vs. defaults, and how `auto-build` derives them from data
- **[Commands](commands/index.md)** — full reference, one page per subcommand
- **[Tutorial: end-to-end holdout study](tutorials/end_to_end_holdout_study.md)** — a full real-study walkthrough

## Repo layout

```
Agentic-ChIMES/
├── codes/                    # gitignored -- cloned fresh by `chimes-agent setup`
├── deps/                     # gitignored -- built/fetched by `chimes-agent setup`
├── src/agentic_chimes/        # the CLI + orchestration layer
├── tests/unit/                # no HPC/allocation needed
└── docs/                      # this site
```
