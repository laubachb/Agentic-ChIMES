# Commands

Every `chimes-agent <stage>` subcommand takes input via `--json-in FILE` or
discrete flags, prints one JSON object to stdout or `--json-out FILE`, and
supports `--describe` to print its full schema without running. See
[Stages and contracts](../concepts/stages_and_contracts.md) for the shared
contract every page below assumes.

| Command | Status | Purpose |
|---|---|---|
| [`setup`](setup.md) | **implemented** | Clone the vendored forks + build/fetch chimes_lsq, chimes_calculator, LAMMPS, Quantum ESPRESSO for a machine |
| [`fm-setup-gen`](fm-setup-gen.md) | **implemented** | Generate `fm_setup.in` from typed parameters (elements, cutoffs, order, fit flags) |
| [`amat-build`](amat-build.md) | **implemented** (local only) | Build A.txt/b.txt/dim.txt via the `chimes_lsq` binary |
| [`solve`](solve.md) | **implemented** (svd/ridge/lassolars/...; dlars/dlasso pending HPC layer) | Solve for `params.txt` |
| [`evaluate`](evaluate.md) | **implemented** | Holdout force/energy RMSE via the ctypes evaluator; multi-model committee spread |
| [`submit`](submit.md) | **implemented** | Generic Slurm submit/status/cancel/dry-run |
| [`dataset-select`](dataset-select.md) | planned | FPS / holdout-split sampling |
| [`sweep`](sweep.md) | planned | Grid sweep over hyperparameters + comparison table (not auto-tuning) |
| [`lammps-run`](lammps-run.md) | planned | Single-point/MD via the ChIMES-patched LAMMPS build |
| [`qe-relabel`](qe-relabel.md) | planned | Submit Quantum ESPRESSO jobs; convert output to `.xyzf` |
| [`al-select`](al-select.md) | planned | Diversity-based active-learning batch selection |

Stages marked "planned" already register a real subcommand — `--describe`
works, the CLI contract is fixed — that echoes its parsed input back.
Swapping the stub for real logic never changes how the stage is invoked.
