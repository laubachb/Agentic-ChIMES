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
| [`amat-build`](amat-build.md) | **implemented** (local or `--machine`) | Build A.txt/b.txt/dim.txt via the `chimes_lsq` binary |
| [`solve`](solve.md) | **implemented** (local algorithms + dlars/dlasso via `--machine`, validated on a real Slurm job) | Solve for `params.txt` |
| [`model-build`](model-build.md) | **implemented** | Complete build: amat-build then solve, sequentially |
| [`auto-build`](auto-build.md) | **implemented** | Full pipeline: unlabeled configs → QE labeling → data-driven cutoffs/λ → order sweep → one optimal model → optional AL stabilization |
| [`dataset-select`](dataset-select.md) | **implemented** | FPS / random / stratified-holdout sampling |
| [`sweep`](sweep.md) | **implemented** (local algorithms) | Grid sweep over 2b/3b/4b order, cutoffs, alpha/algorithm |
| [`evaluate`](evaluate.md) | **implemented** | Holdout force/energy RMSE via the ctypes evaluator; multi-model committee spread |
| [`lammps-run`](lammps-run.md) | **implemented** (local) | Single-point/MD via the ChIMES-patched LAMMPS build |
| [`qe-relabel`](qe-relabel.md) | **implemented** | Submit Quantum ESPRESSO single-point jobs; convert output to `.xyzf` |
| [`submit`](submit.md) | **implemented** | Generic Slurm submit/status/cancel/dry-run |
| [`al-run`](al-run.md) | **implemented** | Launch al_driver's own active-learning loop as a detached background process |
| [`al-select`](al-select.md) | **implemented** | Diversity-based active-learning batch selection via al_driver's own `gen_subset` |

All stages are implemented now.
