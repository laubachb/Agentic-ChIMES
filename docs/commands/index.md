# Commands

Every `chimes-agent <stage>` subcommand takes input via `--json-in FILE` or
discrete flags, prints one JSON object to stdout or `--json-out FILE`, and
supports `--describe` to print its full schema without running. See
[Stages and contracts](../concepts/stages_and_contracts.md) for the shared
contract every page below assumes.

| Command | Status | Purpose |
|---|---|---|
| [`study`](study.md) | **implemented** | Create a study directory, register phase artifacts, show status |
| [`usage`](usage.md) | **implemented** | CPU-hours a study used (Slurm via sacct + login-node ledger), charged vs used, by phase |
| [`benchmark`](benchmark.md) | **implemented** | Strong/weak LAMMPS scaling of the final model → cost model and CPU-hour estimates |
| [`deploy`](deploy.md) | **implemented** | Package the model: params, LAMMPS example, MODEL_CARD.md |
| [`study-report`](study-report.md) | **implemented** | Collate a study into REPORT_FACTS.json + REPORT.md |
| [`setup`](setup.md) | **implemented** | Clone the vendored forks + build/fetch chimes_lsq, chimes_calculator, LAMMPS, Quantum ESPRESSO for a machine |
| [`doctor`](doctor.md) | **implemented** | Check components, numerics, machine profile and credentials; print fixes |
| [`hyper-analyze`](hyper-analyze.md) | **implemented** | Per-pair distances/RDF → inner cutoffs, Morse lambdas, outer-cutoff candidates, N_LAYERS |
| [`hyper-search`](hyper-search.md) | **implemented** | Staged search over cutoffs, lambdas, 2b/3b/4b orders; cheapest model statistically tied with the best |
| [`fm-setup-gen`](fm-setup-gen.md) | **implemented** | Generate `fm_setup.in` from typed parameters (elements, cutoffs, order, fit flags) |
| [`amat-build`](amat-build.md) | **implemented** (local or `--machine`) | Build A.txt/b.txt/dim.txt via the `chimes_lsq` binary |
| [`solve`](solve.md) | **implemented** (local algorithms + dlars/dlasso via `--machine`, validated on a real Slurm job) | Solve for `params.txt` |
| [`weights`](weights.md) | **implemented** | Per-row fitting weights: al_driver methods + published presets, AL decay |
| [`model-build`](model-build.md) | **implemented** | Complete build: amat-build then solve, sequentially |
| [`auto-build`](auto-build.md) | **implemented** | Full pipeline: unlabeled configs → QE labeling → data-driven cutoffs/λ → order sweep → one optimal model → optional AL stabilization |
| [`data-search`](data-search.md) | **implemented** | Search open DFT datasets (ColabFit on Hugging Face) for a chemical system |
| [`data-fetch`](data-fetch.md) | **implemented** | Fetch matching configurations from a dataset or local DFT files into one `.xyzf` + provenance |
| [`data-generate`](data-generate.md) | **implemented** | Strained/rattled/substituted supercells for QE labeling |
| [`data-curate`](data-curate.md) | **implemented** | Filter, analyze coverage, subsample and split into a base dataset + `data_manifest.json` |
| [`dataset-select`](dataset-select.md) | **implemented** | FPS / random / stratified-holdout sampling |
| [`sweep`](sweep.md) | **implemented** (local algorithms) | Grid sweep over 2b/3b/4b order, cutoffs, alpha/algorithm |
| [`evaluate`](evaluate.md) | **implemented** | Holdout force/energy RMSE via the ctypes evaluator; multi-model committee spread |
| [`lammps-run`](lammps-run.md) | **implemented** (local) | Single-point/MD via the ChIMES-patched LAMMPS build (triclinic; thin cells replicated) |
| [`md-check`](md-check.md) | **implemented** (local or one Slurm job) | Short MD of candidate models: stability, close contacts, RDF vs reference, AL harvest |
| [`qe-relabel`](qe-relabel.md) | **implemented** | Submit Quantum ESPRESSO single-point jobs; convert output to `.xyzf` |
| [`submit`](submit.md) | **implemented** | Generic Slurm submit/status/cancel/dry-run |
| [`al-run`](al-run.md) | **implemented** | Launch al_driver's own active-learning loop as a detached background process |
| [`al-select`](al-select.md) | **implemented** | Diversity-based active-learning batch selection via al_driver's own `gen_subset` |

All stages are implemented.
