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
| [`learning-curve`](learning-curve.md) | **implemented** | Holdout error vs training size for one basis: data-limited or plateau |
| [`fm-setup-gen`](fm-setup-gen.md) | **implemented** | Generate `fm_setup.in` from typed parameters (elements, cutoffs, order, fit flags) |
| [`amat-build`](amat-build.md) | **implemented** (local or `--machine`) | Build A.txt/b.txt/dim.txt via the `chimes_lsq` binary |
| [`solve`](solve.md) | **implemented** (local algorithms + dlars/dlasso via `--machine`, validated on a real Slurm job) | Solve for `params.txt` |
| [`weights`](weights.md) | **implemented** | Per-row fitting weights: al_driver methods + published presets, AL decay |
| [`hierarch`](hierarch.md) | **implemented** | Hierarchical fitting: subtract fixed element models, merge cross + element params |
| [`model-build`](model-build.md) | **implemented** | Complete build: amat-build then solve, sequentially |
| [`auto-build`](auto-build.md) | **implemented** | Full pipeline: unlabeled configs → QE labeling → data-driven cutoffs/λ → order sweep → one optimal model → optional AL stabilization |
| [`data-search`](data-search.md) | **implemented** | Search open DFT datasets (ColabFit on Hugging Face) for a chemical system |
| [`data-fetch`](data-fetch.md) | **implemented** | Fetch matching configurations from a dataset or local DFT files into one `.xyzf` + provenance |
| [`data-generate`](data-generate.md) | **implemented** | Strained/rattled/substituted supercells for QE labeling |
| [`data-curate`](data-curate.md) | **implemented** | Filter, analyze coverage, subsample and split into a base dataset + `data_manifest.json` |
| [`fingerprint`](fingerprint.md) | **implemented** | Cluster-graph fingerprints: dataset coverage (D²), frame novelty, AL stopping |
| [`quests`](quests.md) | **implemented** (extra `quests`) | QUESTS entropy/diversity, per-environment novelty (dH), entropy-maximizing selection |
| [`dataset-select`](dataset-select.md) | **implemented** | FPS / random / stratified-holdout sampling |
| [`sweep`](sweep.md) | **implemented** (local algorithms) | Grid sweep over 2b/3b/4b order, cutoffs, alpha/algorithm |
| [`evaluate`](evaluate.md) | **implemented** | Holdout force/energy RMSE via the ctypes evaluator; multi-model committee spread |
| [`lammps-run`](lammps-run.md) | **implemented** (local) | Single-point/MD via the ChIMES-patched LAMMPS build (triclinic; thin cells replicated) |
| [`md-check`](md-check.md) | **implemented** (local or one Slurm job) | Short MD of candidate models: stability, close contacts, RDF vs reference, AL harvest |
| [`eos-check`](eos-check.md) | **implemented** (local) | Equation of state (V0, B0, B0') and clamped-ion elastic constants; Born stability |
| [`qe-relabel`](qe-relabel.md) | **implemented** | Submit Quantum ESPRESSO single-point jobs; convert output to `.xyzf` |
| [`qe-converge`](qe-converge.md) | **implemented** | Converge QE ecutwfc / k-spacing on one frame before labeling |
| [`submit`](submit.md) | **implemented** | Generic Slurm submit/status/cancel/dry-run |
| [`job-status`](job-status.md) | **implemented** | Verdict on a submitted job: succeeded / failed (why, fix) / completed without results |
| [`al-run`](al-run.md) | **implemented** | Launch al_driver's own active-learning loop as a detached background process |
| [`al-select`](al-select.md) | **implemented** | Diversity-based active-learning batch selection via al_driver's own `gen_subset` |
| [`al-merge`](al-merge.md) | **implemented** | Merge labeled AL frames into the training set (theory check, fixed holdout, cycles) |
| [`al-batch`](al-batch.md) | **implemented** | One AL batch from close contacts + novelty (QUESTS, fingerprint) + committee, within a budget |
| [`al-status`](al-status.md) | **implemented** | Per-round errors/stability/novelty across an AL campaign; CONVERGED/CONTINUE verdict |
| [`committee`](committee.md) | **implemented** | Bootstrap committee of one basis; candidates ranked by model disagreement |

All stages are implemented.
