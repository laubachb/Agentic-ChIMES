---
name: chimes-benchmarking
description: Account for the compute a ChIMES study used (CPU-hours by phase and job, allocation efficiency) and measure the final model's strong and weak scaling in LAMMPS to size downstream compute requests. Use when the user asks how many CPU-hours something took, how a model scales, how much to request for a production MD run, or when a study reaches its benchmark phase. The chimes-benchmark subagent follows this playbook.
---

# Benchmarking and compute accounting

Two questions, two tools:

- **What did developing this model cost?** `chimes-agent usage --study <dir>`
- **What will running it cost?** `chimes-agent benchmark` (submit, then `--collect`)

## Compute used (`usage`)

- Finds every Slurm job whose working directory is inside the study (or a
  registered `extra_root`) through `sacct`, including jobs al_driver
  submits itself, plus login-node stage runs from `usage/local.jsonl` (the
  CLI records those automatically when a stage writes inside a study).
- Report both numbers: **charged** CPU-hours (allocated cores × elapsed,
  what the allocation pays) and **used** CPU-hours (CPU the processes
  consumed). Their ratio, `allocation_efficiency`, shows waste. On the Cu-Zr
  development study it was ~2 %: 112-core nodes held for ~30 short fits.
  `hyper-search --machine` now requests only the cores its largest stage
  uses; recommend the same right-sizing elsewhere.
- Jobs outside the standard layout: register their directory
  (`study --study S --extra-root DIR`) and pass `--phase-hints` so they are
  attributed to the right phase.

## Scaling (`benchmark`)

1. **Structure**: orthorhombic, representative of production runs.
   `--prototype` (ase bulk kwargs, e.g. B2 `{"name":"CuZr","crystalstructure":"cesiumchloride","a":3.26}`)
   or a `data-generate` frame. Check it is stable with the model first: one
   single-rank case locally takes seconds (see below).
2. **Sizes**: cost per atom-step is roughly constant, so estimate first:
   one ~250-atom, 100-step single-rank run gives core-s/atom-step; the
   single-rank strong case costs that × `strong_atoms` × 110 steps. Keep it
   under ~10 minutes (e.g. `--strong-atoms 8000` for a ~3e-4 core-s model).
3. **Ranks**: default 1…112 on one node. Add `--nodes 2,4` only if the user
   will run multi-node production, and say what it costs.
4. `--dry-run`, show the case list and job, submit on approval (it is an HPC
   job), wait via `chimes-job-monitor`, then `--collect DIR`.

## Reading the result (`benchmark.json`)

- `strong.rows`: speedup and efficiency vs the fewest ranks;
  `recommended_ranks` = most ranks still ≥ `min_efficiency` (0.7). Beyond
  it, extra ranks mostly add communication.
- `weak.rows`: efficiency at fixed atoms/rank; near 1 means cost scales
  linearly with system size.
- `cost_model.core_s_per_atom_step`: the unit for every estimate.
  CPU-hours = unit × atoms × steps / 3600.
- `estimates_1ns`: ready-made 1,000-1,000,000-atom, 1 ns requests with
  suggested ranks/nodes and wall time.
- `unstable`/`failed` cases: an unstable case means the model blew up in MD
  for that structure. That is a model finding, not a benchmark bug; report it.

Context for the reader (`chimes-literature`, "Accuracy and cost
benchmarks"): for 256-atom liquid carbon over 5 ps, DFT cost about
50,000 CPU-h, ChIMES 2-body 0.15 CPU-h and ChIMES 3-body 5 CPU-h (Lindsey
2017). Many-body terms and cutoff length dominate cost. Quoting the model's
cost per atom-step next to the DFT cost it replaces is the usual way to
state the speedup.

Write `05_bench/BENCHMARK.md`: the scaling tables, the cost unit, a sizing
recipe for the user's intended runs (atoms, ns, timestep → CPU-hours, ranks,
nodes, wall time, with ~20 % margin), and the development-cost summary with
concrete savings for the next study. Register results:
`study --register benchmark=<dir>/benchmark.json --register usage=<study>/usage/usage_report.json`.

Reference: `docs/commands/usage.md`, `docs/commands/benchmark.md`.
