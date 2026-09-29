# Compute: approvals, accounting and sizing

## Where work runs

- **Login node:** only quick things: searches of dataset catalogs, curation,
  single fits, analysis, reports.
- **Slurm:** QE labeling, hyperparameter searches, DLARS solves, al_driver,
  benchmarks, production MD. Always as a dry run first, then with your
  approval.
- **Shared filesystem:** studies must live where compute nodes can read and
  write them (`/p/lustre2/...` on LC). A job in `/tmp` "completes" with no
  output.

## What development costs: `chimes-agent usage`

`usage` counts every Slurm job whose working directory is inside the study
(via `sacct`), including jobs al_driver submits itself, plus login-node stage
runs, which the CLI records automatically. It reports two numbers per phase
and job:

- **charged CPU-hours**: allocated cores × elapsed time, which is what your
  bank pays;
- **used CPU-hours**: CPU time actually consumed.

Their ratio is the allocation efficiency. In the Cu-Zr development study it
was about 2 %, because hyperparameter searches held full 112-core nodes for
~30 short fits. `hyper-search --machine` now requests only as many cores as
its largest stage can use. Look at this number after every study; it is the
easiest saving.

## What running the model costs: `chimes-agent benchmark`

The benchmark runs short LAMMPS MD with the final model:

- **strong scaling:** a fixed system size on 1 to 112+ ranks;
- **weak scaling:** fixed atoms per rank.

From these it derives the cost unit, **core-seconds per atom-step**, and
the largest efficient rank count.

Sizing a production run:

```
CPU-hours ≈ core_s_per_atom_step × atoms × (simulated time / timestep) / 3600
ranks     ≈ atoms / atoms_per_rank (from the benchmark)      nodes = ranks / cores per node
```

Add ~20 % margin. The benchmark's `estimates_1ns` table has this worked out
for 10³-10⁶ atoms. Cost per atom-step depends on the model: longer 3-body
cutoffs and more coefficients raise it steeply. That's why the
hyperparameter search breaks ties in favor of cheaper models, estimating
MD cost per atom as clusters within each cutoff × coefficients.

## Keeping costs down

- Let `hyper-search` right-size its allocation, and keep 4-body sweeps light
  (they scale steeply: quartets grow as r⁹).
- Use `pdebug` for anything under an hour.
- Curate before labeling: `data-curate --target-size` picks a diverse subset,
  so fewer QE jobs are needed.
- Reuse cached fits: widen a search grid instead of starting over.
