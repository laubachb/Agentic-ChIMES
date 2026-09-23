# `chimes-agent sweep`

**Status: planned (Phase 3), not yet implemented.** The subcommand exists
today as a stub (`stages/sweep.py`) that echoes its parsed input —
`--describe` works, the CLI contract below is fixed.

## Intent

A grid sweep over a small explicit set of hyperparameters (λ/alpha, basis
order, cutoffs), each point run via `fm-setup-gen` → `amat-build` → `solve`
→ `evaluate` in sequence (calling those stages' `run()` functions directly,
not re-invoking the CLI as subprocesses), collecting one comparison table.
**This is a grid sweep + reporting tool, not an auto-tuner** — consistent
with this repo's design stance that hyperparameter choices stay a
human/agent judgment call: `sweep` produces a table to look at, it does not
pick a "best" model as a final answer.

## Planned schema

```json
{
  "base_fm_setup": "path to a template fm_setup.in",
  "grid": {"alpha": [1e-6, 1e-5, 1e-4], "order": [{"2":12,"3":5}, {"2":12,"3":4}]},
  "holdout_xyzf": "path",
  "machine": "dane",
  "max_parallel_jobs": 4
}
```

Planned output: one row per grid point (`model_path`,
`holdout_rmse_force`, `holdout_rmse_energy`, `n_nonzero_params`,
`wall_time_s`) plus a CSV/plot, and `best_by` pointers (e.g. lowest RMSE,
sparsest model) as suggestions, not automatic choices.
