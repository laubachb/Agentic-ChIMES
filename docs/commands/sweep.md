# `chimes-agent sweep`

**Status: implemented (local algorithms; same scope as `solve`'s
dlars/dlasso limitation).**

Grid sweep over ChIMES hyperparameters, composing `fm-setup-gen` +
`model-build` (amat-build+solve) + `evaluate` for each grid point and
reporting a comparison table. **Not an auto-tuner** — consistent with this
repo's design stance that hyperparameter choices stay a human/agent
judgment call: this produces a table (JSON + CSV) and a `best_by` pointer
per metric to look at, it does not pick a "best" model as a final answer.
See `src/agentic_chimes/stages/sweep.py`.

## Sweepable dimensions

Any subset of `grid`'s keys (cartesian product across whichever are
given; anything not given uses `base`'s value):

| Key | Meaning |
|---|---|
| `order_2b`, `order_3b`, `order_4b` | independently swept Chebyshev orders. `order_4b` entries of `null`/`0` mean "no 4-body term" for that point (matches `fm-setup-gen`'s own convention) |
| `default_s_minim`, `default_s_maxim` | inner/outer cutoff fallback (same as `fm-setup-gen`'s `--default-s-minim`/`--default-s-maxim`) |
| `alpha` | `solve`'s regularization strength |
| `algorithm` | `solve`'s algorithm choice |

## Usage

```bash
cat > sweep.json <<'EOF'
{
  "base": {
    "trjfile": "/abs/path/train.xyzf", "nframes": 250,
    "elements": ["C", "H"],
    "pair_cutoffs": {"C-C": [1.29, 5.0], "C-H": [1.29, 5.0], "H-H": [0.9, 5.0]},
    "algorithm": "lassolars"
  },
  "grid": {
    "order_2b": [8, 12], "order_3b": [3, 5], "order_4b": [null, 2],
    "alpha": [1e-6, 1e-5, 1e-4]
  },
  "holdout_xyzf": "/abs/path/holdout.xyzf"
}
EOF
chimes-agent sweep --json-in sweep.json --output-dir ./sweep_run
```

Each grid point runs sequentially into its own `./sweep_run/point_NNNN/`
(no parallelism in this phase). A failed point (e.g. a solve that doesn't
converge) is recorded with `"status": "failed"` and doesn't stop the rest
of the sweep.

## `base`

Everything `fm-setup-gen` + `solve` + `evaluate` accept — see those
pages' `--describe` output. Required: `trjfile`, `nframes`, `elements`.
Recommended: `pair_cutoffs` (or the sweep will use `default_s_minim`/
`default_s_maxim`, i.e. the same cutoff for every pair); `special_maxim_3b`/
`special_maxim_4b` if you want a shorter 3-/4-body outer cutoff (see
[Cutoffs and lambdas](../concepts/cutoffs_and_lambdas.md)).

For picking cutoffs/λ *and* the order grid automatically from your
training data in one call, see [`auto-build`](auto-build.md) — it's
`sweep` plus the data-driven parameter derivation and a final pick.

## Output

```json
{
  "n_points": 12, "n_done": 12, "n_failed": 0,
  "results": [
    {"index": 0, "overrides": {"order_2b": 8, "order_3b": 3, "alpha": 1e-6}, "status": "done",
     "params": "./sweep_run/point_0000/params.txt",
     "rmse_force_kcal_mol_ang": 3.21, "rmse_energy_kcal_mol": 0.9, "wall_time_s": 12.4}
  ],
  "table_csv": "./sweep_run/sweep_results.csv",
  "best_by": {"rmse_force": 0, "rmse_energy": 3}
}
```

`best_by`'s values are `results` indices, not a recommendation to act on
automatically — open `sweep_results.csv` and decide.
