# `chimes-agent dataset-select`

**Status: planned (Phase 3), not yet implemented.** The subcommand exists
today as a stub (`stages/dataset_select.py`) that echoes its parsed input —
`--describe` works, the CLI contract below is fixed.

## Intent

FPS (farthest-point sampling) / holdout-split dataset sampling over a frame
pool, formalizing prior hand-rolled `make_holdout_split.py`-style scripts.
The `stratified_holdout` method is planned to optionally reuse the
Metropolis-MC energy-histogram-flattening mechanism already implemented in
`codes/al_driver-LLfork/src/gen_selections.py:gen_subset`, generalized off
its AL-cycle-specific "central repository" framing.

## Planned schema

```json
{
  "frames": "path to packed .xyzf/.npz frame pool",
  "method": "fps | random | stratified_holdout",
  "n_select": 100,
  "holdout_fraction": 0.2,
  "seed": 42
}
```
