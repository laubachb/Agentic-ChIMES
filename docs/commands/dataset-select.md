# `chimes-agent dataset-select`

**Status: implemented.**

FPS / random / stratified-holdout sampling over a `.xyzf` frame pool,
formalizing prior hand-rolled `make_holdout_split.py`-style scripts. See
`src/agentic_chimes/stages/dataset_select.py`.

## Usage

```bash
# farthest-point sampling: pick 100 maximally-diverse frames from a pool
chimes-agent dataset-select --frames pool.xyzf --method fps \
  --n-select 100 --descriptor composition --output-dir ./split

# random split by holdout fraction
chimes-agent dataset-select --frames pool.xyzf --method random \
  --holdout-fraction 0.2 --output-dir ./split

# stratified: every composition class represented in both splits
chimes-agent dataset-select --frames pool.xyzf --method stratified_holdout \
  --holdout-fraction 0.2 --output-dir ./split
```

## Methods

- **`fps`** — greedy farthest-point sampling in descriptor space. Picks
  `n_select` frames maximizing minimum pairwise distance; the complement is
  returned as `holdout_indices` but isn't a meaningful test set on its own
  (FPS is normally used to subsample a *training* pool, e.g. for a
  retention-curve study, not to carve out held-out test data).
- **`random`** — plain seeded random sample.
- **`stratified_holdout`** — bins frames by composition class (the set of
  elements present, e.g. `("C",)` vs `("C","H")`) and splits proportionally
  within each class, so every class observed in the pool appears in both
  the selected and holdout sets. The same mixed/pure stratification
  pattern used in prior HEA/binary-alloy studies with this toolchain,
  generalized to any element set.

## The `fps` descriptor

Deliberately simple and self-contained — **not** the paper's full
cluster-graph structural fingerprint (that needs a LAMMPS fingerprint
build; see `docs/concepts/qm_driver_plugins.md`'s committee-spread note for
where a heavier descriptor could plug in later):

- `composition` (default) — per-element atom-count fraction.
- `energy` — per-atom energy, z-score standardized. Requires every frame
  in the pool to have an energy (`fm_setup.in`'s `FITENER`/xyzf convention)
  — raises a clear error naming a missing frame otherwise.
- `composition_energy` — both, concatenated (each independently
  standardized first so one doesn't dominate the Euclidean distance).

## Flags

- `--frames PATH` (required)
- `--method {fps,random,stratified_holdout}` (required)
- `--n-select N` or `--holdout-fraction F` (one required; `n_select` is the
  size of the selected/train set, `holdout_fraction` is the complementary
  way to express it: `n_select = round(n_pool * (1 - holdout_fraction))`)
- `--seed` (default 42)
- `--descriptor {composition,energy,composition_energy}` (default
  `composition`; `fps` only)

## Output

```json
{
  "method": "fps",
  "n_pool": 251, "n_selected": 50, "n_holdout": 201,
  "selected_indices": [...], "holdout_indices": [...],
  "selected_xyzf": "./split/selected.xyzf",
  "holdout_xyzf": "./split/holdout.xyzf",
  "indices_json": "./split/indices.json"
}
```
