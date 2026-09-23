# `chimes-agent evaluate`

**Status: implemented.**

Computes holdout force/energy RMSE for one or more `params.txt` models,
in-process via `chimes_calculator`'s serial ctypes API
(`serial_interface/api/chimescalc_serial_py.py`, loaded from its vendored
path) — no subprocess, no Slurm. Validated against
`codes/chimes_calculator-LLfork/serial_interface/tests/` fixtures (a
published force field + config with known reference energy/forces; see
`tests/unit/test_evaluate.py`).

## Usage

```bash
chimes-agent evaluate --params ./run1/params.txt \
  --holdout-xyzf ./holdout.xyzf --max-frames 100
```

Committee mode (multiple models on the same frames):

```bash
chimes-agent evaluate --params modelA/params.txt --params modelB/params.txt \
  --holdout-xyzf ./holdout.xyzf
```

## Flags

- `--params PATH` (repeatable; >1 = committee mode)
- `--holdout-xyzf PATH` (required) — a ChIMES `.xyzf` file with reference
  forces/energy (see `io/xyzf.py`)
- `--max-frames N` — cap the number of frames evaluated

## Output

```json
{
  "n_frames": 100,
  "n_models": 1,
  "results": [
    { "params": "./run1/params.txt", "rmse_force_kcal_mol_ang": 3.21, "rmse_energy_kcal_mol": 0.9 }
  ],
  "committee_spread": null
}
```

With `n_models > 1`, `committee_spread.per_frame_energy_stdev` is the
stdev across models of each frame's predicted energy — this is the
concrete plug-in point for a future uncertainty-based active-learning
selector (diff models on unlabeled candidate frames, pick the
highest-disagreement ones), deliberately not wired into `al-select` yet.
See `docs/concepts/qm_driver_plugins.md` for why al_driver's own AL
selection is diversity-based, not uncertainty-based, today.

## Prerequisite

Needs `chimescalc_lib` built: `chimes-agent setup --component
chimes_calculator --machine <name>`.

## Known noise: use `--json-out`, not raw stdout, for programmatic parsing

The vendored ChIMES calculator library prints its own verbose init logging
(parameter file contents, pair/cluster maps, ...) directly to the process's
real stdout via C++ `std::cout` on `init_chimes_instance` -- this happens
below the Python layer, so `evaluate` cannot cleanly suppress it. That
logging lands *before* this stage's JSON result on stdout. A human reading
the terminal can just look at the last JSON object; a script or agent
parsing stdout as JSON should instead pass `--json-out result.json` and
read that file, which contains only the JSON result.
