# `chimes-agent evaluate`

**Status: implemented.**

Computes holdout force and energy errors for one or more `params.txt`
models, in-process via `chimes_calculator`'s serial ctypes API
(`serial_interface/api/chimescalc_serial_py.py`, loaded from its vendored
path). No subprocess, no Slurm.

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
- `--holdout-xyzf PATH` (required): a ChIMES `.xyzf` with reference forces
  (hartree/bohr) and, optionally, energies (kcal/mol)
- `--max-frames N`: cap the number of frames evaluated
- `--per-frame`: also return each frame's force squared-error sum,
  reference squared sum and component count (used by `hyper-search` for
  bootstrap uncertainty)

## Output

```json
{
  "n_frames": 30,
  "n_models": 1,
  "reference_force_rms_kcal_mol_ang": 6.58,
  "results": [
    {"params": "./run1/params.txt",
     "rmse_force_kcal_mol_ang": 2.90,
     "relative_force_error": 0.44,
     "rmse_energy_kcal_mol": 2.31,
     "rmse_energy_kcal_mol_per_atom": 0.92}
  ],
  "committee_spread": null
}
```

- `relative_force_error` = force RMSE ÷ the RMS of the reference forces. It
  is comparable across datasets; 1.0 means the model predicts nothing
  better than zero force.
- `reduced_force_rmse` = force RMSE ÷ the mean absolute reference force
  component: the "reduced RMSE" used throughout the ChIMES literature
  (0.24-0.31 for water, 0.44 → 0.28 for molten carbon 2017 → 2024; see
  [literature](../concepts/literature.md)). Use it when comparing with
  published models. It is larger than `relative_force_error` by a factor that
  depends on the force distribution (1.48× on the Cu-Zr example, ~1.25× for a
  Gaussian).
- `rmse_energy_kcal_mol_per_atom` is the energy error divided by each
  frame's atom count before averaging. Use it when frame sizes differ.

With `n_models > 1`, `committee_spread.per_frame_energy_stdev` is the
spread across models of each frame's predicted energy: the plug-in point
for a future uncertainty-based active-learning selector. It is not used by
`al-select`; see `docs/concepts/qm_driver_plugins.md`.

### Stresses and the penalty region

- `rmse_stress_gpa` (all six components) and `rmse_pressure_gpa` are
  reported when the holdout frames carry stresses (GPa, pressure sign).
  `n_frames_with_stress` counts them. chimes_calculator's stress is −dE/dV
  (checked by finite differences to 1e-7), converted from kcal/mol/Å³.
- `n_frames_below_inner_cutoff`: holdout frames with a contact inside the
  model's inner cutoff. There the repulsive penalty dominates the error. A
  warning says so, because the fix is in the data split, not the model.

### Per composition and per element

Every result carries:

- `by_composition`: relative force error for each element set, e.g.
  `Cu`, `Zr`, `Cu-Zr`;
- `by_element`: per atom type;
- `worst_frames`: the five frames with the largest relative error (likely
  mislabeled frames or coverage holes).

`--per-frame` adds `per_frame_group`. On Cu-Zr the pooled 0.31 hid alloy
frames at 0.43 vs pure Cu at 0.15.

Frames containing an element a model does not describe are refused up
front. chimes_calculator calls `exit()` inside the library on an unknown
type, which used to end the process silently with status 0.

### Plots

`--plot` with `--output-dir` writes `parity_forces.png` (predicted vs
reference force components, colored by element) and
`error_by_composition.png`. Other stages (cross-validation, learning
curves, `al-status`) score frames through `evaluate.evaluate_frames()`
without temporary files.

## Units

Training/holdout `.xyzf` forces are **hartree/bohr** (ChIMES
`doc/source/units.rst`); `chimes_calculator` predicts **kcal/mol/Å**.
Reference forces are converted (×1185.8) before comparing. Energies are
kcal/mol on both sides.

> **Corrected bug.** Before this fix, `evaluate` subtracted the two without
> converting, so its "force RMSE" was essentially the RMS of the
> *predicted* forces. Every force RMSE reported by `evaluate`, `sweep` or
> `auto-build` before the fix, and every choice made on it, is invalid.
> `tests/unit/test_evaluate.py` now feeds a model's own predictions back as
> references and requires zero error.

## Small cells are evaluated as exact supercells

`chimes_calculator`'s serial interface is wrong for cells thinner than
twice the model's outer cutoff, which covers most open-database frames:

- `small=False` misses periodic images: energies were off by ~20 kcal/mol
  on MatPES Cu-Zr cells.
- `small=True` gets energies right but sums forces over replicas.

`evaluate` therefore replicates each frame until every perpendicular width
exceeds 2 × the largest `S_MAXIM` in `params.txt`, evaluates that, divides
the energy by the number of copies, and keeps the first copy's forces.
This is exact for a periodic frame, and it reproduced `chimes_lsq`'s own
training-set predictions (`force.txt`) to 5×10⁻⁶ kcal/mol/Å and
5×10⁻⁴ kcal/mol on 122 real triclinic frames. `al-select` uses the same
code path.

## Prerequisite

Needs `chimescalc_lib` built: `chimes-agent setup --component
chimes_calculator --machine <name>`.

The calculator's banner and progress output go to `<output-dir>/evaluate.log`
(or a temp log), not stdout. Stdout carries only the JSON result, like
every stage.
