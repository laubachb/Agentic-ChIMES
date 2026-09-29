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
- `rmse_energy_kcal_mol_per_atom` is the energy error divided by each
  frame's atom count before averaging. Use it when frame sizes differ.

With `n_models > 1`, `committee_spread.per_frame_energy_stdev` is the
spread across models of each frame's predicted energy: the plug-in point
for a future uncertainty-based active-learning selector. It is not used by
`al-select`; see `docs/concepts/qm_driver_plugins.md`.

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
