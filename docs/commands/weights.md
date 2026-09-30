# `chimes-agent weights`

**Status: implemented.**

Writes per-row fitting weights (`weights.dat`) for an A-matrix built by
`amat-build`, for `solve --weights`. It ships the published ChIMES weighting
schemes as presets. `model-build --weights-preset` and
`hyper-search --weights-preset` call it for you.

## Usage

```bash
chimes-agent amat-build --fm-setup-in run/fm_setup.in --output-dir run
chimes-agent weights --work-dir run --preset hierarchical2026
chimes-agent solve --A run/A.txt --b run/b.txt ... --weights run/weights.dat

# or in one step
chimes-agent model-build --fm-setup-in run/fm_setup.in --weights-preset hierarchical2026 --output-dir run
```

## Presets (forces / energies / stresses)

| preset | F | E | S | source |
|---|---|---|---|---|
| `uniform` | 1 | 1 | 1 | what every fit used before this stage |
| `al_driver` | 1 | 0.1 | 250 | al_driver's defaults |
| `lindsey2020` | 1 | 5 | 1 | Lindsey et al., JCP 153, 134117 (2020) |
| `carbon2_large` | 1 | 0.1 | 100 | Lindsey et al., npj Comput. Mater. 11, 26 (2025) |
| `hierarchical2026` | 1 | 0.3 | 100 | Lindsey et al., npj Comput. Mater. 12, 18 (2026) |

Weights multiply rows, so their meaning is relative. A frame has 3N force
rows and few energy or stress rows; chimes_lsq writes 3 energy rows per
frame. See [literature](../concepts/literature.md) for why energy and stress
weights matter (force-only fits do not constrain pressure or density).

## Custom schemes: al_driver's methods

Override any row type with `--force-method`, `--energy-method`,
`--stress-method`, `--force-gas-method` or `--energy-gas-method`. Each takes
al_driver's `config.py` form, so a scheme means the same thing in both
tools:

| method | weight | params |
|---|---|---|
| `A` | a0 | 1 |
| `B` | a0 · I^a1 (I = `--cycle`, 0 treated as 1) | 2 |
| `C` | a0 · exp(a1·\|X\|/a2) | 3 |
| `D` | a0 · exp(a1·(X − a2)/a3) | 4 |
| `E` | n_atoms^a0 | 1 |
| `F` | a0 · exp(a1·(X/n_atoms − a2)/a3) | 4 |
| `G` | a0 · exp(a1·(\|X\| − a2)/a3) | 4 |

X is the row's reference value. Example: a Boltzmann-style force weight
that de-emphasizes large forces, `--force-method '["C", [1.0, -1.0, 50.0]]'`.

Rows are classified from `b-labeled.txt` exactly as al_driver's `gen_ff.py`
does:

- `+1` = energy;
- a tag containing `s_` = stress;
- anything else = force;
- a `G_` prefix marks gas-phase clusters.

## Active-learning decay

`--frame-cycles cycles.json` (the cycle each training frame was added in,
one entry per frame in `fm_setup.in`'s trajectory order) plus
`--decay-cycles n` multiplies every row of a cycle-I frame by n / max(I, 1).
Early, possibly unphysical frames then cannot pull the fit away from
later, ground-truth data (Lindsey 2025, 2026).

## Output

```json
{"weights": "run/weights.dat", "preset": "hierarchical2026", "rows": 1581,
 "scheme": {"force": ["A", [1.0]], "energy": ["A", [0.3]], "...": "..."},
 "by_row_type": {"force": {"rows": 1215, "min": 1.0, "max": 1.0},
                 "energy": {"rows": 366, "min": 0.3, "max": 0.3}},
 "decay": null}
```

Weights change what the *training* error means. Compare schemes on the
holdout with `evaluate`.
