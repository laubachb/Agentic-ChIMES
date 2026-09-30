# `chimes-agent hierarch`

**Status: implemented.**

Hierarchical (element-block) fitting, following al_driver 2.0 and Lindsey
et al., *npj Comput. Mater.* **12**, 18 (2026). Pure-element blocks
{X, XX, XXX, XXXX} are fitted once from single-element data and kept fixed.
Only the cross blocks {XY, XXY, XYY, ...} are fitted to mixture data. The
paper found this as accurate as fitting everything at once, and the element
models are reusable across systems.

## Workflow

```bash
# 1. element models (ordinary fits on single-element frames)
chimes-agent fm-setup-gen --json-in Cu.json --output-dir Cu && chimes-agent model-build --fm-setup-in Cu/fm_setup.in --output-dir Cu
chimes-agent fm-setup-gen --json-in Zr.json --output-dir Zr && chimes-agent model-build --fm-setup-in Zr/fm_setup.in --output-dir Zr

# 2. subtract them from the mixture data
chimes-agent hierarch --element-params Cu/params.txt --element-params Zr/params.txt \
  --subtract mixed.xyzf --output-dir sub

# 3. fit the cross terms on the residual
#    fm-setup-gen: hierarc true, exclude_1b [[Cu],[Zr]], exclude_2b [[Cu,Cu],[Zr,Zr]],
#    exclude_3b [[Cu,Cu,Cu],[Zr,Zr,Zr]] (and pure 4-body types), the element pairs' parameters
#    equal to the element models'
chimes-agent fm-setup-gen --json-in cross.json --output-dir cross && chimes-agent model-build --fm-setup-in cross/fm_setup.in --output-dir cross

# 4. merge into one model
chimes-agent hierarch --element-params Cu/params.txt --element-params Zr/params.txt \
  --combine cross/params.txt --output-dir combined     # -> combined/hierarch.params.txt
```

- **`--subtract`**: for each element model, keeps only that element's atoms
  in every frame, predicts their energy, forces and stress (exact
  supercells, like `evaluate`) and subtracts them from the labels. Each
  element model's pair penalty is set to zero first, as al_driver does;
  otherwise close contacts would subtract unphysical energy.
- **`--combine`**: merges through al_driver's own `param_file` machinery,
  with one guard: al_driver marks the special 4-body cutoff block as
  specific even when no model has 4-body terms, and then fails to write it.

## Checked on Cu-Zr (126 MatPES frames: 34 Cu, 31 Zr, 61 mixed)

| | holdout relative force error | reduced RMSE | E/atom (kcal/mol) |
|---|---|---|---|
| all at once (6/4, 3-body at 7 Å) | 0.306 | 0.453 | 0.79 |
| hierarchical (same orders/cutoffs) | 0.323 | 0.478 | 0.94 |

The two are tied within the standard error (~0.06). The combined model equals
the sum of its parts to 5×10⁻¹⁰ kcal/mol, checked on a mixed frame: Cu model
on the Cu atoms plus Zr model on the Zr atoms plus the cross model.

Element models are applied to distances they may not have sampled: in mixed
frames, like-element distances can be shorter than in the pure data. On this
sparse set the Zr model removed 9.3 kcal/mol/Å RMS of force from the mixed
frames. Give the element models coverage of the mixture's like-element
distances.
