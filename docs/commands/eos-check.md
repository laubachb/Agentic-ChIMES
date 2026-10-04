# `chimes-agent eos-check`

**Status: implemented.** Local, seconds.

Equation of state and elastic constants of a structure under a model. These
are physical checks that go beyond MD stability, as in published ChIMES
validation.

```bash
chimes-agent eos-check --params 06_deploy/params.txt \
  --prototype '{"name":"CuZr","crystalstructure":"cesiumchloride","a":3.26}' \
  [--reference-xyzf 01_data/curate/holdout.xyzf] --output-dir 04_md/eos
```

| output | meaning |
|---|---|
| `eos.V0_A3_per_atom`, `E0`, `B0_GPa`, `B0_prime` | third-order Birch-Murnaghan fit of E(V) over ±`volume_range` (8 %) |
| `eos.stress_vs_dEdV_max_abs_GPa` | pressure from the model's stress vs −dE/dV: an internal consistency check (should be ≪ 1 GPa) |
| `elastic.C_GPa` | 6×6 elastic tensor at V0 from ±`strain` (0.5 %) in each Voigt direction, via the model's stress (Cauchy sign) |
| `elastic.cubic` | C11, C12, C44 when the tensor is cubic |
| `elastic.born_stable` | tensor positive definite: mechanically stable under the model |
| `reference_pressure` | with `--reference-xyzf`: model vs DFT pressure on frames of the same composition |

Elastic constants are **clamped-ion** (no internal relaxation). They are
exact when every atom sits on an inversion centre (fcc, bcc, B2) and
approximate otherwise.

On the Cu-Zr model for B2 CuZr:

- V0 = 16.92 Å³/atom;
- B0 = 79.5 GPa from E(V), 79.6 GPa from the elastic tensor;
- stress vs −dE/dV agree to 0.03 GPa;
- C11/C12/C44 = 98.7/70.0/62.7 GPa, Born-stable.

`eos.png` shows the E(V) points with the Birch-Murnaghan fit.

Compare V0 and B0 with DFT for the same structure before trusting the
model's mechanics.
