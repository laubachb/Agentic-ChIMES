# `chimes-agent benchmark`

**Status: implemented.** Strong and weak scaling of a ChIMES model in
LAMMPS, and the cost model for sizing production runs.

```bash
# preview, then submit (one Slurm job runs every case, smallest first)
chimes-agent benchmark --params study/02_fit/search/best/params.txt \
  --prototype '{"name":"CuZr","crystalstructure":"cesiumchloride","a":3.26}' \
  --elements Cu,Zr --masses '{"Cu":63.546,"Zr":91.224}' --strong-atoms 8000 \
  --machine dane --queue debug --walltime-hours 1 --dry-run --output-dir study/05_bench
# ... after the job finishes
chimes-agent benchmark --collect study/05_bench
```

## Cases

- **strong**: one system of ~`--strong-atoms` atoms on each rank count.
- **weak**: ~`--atoms-per-rank` (250) × ranks atoms on each rank count.
- Ranks default to 1, 2, 4, 8, 16, 28, 56, 112 on one node; `--nodes 2,4`
  adds multi-node points.
- Each case: NVT at `--temperature`, `--warmup-steps` untimed, then
  `--steps` timed (LAMMPS `Loop time`), with a `--case-timeout-s` cap.

The structure is an orthorhombic supercell of `--prototype` (ase bulk
kwargs) or `--structure-xyzf`. Choose something representative of
production and check it is stable with the model first. Cost per atom-step
depends on the model's cutoffs and cluster counts, so always benchmark the
final model.

## Results (`benchmark.json`)

| field | meaning |
|---|---|
| `core_s_per_atom_step` | ranks × loop time ÷ (steps × atoms): the cost unit |
| strong `speedup`, `efficiency` | vs the fewest ranks: (T_ref·P_ref)/(T·P) |
| weak `efficiency` | cost per atom-step at the fewest ranks ÷ at this rank count |
| `ns_per_day` | at that case's size and rank count |
| `recommended_ranks` | most ranks with efficiency ≥ `--min-efficiency` (0.7) |
| `cost_model` | cost unit used for estimates (weak scaling at its recommended ranks) |
| `estimates_1ns` | CPU-hours, ranks, nodes and wall time for 10³-10⁶ atoms × 1 ns at the benchmark timestep |
| `unstable` | cases whose MD produced NaNs or lost atoms: a model problem |

**Sizing a request:** CPU-hours ≈ `core_s_per_atom_step` × atoms × steps ÷ 3600,
at about `atoms_per_rank` atoms per rank; add ~20 % margin.
