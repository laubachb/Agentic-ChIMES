# `chimes-agent md-check`

**Status: implemented.**

Short NVT MD of one or more candidate models, which is the check the ChIMES
literature applies before trusting a model chosen on holdout error. For
water, holdout cross-validation kept preferring larger bases that were
over-structured or unstable in MD; the final model was chosen by comparing
MD with DFT (Lindsey et al., JCTC 15, 436, 2019). The stage also harvests
frames for active learning.

## Usage

```bash
# locally (small runs), several candidates, two temperatures
chimes-agent md-check \
  --params 02_fit/search/best/params.txt --params 02_fit/search/points/<runner-up>/params.txt \
  --prototype '{"name":"CuZr","crystalstructure":"cesiumchloride","a":3.26}' \
  --temperatures 300,1200 --nsteps 5000 --reference-xyzf dft_md_1200K.xyzf \
  --output-dir 04_md/check

# as one Slurm job (all runs in parallel on one node); --dry-run first
chimes-agent md-check ... --machine dane --queue debug --walltime-hours 0.5 --output-dir 04_md/check
```

The structure comes from `--structure-xyzf` (+ `--frame-index`) or
`--prototype` (ase `bulk` arguments). Triclinic cells are fine. It is
replicated to at least 2× the largest outer cutoff and `--min-atoms` (200).

## What each run reports

| field | meaning |
|---|---|
| `stable` | no runaway: LAMMPS finished, no lost atoms, finite energies, no potential-energy jump above `--max-pe-jump-per-atom` (1 kcal/mol/atom) between dumps in the second half, and the second-half mean temperature not above target × (1 + `--temperature-tolerance`) |
| `instability` | the reasons, when not stable |
| `equilibrated` | second-half mean temperature within the tolerance. A crystal started at T first settles near T/2, so short runs can be stable but not yet equilibrated. |
| `closest_distance` | closest approach per element pair over the run |
| `close_contact_fraction` | dumped frames with some pair within `--close-contact-margin` (0.1 Å) of that pair's inner cutoff |
| `below_inner_cutoff_frames` | frames inside an inner cutoff: the model is extrapolating (penalty region) |
| `rdf_file`, `rdf_distance` | partial RDFs over the second half; with `--reference-xyzf`, the mean \|Δg(r)\| per pair |

Per model, `models` summarizes:

- `stable_at_all_temperatures`
- `unstable_temperatures`
- `not_equilibrated_temperatures`
- `max_close_contact_fraction`
- `mean_rdf_distance`

## Penalty

Each candidate runs with an explicit repulsive penalty below its inner
cutoff: the model's own lines if it has them, else what `deploy` writes
(0.02 Å, 1e5 kcal/mol/Å³), or `--penalty-dist` / `--penalty-scaling`.
So MD is validated in the form the model will be used. On Cu-Zr at
1200 K, chimesFF's implicit default (0.01 Å, 1e4) let 20 of 81 frames
sample distances inside the inner cutoff; the explicit default, 2.

## Choosing between candidates

1. Drop models that are unstable at any temperature that matters.
2. Among stable ones, prefer the lower `mean_rdf_distance` against a DFT
   reference at the same conditions (ideally DFT-MD frames). Without a
   reference, RDFs are saved but not ranked.
3. Many `below_inner_cutoff_frames` means the data lacks short-range
   coverage. The fix is active learning on the harvested frames, not
   choosing a different basis.

## Harvest for active learning

Every run contributes up to `--harvest-close` (20) close-contact frames and
up to `--harvest-other` (20) others to `harvest.xyzf`. This follows the
parallel active-learning recipe of Lindsey et al. (2025, 2026). Label it
with `qe-relabel` (same QE settings as the base set), curate it together
with the base data, and refit.

## Plots

Per run: `temperature.png`, `energy.png` (potential energy per atom) and
`rdf.png` (partial RDFs of the second half). `--no-plot` skips them.

## Output

`<output-dir>/md_check.json` holds everything above plus `notes`. Each run
has its own `model<i>_T<T>/` directory with `in.lammps`, `log.lammps`,
`dump.out` and `rdf.npz`.
