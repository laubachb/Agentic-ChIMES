# `chimes-agent fingerprint`

**Status: implemented** (native; checked against chimes_calculator's shipped tool).

Cluster-graph fingerprints of configurations, and Mahalanobis comparisons of
datasets (Laubach, Lordi, Lindsey, *J. Chem. Inf. Model.* **66**, 182, 2026).
It answers:

- **Coverage.** Does a candidate set (MD frames, a new database pull) sample
  configurations the training set does not?
- **Novelty.** Which candidate frames lie outside the training distribution?
  Label those first.
- **Stopping active learning.** Once ChIMES-MD and DFT-MD frames at a state
  point are statistically indistinguishable, active learning has converged
  there.

```bash
chimes-agent fingerprint --params model/params.txt \
  --reference-xyzf 01_data/curate/train.xyzf --candidates-xyzf 04_md/check/run/harvest.xyzf \
  --output-dir 04_md/fingerprint [--machine dane --dry-run]
```

## How it is computed

For each frame:

1. Every unique periodic 2-, 3- and 4-body cluster whose edges all lie inside
   the model's outer cutoffs is found (per pair and cluster type, as chimesFF
   reads them from `params.txt`).
2. Edge lengths are Morse-transformed with each pair's cutoffs and λ, then
   sorted.
3. Pairwise Euclidean dissimilarities within each body order are
   histogrammed: 100 bins on [0, d_max], with d_max = 3, √12+1 and √24+1.
4. The normalized histograms are concatenated.

Body orders default to the model's (2, plus 3/4 if it has them). Above
`--max-clusters` (5000) clusters per order and frame, a random subset
estimates the histogram.

This reproduces the shipped `chimesFF/src/FP` tool. On its example (96-atom
water), all three histograms match the expected output to the printed
precision, and replicated cells give exactly the replicated cluster counts.
The shipped tool needs a separate FINGERPRINT build of LAMMPS and writes
cluster files per frame and rank, which lustre's file-count quota punishes.

The fingerprint does not distinguish atom types, like the paper and the
shipped tool: the tool's composition term is currently always zero. For
alloys it compares structure, not chemical order.

## Statistics

- `sets`: D² between the two sets' mean fingerprints (pooled covariance,
  pseudo-inverse), compared with the χ² critical value at `--alpha` (0.1)
  and degrees of freedom = covariance rank. Following the paper, this
  measures how far apart the distributions sit (an effect size), not whether
  the means differ significantly.
- `novelty`: D_j² of each candidate frame against the reference mean and
  covariance. `fraction_novel`, its quantiles, and `novel.xyzf` with the
  novel frames.

## Example (Cu-Zr)

The `md-check` harvest of the Cu-Zr model (B2 supercell MD at 300 and
1200 K) against its MatPES training set: D² = 598 vs a critical 155, and
all 46 frames novel. MD explores configurations the training data lacks,
which is the expected finding before active learning.

## Output

`fingerprint.json` (the above) and `fingerprints.npz` (per-frame
fingerprints for both sets, D_j²). `--machine` runs everything as one Slurm
job on a full node.
