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

By default the fingerprint does not distinguish atom types, like the shipped
tool (whose composition term is inactive): for alloys it compares structure,
not chemical order.

## Element awareness (`--structure-weight`)

The paper's hybrid metric adds composition through one weight α ∈ [0, 1]:

    D = α · D_struct / (2√m) + (1 − α) · D_comp / (t_max − t_min)

`D_struct` is the edge-space distance above (m edges), `D_comp` compares the
clusters' element descriptors `t` (the mass column of `params.txt`, or
atomic numbers with `--descriptor number`): for pairs, the mean absolute
difference of the sorted descriptors; for triplets and quadruplets, the
Euclidean distance (over √n) between descriptors ordered by each atom's
centrality (its edge-sum over the sorted edges, most central first), so a
B atom at the apex of an A–A–B triangle differs from one at a corner. Both
terms lie in [0, 1], so D is histogrammed on [0, 1]. α = 1 is structure
only, α = 0 composition only; the descriptor keeps its size.

`--structure-weights 0,0.5,1` reports D², D²/critical and the novel fraction
at each α (plus the type-agnostic baseline) in `by_structure_weight`, and a
note says whether composition or structure separates the two sets more.
The comparison uses D²/critical because composition-only histograms take
few distinct values, so their covariance rank (the χ² degrees of freedom)
is much lower. The paper's reading:

- **Metallic alloys** (Y–Mg HEA there): composition resolves configurations
  that structure treats as degenerate; a small structural weight (α ≈ 0.25)
  selected training data most efficiently, and composition-only selection
  was the least reliable.
- **Molecular systems** (C–N): geometry already encodes the chemistry, and
  α hardly matters, except that composition-only sampling kept rare
  minority-element environments at dilute composition.

`al-batch --structure-weight 0.25` scores novelty with the hybrid metric.

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

The same harvest with `--structure-weight 0.25 --structure-weights 0,0.5,0.75,1`
(one pdebug node, 20 s):

| α | D² | critical | D²/critical |
|---|---|---|---|
| 0 (composition) | 522 | 9.2 | 56 |
| 0.25 | 42 651 | 168 | 253 |
| 0.5 | 43 404 | 191 | 227 |
| 0.75 | 17 406 | 191 | 91 |
| 1 (structure) | 6 551 | 192 | 34 |
| type-agnostic (default) | 344 | 155 | 2.2 |

Composition alone separates MD from training more than structure alone
(56 vs 34), and the mixed metric far more than either: the B2 supercell's
chemical order differs from the MatPES pool's, as the paper's alloy regime
predicts, so `al-batch --structure-weight 0.25` is the right novelty signal
for this system.

## Output

`fingerprint.json` (the above) and `fingerprints.npz` (per-frame
fingerprints for both sets, D_j²). `--machine` runs everything as one Slurm
job on a full node.
