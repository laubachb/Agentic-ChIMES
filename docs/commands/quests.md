# `chimes-agent quests`

**Status: implemented** (needs `pip install -e ".[quests]"`).

Information-theoretic coverage, novelty and selection with
[QUESTS](https://github.com/dskoda/quests) (Quick Uncertainty and Entropy
via STructural Similarity, Schwalbe-Koda et al. 2024). Every atomic
environment is described by its sorted neighbor distances and the distances
among those neighbors; a dataset is a kernel density in that space.

| quantity | meaning |
|---|---|
| entropy H | how much structural information the set holds (nats); H(n) over nested subsets shows whether it has saturated |
| diversity | an estimate of the number of distinct environments |
| dH(y \| X) | differential entropy of environment y against set X: ≤ 0 means covered at least as well as X's own environments; large means novel |

```bash
# coverage of the training set
chimes-agent quests --reference-xyzf 01_data/curate/train.xyzf --output-dir 01_data/quests
# novelty of an MD harvest, plus 20 frames chosen by greedy entropy gain
chimes-agent quests --reference-xyzf 01_data/curate/train.xyzf --candidates-xyzf 04_md/check/run/harvest.xyzf \
  --select 20 --output-dir 03_al/round1/quests
```

Outputs: `entropy`, `diversity`, `self_dH_quantiles`, `entropy_curve` and
`entropy_saturated`; with candidates, per-frame `dH_max`/`dH_mean`,
`fraction_novel_frames`, `entropy_gain_if_all_added`, `novel.xyzf`
(frames above the threshold) and `selected.xyzf` (greedy entropy-maximizing
choice). The default novelty threshold is the reference's own 99th
percentile of self-dH, which is ≈ 0: a QUESTS dH above 0 means the
environment is not covered. `quests_dH.png` overlays the dH distributions.

Descriptors: `--descriptor single` (species-agnostic, the published form)
or `multi` (per-species blocks; marked experimental upstream). QUESTS'
defaults k = 32 neighbors, 5 Å cutoff and bandwidth 0.015 are used.

## Where it fits

- `data-curate --selection quests` and `dataset-select --method quests`
  choose a subset by greedy entropy gain (closest contacts first), instead
  of farthest-point sampling on composition.
- `al-batch` uses dH as one of its novelty signals.
- `al-status` reads a round's `quests.json` for the stopping rule.
- The ChIMES cluster-graph [fingerprint](fingerprint.md) is the other
  coverage measure: whole-configuration, through the model's own clusters.
  QUESTS is per environment and model-free; the two agree on the Cu-Zr
  example (MD harvest: fingerprint D² ≫ critical; QUESTS 96 % of frames
  novel).

On the Cu-Zr training set (126 frames, 407 environments): H = 4.94 nats,
not yet saturated (+0.31 nats from the last quarter of the frames); the
1200 K MD harvest adds +0.32 nats, with 44 of 46 frames novel.
