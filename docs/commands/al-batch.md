# `chimes-agent al-batch`

**Status: implemented.**

Assembles one active-learning batch to label, from candidate frames
(usually the `md-check` harvest), within a budget. It combines the signals
the literature and this toolkit provide:

| signal | source | why |
|---|---|---|
| close contacts | the model's inner cutoffs (`params.txt`) | frames at or inside an inner cutoff are what makes MD unstable (Lindsey 2025); up to `--min-close` are taken first, closest first |
| QUESTS dH | [quests](quests.md) | environments not covered by the training set |
| fingerprint D_j² | [fingerprint](fingerprint.md) | configurations outside the training distribution, as the model's clusters see them |
| committee spread | [committee](committee.md), when `--fm-setup-in` is given | where the fitted model is undetermined |

Each candidate gets the mean of its normalized ranks over the available
signals; close contacts never rank below the median. Exact duplicates of
training frames are dropped, near-duplicate candidates (`dataset-select`'s
correlated groups) collapse to their best member, and the top `--budget`
frames become `batch.xyzf`. `batch.json` lists every score, so the choice
can be inspected or re-cut by hand.

```bash
chimes-agent al-batch --candidates-xyzf 04_md/check/run/harvest.xyzf \
  --train-xyzf 01_data/curate/train.xyzf --params 02_fit/search/best/params.txt \
  --fm-setup-in 02_fit/search/best/fm_setup.in --budget 40 --min-close 10 --output-dir 03_al/round1/batch
chimes-agent qe-relabel --structure-xyzf 03_al/round1/batch/batch.xyzf ... --output-dir 03_al/round1/qe   # same QE settings as the base set
```

Missing tools are skipped with a note (`signals_used` says what counted).
`--structure-weight 0.25` scores fingerprint novelty with the element-aware
metric ([fingerprint](fingerprint.md)), the paper's choice for alloys. On
the Cu-Zr harvest (46 candidates, budget 15) the two metrics rank
candidates almost independently (Spearman 0.14 between their D_j²), yet 12
of the 15 selected frames agree, because close contacts are taken first
and the near-duplicate collapse leaves only 16 distinct candidates. The
difference shows when the budget is smaller than the number of distinct,
non-close-contact candidates. One node, 40 s with the full 126-frame
reference (`--max-fingerprint-frames 126`).
