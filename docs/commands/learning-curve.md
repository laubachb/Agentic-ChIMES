# `chimes-agent learning-curve`

**Status: implemented.** Local, one design-matrix build plus a few solves.

Holdout error of one basis fitted on nested subsets of its training data.
It answers the question every active-learning decision depends on: is the
model **data-limited** (error still falling) or at a **plateau** (more of
the same data will not help; change the basis, the coverage or the labels)?

```bash
chimes-agent learning-curve --fm-setup-in 02_fit/search/best/fm_setup.in \
  --holdout-xyzf 01_data/curate/holdout.xyzf --fractions 0.125,0.25,0.5,0.75,1 --repeats 2 --output-dir 02_fit/learning_curve
```

The design matrix is built once; each subset fit gives the frames left out
a row weight of 0 (the same device `hyper-search --cv-folds` uses), so a
five-point curve with two repeats is ten solves. Subsets are nested,
group-aware (correlated frames enter together) and always include each
pair's closest-contact frame, since the inner cutoff is set from it.

The tail (last three points) is fitted as err = a·n^(−b) and extrapolated
to 2× and 4× the data. Verdict: `data-limited` when the log-log slope is
≤ −0.1 (doubling the data would cut the error by more than ~7 %),
otherwise `plateau`.

On the deployed Cu-Zr basis: 0.469 → 0.376 → 0.312 → 0.313 → 0.313 at
16/32/63/94/126 frames: a plateau from 63 frames on. More MatPES frames of
the same kind would not improve it; the basis or the coverage is the
limit. The energy error kept falling (1.89 → 0.85 kcal/mol/atom), so energies
are still data-limited.
