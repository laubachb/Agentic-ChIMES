# `chimes-agent committee`

**Status: implemented.**

A bootstrap committee of one ChIMES basis, and candidate frames ranked by
how much its members disagree. This is query-by-committee uncertainty for
active learning, which al_driver documents but never finished.

```bash
chimes-agent committee --fm-setup-in 02_fit/search/best/fm_setup.in --algorithm lassolars --alpha 1e-5 \
  --n-models 5 --candidates-xyzf 04_md/check/run/harvest.xyzf --n-select 20 --output-dir 03_al/round1/committee
```

- The basis's design matrix is built **once**. Each member re-solves it on
  a bootstrap resample of the training frames: a frame drawn k times gets
  weight √k on all its rows, which is what duplicating those rows does to a
  least-squares objective. Members use the same solver and α as the
  original fit.
- With `--candidates-xyzf`, every candidate is predicted by every member.
  **Force spread** is the RMS over atoms and components of the standard
  deviation across members. **Energy spread** is the standard deviation
  per atom. The `--n-select` frames with the largest force spread go to
  `uncertain.xyzf`, ready for `qe-relabel`.

On the Cu-Zr model (4 members), the median force spread was
0.41 kcal/mol/Å on in-distribution holdout frames and 1.12 on frames
harvested from 1200 K MD. The signal separates what the data covers from
what it does not. The holdout maximum (7.1) sits on the frame `evaluate`
lists as its worst.

Use it next to [fingerprint](fingerprint.md) novelty: fingerprints measure
structural distance from the training set, the committee measures where
the fitted model is undetermined.
