# `chimes-agent al-status`

**Status: implemented.** Read-only apart from `AL_STATUS.json/.md` and a plot.

Where an active-learning campaign stands, across rounds, with a verdict.
It expects the playbook layout under `<study>/03_al/`:

```
round<k>/batch/batch.json          al-batch
round<k>/qe/                       qe-relabel
round<k>/merge/data_manifest.json  al-merge
round<k>/fit/params.txt            refit at the chosen hyperparameters
round<k>/md/run/md_check.json      md-check of the refit
round<k>/quests/quests.json        quests on the round's harvest (optional)
round<k>/fingerprint/...           fingerprint on the harvest (optional)
```

Every round's `fit/params.txt` (and the base model) is scored on the
study's **fixed holdout**, so errors are comparable. The verdict:

| verdict | when |
|---|---|
| `CONVERGED` | the latest round is MD-stable at every checked temperature, samples nothing inside the inner cutoffs, and its harvest is no longer novel (QUESTS novel fraction < 0.1, or fingerprint indistinguishable); or the holdout error has been flat (< 0.02) for three rounds with stable MD |
| `CONTINUE` | otherwise, with the reasons |
| `NO_ROUNDS` | nothing to assess |

```bash
chimes-agent al-status --study /p/lustre2/$USER/studies/cuzr
```

`AL_STATUS.md` holds the per-round table (training frames, added, holdout
force and energy error, MD stability, frames inside the inner cutoff, novel
fraction) and `al_progress.png` the error per round. `study-report`
includes both.
