# `chimes-agent study-report`

**Status: implemented.** Collates a study into `REPORT_FACTS.json` and a
`REPORT.md` skeleton.

```bash
chimes-agent study-report --study /p/lustre2/$USER/studies/cuzr
```

Reads the registry and standard layout: data manifest and curation report,
hyperparameter report (stage decisions, 4-body results, cluster-type
exclusion analysis), final model and accuracy, active-learning directory,
MD runs (stability from each LAMMPS log: mean temperature, energy drift per
atom, NaNs or lost atoms), benchmark, CPU-hours, and the deployed model
card.

`REPORT.md` has every table filled in, plus `<!-- NARRATIVE: ... -->`
placeholders. The `chimes-report-writer` agent replaces those with prose
drawn from the facts. `REPORT_FACTS.json` is the ground truth the prose must
agree with. Missing phases are listed as gaps.
