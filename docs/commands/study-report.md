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

## Sections filled from registered validation artifacts

| registered key | section |
|---|---|
| `learning_curve` | 3. Final model → data sufficiency table and verdict |
| `al_status` | 4. Active learning → per-round table and verdict |
| `quests`, `fingerprint`, `committee` | 4. → coverage and uncertainty |
| `md_check` | 5. MD validation → per-run stability, equilibration, close contacts, RDF distance, penalty |
| `eos_check` | 5. → equation of state, elastic constants, Born stability, pressure vs DFT |
| `evaluate` | Figures (parity plot) |
| `deploy` | summary → nonzero coefficients, penalty |

The hyper-search section adds cross-validation (when used) and the
one-dimensional sensitivity profiles. A **Figures** section embeds every
PNG found under the registered directories (relative paths, so the report
renders where it is written). `search/HYPER_REPORT.md`, `04_md/MD_REPORT.md`
and `03_al/AL_STATUS.md` are read as source texts for the narrative.
