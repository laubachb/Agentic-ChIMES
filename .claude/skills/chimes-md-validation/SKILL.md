---
name: chimes-md-validation
description: Validate ChIMES models in molecular dynamics - stability at the target temperatures, close contacts against the inner cutoffs, RDFs against a DFT reference, choosing between statistically tied candidates, fingerprint coverage of MD configurations versus the training set, and harvesting frames for active learning. Use when a fit or search finishes, when the user asks "is this model stable / usable in MD", wants to pick between candidate models, or when a study reaches its MD phase. The chimes-md-validator subagent follows this playbook.
---

# MD validation (study phase 6)

Holdout error is necessary but not sufficient. For water, holdout
cross-validation preferred bases that were over-structured or unstable in MD,
and the published model was chosen by MD against DFT (Lindsey 2019; see
`chimes-literature`, "Validation"). Work in `<study>/04_md/`.

## 1. Plan

- **Candidates**: the chosen model (`02_fit/search/best/params.txt`) plus
  the tied runners-up from `hyper_report.json` stage tables. Each table row
  has a `key`; its `params.txt` is `02_fit/search/points/<key>/params.txt`.
  Two or three candidates are enough.
- **Conditions**: the temperatures (and phases) the user cares about, plus
  one above the highest, since instabilities appear hot first.
- **Structure**: a representative orthorhombic or triclinic cell, via
  `--prototype` (ase bulk arguments) or a training frame. It is replicated
  past 2× the cutoff and `--min-atoms`.
- **Reference**: DFT-MD frames at one of the conditions, if any exist
  (`--reference-xyzf`). Without them, RDFs are saved but not ranked.
- **Size**: steps × atoms × candidates × temperatures × ~3e-4 core-s. A few
  hundred atoms × a few thousand steps is minutes per run. Anything beyond a
  single small run goes to Slurm: `md-check --machine`, dry run first.

## 2. Run

```bash
chimes-agent md-check --params A/params.txt --params B/params.txt \
  --prototype '{"name":"CuZr","crystalstructure":"cesiumchloride","a":3.26}' \
  --elements Cu,Zr --masses '{"Cu":63.546,"Zr":91.224}' \
  --temperatures 300,1200,1600 --nsteps 5000 \
  --machine dane --queue debug --walltime-hours 0.5 --output-dir 04_md/check --dry-run
```

## 3. Judge (`run/md_check.json`)

- `stable`: no runaway (lost atoms, crash, energy jumps, overheating).
  Unstable at a needed temperature disqualifies a candidate.
- `equilibrated: false` with `stable: true` means the run was too short to
  thermalize (a crystal starts near T/2). Lengthen it before comparing RDFs.
- `below_inner_cutoff_frames` > 0: the model visits distances it never saw.
  The data lacks short-range coverage. Active learning fixes this, not
  another basis. The Cu-Zr example did this at 1200 K.
- `close_contact_fraction` near 1 at high T means the same thing, less
  severely.
- `mean_rdf_distance` (with a reference): among stable candidates, prefer the
  lower one. Differences below ~0.05 are noise at these run lengths.

Then compare the MD configurations with the training data:

```bash
chimes-agent fingerprint --params <model>/params.txt --reference-xyzf 01_data/curate/train.xyzf \
  --candidates-xyzf 04_md/check/run/harvest.xyzf --output-dir 04_md/fingerprint [--machine dane]
```

- `sets.distinguishable: true`: MD explores configurations the training set
  does not cover. Expected before active learning.
- `novelty.fraction_novel` and `novel.xyzf`: the frames to label first.
- Once DFT-MD and ChIMES-MD at a state point are indistinguishable,
  active learning has converged there (Laubach 2026).

The fingerprint does not distinguish atom types (paper and shipped tool);
for alloys it compares structure, not chemical order.

## 4. Report

Write `04_md/MD_REPORT.md`: candidates, conditions, a stability table,
close contacts, RDF ranking (or why none), the fingerprint verdict, and the
recommendation. The recommendation is one of: use model X; choose X over Y
because ...; or needs active learning, with `harvest.xyzf` / `novel.xyzf`
as the next batch. Register runs: `study --register md_runs=04_md/check`.

Reference: `docs/commands/md-check.md`, `docs/commands/fingerprint.md`.
