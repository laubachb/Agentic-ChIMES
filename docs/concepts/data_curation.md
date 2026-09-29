# The data phase

A ChIMES study runs in five phases: plan, **data**, hyperparameter search,
model build, active learning (`.claude/skills/chimes-study/SKILL.md`). This
page covers the data phase: how a request like *"I need a potential for
Cu-Zr metallic glasses"* becomes a base training set, and why the tools
behave as they do.

## Who does what

- **`chimes-data-curator`** (Claude Code subagent, `.claude/agents/`) is the
  data agent. Given the request and a study directory it writes
  `01_data/DATA_PLAN.md`, searches, fetches, generates, curates, and returns
  a fixed-format report. It may download and process data, but it never
  submits HPC jobs: QE labeling comes back as a dry-run command for you to
  approve. Label choices it can't make come back as questions.
- **`chimes-data-curation`** (skill) is its playbook. It's a skill, not a
  prompt, so the main conversation can follow the same steps interactively.
- **Stages** do the deterministic work: `data-search`, `data-fetch`,
  `data-generate`, `data-curate`, plus `qe-relabel` and `dataset-select`.

```
request ─► DATA_PLAN.md ─► data-search ─► data-fetch --count-only
                                   │
             ┌─────────────────────┼──────────────────────┐
     A: one dataset,      B: open-data structures    C: data-generate
        its labels           (--label-policy relabel)     structures
             │                     └──────────┬───────────┘
             │                     data-curate (dedupe, FPS subsample)
             │                                │
             │                qe-relabel (dry run → user approves → submit → collect)
             └──────────────► data-curate ◄───┘
                                   │
                        data_manifest.json ─► model building
```

## The decisions that matter

**One level of theory.** A ChIMES fit regresses absolute energies and forces.
Datasets computed with different codes, pseudopotentials, cutoffs or
smearing have different energy references, even under the same functional
label, so their labels cannot share a fit. `data-curate` enforces this from
provenance. The structures themselves can be mixed freely, so strategy B uses
any number of datasets and relabels them once.

**Label target = the active-learning engine.** Active learning adds frames
labeled by some QM engine, and those must match the base set. So "use
MatPES labels" is only right if the later labeling reproduces MatPES's VASP
settings. The built-in labeler is QE (`qe-relabel`), whose provenance records
a settings hash. The agent surfaces this choice when the request doesn't
settle it.

**Coverage over count.** What destabilises a ChIMES model in MD is usually
an element pair with no short-range data (unconstrained repulsive wall) or a
region of configuration space that is never sampled. `data-curate` reports
per-pair minimum distances and short-range counts for exactly this reason.

**Cell size.** Open databases favour small primitive cells: the MatPES Cu/Zr
frames are 1–34 atoms, some only 2 Å thick. chimes_lsq handles them with
periodic layers, and the manifest's `nlayers_required` gives the `N_LAYERS`
needed. `data-generate` builds ≥ 8 Å orthorhombic supercells instead, which
also suit `auto-build` and `lammps-run`.

## A real example: Cu-Zr

- `data-search --elements Cu,Zr` finds 71 candidate datasets. System-specific
  data is pure Cu only (three sets, one of them FHI-aims). The alloy set
  `UNEP_v1_2023_train` is corrupt at the source. The rest are general
  databases.
- `data-fetch --count-only` on MatPES-PBE-2025.2: 169 Cu/Zr-only frames
  (5 s, two columns read).
- `data-curate` removes 9 vacuum clusters/isolated atoms, 1 duplicate and 1
  energy outlier, leaving 158 frames, 156 of them triclinic (several are
  1-atom crystals), with 2 Å thinnest width (`N_LAYERS 4` for an 8 Å
  cutoff).
- The hyperparameter phase then fits a 2-body order 14 @ 7 Å + 3-body order
  4 @ 6.3 Å model at holdout relative force error 0.32 ± 0.07 (see
  `docs/commands/hyper-search.md`), a level that points at the data as the
  limit.
- A reasonable plan is therefore C (or B+C): generated Cu-Zr supercells plus
  MatPES structures, relabeled with one QE settings set, rather than 158
  small MatPES frames on their own.

## Limits

- ColabFit is the only remote source for now. Materials Project's API needs
  a key, and MPtrj exists only as one 12 GB JSON, so neither is wired in.
  Local files cover anything ASE reads.
- Stress is not carried from open data.
- `data-generate` samples near crystals only. Liquids, defects and finite
  temperature come from MD (a later phase) or from open AIMD data (OMat24's
  `aimd` subsets).
