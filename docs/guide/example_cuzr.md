# Worked example: Cu-Zr from MatPES

A real study, run on LLNL Dane while this toolkit was being built. Every
number below comes from the study's own artifacts (`REPORT_FACTS.json`).
It shows what each phase produces and what the final report reads like.

**Goal:** a Cu-Zr ChIMES potential from open DFT data, labels kept as
published (MatPES, PBE).

## Phase by phase

**Plan.** `chimes-agent study --init ... --elements Cu,Zr` created the study
directory and registry.

**Data** (`chimes-data-curator`).

- `data-search` found 71 candidate datasets among ~500. The only
  system-specific ones were pure Cu, and the one Cu-Zr alloy set is corrupt
  at the source (flagged).
- MatPES-PBE-2025.2 held 169 Cu/Zr-only frames (`data-fetch --count-only`,
  5 s).
- `data-curate` kept 158. It removed 9 isolated atoms and vacuum clusters, 1
  duplicate and 1 energy outlier, then split the rest 126 train / 32
  holdout.
- Cells are tiny (1-34 atoms, 2 Å thinnest width), so the fit needs 4
  periodic layers.

**Hyperparameters** (`chimes-hyperparameter-tuner`, `hyper-search` on
`pdebug`). 61 distinct fits:

| stage | decision |
|---|---|
| analysis | Morse λ from first RDF peaks: Cu-Cu 2.51, Cu-Zr 2.79, Zr-Zr 3.21 Å; inner cutoffs 0.02 Å below the closest contacts |
| 2-body | order 8 at 8 Å |
| 3-body | order 4 at 7.0 Å: tied with order 6 and half the MD cost. At the 4.1 Å first-shell cutoff, 3-body terms do almost nothing |
| 4-body | 4 light fits, no gain; not supported by the data |
| exclusions | Zr-Zr-Zr essential (+0.10 when removed), Cu-Cu-Cu kept; Cu-Cu-Zr and Cu-Zr-Zr dropped (within noise, MD cost down ~3×, training error up 28 %) |
| λ, refine | λ unchanged; 2-body order refined to 6 |

**Model.** 52 coefficients. Holdout relative force error 0.31 ± 0.06
(force RMSE 2.82 kcal/mol/Å) and energy RMSE 0.85 kcal/mol/atom, with
training error the same (0.31): no overfitting, and the data is the limit.

**MD validation.** 2,000-atom B2 CuZr, 10 ps NVT at 300 K and 1,200 K:
both stable (no NaNs, no lost atoms, temperatures held).

**Benchmark** (`chimes-benchmark`, one Dane node):

| | |
|---|---|
| cost per atom-step | 2.1×10⁻⁴ core-s (1 rank) → 3.8×10⁻⁴ (full 112-rank node) |
| strong scaling (8,192 atoms) | ≥70 % efficient to 28 ranks (~290 atoms/rank) |
| weak scaling (~250 atoms/rank) | 70-75 % to 16 ranks, 55 % on a full node |
| 100,000 atoms × 1 ns | ≈10,600 CPU-hours, 4 nodes, ~24 h |

**Development cost** (`usage`): 103 CPU-hours charged, 7.3 used, across 12
jobs. Most of the waste came from hyperparameter searches that held full
nodes for ~30 short fits. They now request 30 cores, which would have cut
that phase's charge ~4×.

**Deploy and report.** `deploy` wrote `06_deploy/MODEL_CARD.md`, and
`study-report` plus the report writer produced `REPORT.md`.

## What the report concluded

1. Open data alone gives a usable but rough Cu-Zr model, stable in short
   crystalline MD.
2. 3-body cutoffs must reach toward the second neighbour shell; 4-body terms
   are not supported by 126 frames.
3. Per-core MD cost rises ~1.8× as a node fills, so production estimates must
   use the packed-node cost.
4. Next: active learning at liquid and quench conditions, then a re-search
   with the grids widened past 8 Å (2-body) and 7 Å (3-body), and
   re-judged exclusions; multi-node benchmark before large runs.

## Since then

The same data was re-searched with the later tooling (`hyper-search
--cv-folds 4`, smoothing, per-pair λ, α and stress stages; 36 fits on
`pdebug` in 9 min): 2-body order 6 at 7.0 Å, 3-body order 4 at 5.77 Å (a
refined midpoint), Cu-Cu-Zr and Cu-Zr-Zr excluded, CUBIC smoothing, α =
1e-5. Cross-validated relative force error 0.411 ± 0.035 (external holdout
0.33), 11 fragile frames; the learning curve plateaus from 63 frames for
forces while energy error keeps falling, so more of the same data will not
improve forces. `md-check` harvested 46 frames at 300/1200 K, all novel by
fingerprint and QUESTS; `al-batch` cut them to 15 to label. The numbers
are in [hyper-search](../commands/hyper-search.md),
[learning-curve](../commands/learning-curve.md),
[fingerprint](../commands/fingerprint.md) and [al-batch](../commands/al-batch.md).

## The same study as commands

```bash
S=/p/lustre2/me/studies/cuzr
chimes-agent study --init $S --elements Cu,Zr --goal "Cu-Zr from MatPES"
chimes-agent data-fetch --source colabfit:colabfit/MatPES-PBE-2025.2 --elements Cu,Zr --all-frames --output-dir $S/01_data/fetch
chimes-agent data-curate --frames $S/01_data/fetch/pool.xyzf --elements Cu,Zr --output-dir $S/01_data/curate
chimes-agent hyper-search --data-manifest $S/01_data/curate/data_manifest.json \
  --s-maxim-3b 4.09,5.2,6.33,7.0 --cv-folds 4 --machine dane --queue debug --walltime-hours 1 --output-dir $S/02_fit   # approve, wait
chimes-agent benchmark --params $S/02_fit/search/best/params.txt \
  --prototype '{"name":"CuZr","crystalstructure":"cesiumchloride","a":3.26}' \
  --elements Cu,Zr --masses '{"Cu":63.546,"Zr":91.224}' --strong-atoms 8000 \
  --machine dane --queue debug --output-dir $S/05_bench                                                # approve, wait
chimes-agent benchmark --collect $S/05_bench
chimes-agent study --study $S --register benchmark=$S/05_bench/benchmark.json
chimes-agent usage --study $S && chimes-agent deploy --study $S && chimes-agent study-report --study $S
```
