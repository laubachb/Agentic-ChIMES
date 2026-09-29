# Study layout and artifacts

```
<study>/
  study.json          registry: goal, elements, artifact paths, extra roots (chimes-agent study)
  STUDY.md            the orchestrator's plan and decision log
  REPORT.md           final report (study-report + chimes-report-writer)
  REPORT_FACTS.json   every number in the report, collated from the artifacts below
  01_data/
    DATA_PLAN.md      data agent's plan: sources, strategy, decisions, gaps
    fetch_*/          pool.xyzf + provenance.json per source (data-fetch)
    generate/         structures.xyzf for QE (data-generate)
    qe_*/             QE labeling runs; labeled.xyzf + provenance.json (qe-relabel)
    curate/           curated.xyzf, train.xyzf, holdout.xyzf, data_manifest.json, curation_report.json
  02_fit/
    analysis/         hyper_analysis.json: distances, RDF shells, cutoff candidates
    search/           hyper_report.json, points/<hash>/ (one fit each), best/{params.txt, fm_setup.in, hyper_choice.json}
    HYPER_REPORT.md   the tuner's write-up
  03_al/              al_driver study or al-select rounds
  04_md/<run>/        LAMMPS validation runs (log.lammps)
  05_bench/           benchmark cases, benchmark.json, BENCHMARK.md
  06_deploy/          params.txt, fm_setup.in, in.lammps.example, MODEL_CARD.md, model_facts.json
  usage/              usage_report.json, usage_jobs.csv, local.jsonl (login-node CPU ledger)
```

**Handoffs.** Each phase ends with one file the next phase reads:
`data_manifest.json` → `hyper_report.json` / `best/` → `params.txt` →
`benchmark.json` → `MODEL_CARD.md` → `REPORT.md`.

**The registry.** `chimes-agent study --study DIR --register key=path`
records where an artifact is when it is not in the standard place (keys:
`data_manifest`, `hyper_report`, `params`, `fm_setup`, `al_run`,
`md_runs`, `benchmark`, `usage`, `deploy`). `chimes-agent study --study DIR`
shows what is present and what is missing.

**Provenance.** Every fetched or labeled pool has a `provenance.json`: source,
license and DOI, level of theory (for QE, a hash of the DFT settings), and
per-frame ids. Curation carries it into `data_manifest.json`, which the
model card and report cite.

**Resumability.** Stages remember their inputs per output directory and
short-circuit on a repeat. Hyperparameter fits are cached per configuration
under `points/`.
