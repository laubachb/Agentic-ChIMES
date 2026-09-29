---
name: chimes-study-report
description: Write the final, human-readable report of a ChIMES study - data selection and curation, hyperparameter search, the final model, active learning, MD validation, scaling/performance, compute used, deployment - from the study's own artifacts. Use when a study is finished or paused and the user wants a write-up, summary, or report for colleagues, or asks "what did we do and what did we find". The chimes-report-writer subagent follows this playbook.
---

# Final study report

The report is for a scientist who was not in the conversation: what was
built, why each choice was made, how good the model is, what it costs, and
what not to trust yet. Every number must come from the study's artifacts.

## 1. Make the facts complete

1. `chimes-agent study --study <dir>`: check `missing`. Register anything
   that exists but is not registered (`--register key=path`; `md_runs`
   repeats). Missing phases are reported as gaps, not invented.
2. If the model is final and `06_deploy` is missing:
   `chimes-agent deploy --study <dir>` (writes the model card).
3. `chimes-agent usage --study <dir>` for current compute totals.
4. `chimes-agent study-report --study <dir>` writes `REPORT_FACTS.json` and
   `REPORT.md` with every table filled and `<!-- NARRATIVE: ... -->`
   placeholders.

## 2. Write the narrative

Replace each placeholder with prose. Keep the generated tables. Read, in
addition to the facts: `STUDY.md` (goal, decisions), `01_data/DATA_PLAN.md`,
`02_fit/HYPER_REPORT.md`, `05_bench/BENCHMARK.md`, the model card, and the
`hyper_report.json` stage reasons.

- **Summary** (first, 3-5 sentences): what model, for what, how accurate
  (holdout relative force error with its ± and what that means), how
  expensive to develop and to run, the main caveat.
- **Data**: source(s) and why, level of theory, what curation removed and
  why, coverage per pair, gaps against the goal.
- **Hyperparameters**: the decision at each stage *and the reason* (tie
  → cheaper model; 4-body gave nothing; which cluster types mattered).
  Explain the tie rule once in plain words.
- **Final model / MD / performance / cost**: interpret, don't restate.
  Is the holdout error good for the purpose? Were MD runs stable, and at
  which temperatures? What should a production request look like?
- **Findings, caveats, next steps**: numbered. Every gap and every search
  note that still applies becomes a caveat. Next steps are concrete
  ("run active learning at 1,200 K", "extend the 2-body grid past 8 Å").

### Literature context

Load `chimes-literature` for the methods and discussion:

- Cite the papers whose practice the study followed (λ at the RDF peak,
  inner cutoff below the closest contact, al_driver's selector) and name any
  departures (CUBIC smoothing, no stress data, holdout-only selection).
- To compare accuracy with published models, use `reduced_force_rmse`
  (RMSE ÷ mean |F|), the literature's convention, and say which convention
  every number uses. Never set `relative_force_error` against a published
  "reduced RMSE".
- Use the "Validation" section to state which published validation checks
  were done and which remain (RDF vs DFT, equation of state, diffusion,
  NpT density).
- End with a references list: author, journal, year, DOI.

## Rules

- No number that is not in `REPORT_FACTS.json` or an artifact you can cite.
  If you compute one (a ratio, a sum), say how.
- Uncertainty is part of the result: report ± and say when differences are
  within noise.
- Plain language first, jargon defined once (e.g. "relative force error:
  force RMSE divided by the typical force").
- Do not claim validation that was not done (no MD → say so).
- Keep it readable: roughly 2-4 pages; the tables carry the detail.

Deliverable: `<study>/REPORT.md` (and `REPORT_FACTS.json`). If the user wants
it shared, offer to publish it as a page.

Reference: `docs/commands/study-report.md`.
