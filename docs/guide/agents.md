# The agents

Claude Code subagents live in `.claude/agents/`, and the playbook each one
follows is in `.claude/skills/`. The main conversation orchestrates them
(skill `chimes-study`). Subagents cannot talk to you directly, so anything
needing your decision comes back through the orchestrator.

| Agent | Phase | Does | Never does | Returns |
|---|---|---|---|---|
| `chimes-data-curator` | data | plans the dataset; searches ~500 open DFT datasets; fetches, generates, curates; checks level-of-theory consistency and pair coverage | submits QE jobs; merges incompatible labels; invents DFT settings | `data_manifest.json`, `DATA_PLAN.md`, decisions, QE dry run |
| `chimes-hyperparameter-tuner` | fit | analyzes distances/RDFs; plans and interprets the staged search over cutoffs, λ, 2/3/4-body orders and cluster exclusions | runs long searches on the login node; submits jobs; calls statistical ties wins | `hyper_choice.json`, `HYPER_REPORT.md`, search job for approval |
| `chimes-md-validator` | MD | plans `md-check` runs of the chosen and tied models; judges stability, close contacts, RDFs vs DFT; fingerprint coverage of MD frames | submits jobs; ranks by RDF without a reference or within noise | `MD_REPORT.md`, verdict (use X / choose X over Y / needs active learning), next batch |
| `chimes-active-learner` | active learning | runs rounds: harvest → fingerprint novelty → select → QE (for approval) → `al-merge` → refit with n/I decay → re-validate; decides when to stop | changes the holdout or level of theory; re-searches every round; submits jobs | `AL_LOG.md`, per-round errors, stopping evidence, next batch |
| `chimes-fit-reviewer` | fit | independent, skeptical read of a finished fit: holdout integrity, grid edges, overfitting, cutoffs vs cell sizes | edits or refits anything | verdict + numbered findings |
| `chimes-benchmark` | benchmark | CPU-hours used (charged vs used, by phase); strong/weak LAMMPS scaling; sizing for production runs | submits jobs; estimates without stating the basis | `BENCHMARK.md`, `benchmark.json`, `usage_report.json` |
| `chimes-report-writer` | report | collates every artifact; writes the narrative of `REPORT.md` | invents numbers; claims validation that was not done | `REPORT.md` + gaps |
| `chimes-job-monitor` | any | cheap read-only status of Slurm jobs, DLARS/QE/al_driver progress | cancels or changes anything | ten-line status |

Playbooks (usable directly in the main conversation, too):
`chimes-study`, `chimes-data-curation`, `chimes-hyperparameter-search`,
`chimes-build-model`, `chimes-auto-build`, `chimes-active-learning`,
`chimes-hpc-jobs`, `chimes-benchmarking`, `chimes-study-report`,
`chimes-md-validation`, `chimes-literature`.

**Literature grounding.** `chimes-literature` distills the published ChIMES
methodology (see [ChIMES literature](../concepts/literature.md)). Every
agent reads it before recommending cutoffs, smoothing, orders, weights or
active-learning settings, and cites the papers in its reports. To give the
agents the full text, put the PDFs in `chimes_papers/` (gitignored) and run
`tools/index_papers.sh`. You can add your own papers the same way.

Every phase of the workflow now has an agent. `chimes-md-validator` and
`chimes-active-learner` follow the `chimes-md-validation` and
`chimes-active-learning` playbooks.

To add an agent, see [Claude Code integration](../concepts/claude_code_integration.md#adding-your-own-skill).
`tests/unit/test_claude_setup.py` checks every agent and skill is
well-formed and referenced.
