# Using Agentic ChIMES with Claude Code

The repo ships a ready-made setup so that opening Claude Code in this
directory is enough. You describe a goal in plain language; Claude runs
`chimes-agent` stages, reads their JSON, and decides the next step.

> "Build a ChIMES model from `configs.xyzf`. Label with QE on Dane, 4 hours."
>
> "The model in `study/run3/params.txt` is unstable in MD — select 50 new
> configurations to relabel."
>
> "Compare 2-body orders 8–16 on this labeled set and tell me if the extra
> order is worth it."

## The mental model

Claude Code has four mechanisms. Each is used for what it is good at:

| Piece | Where | Loaded | Used for |
|---|---|---|---|
| **CLAUDE.md** | repo root | every session, always | Rules that prevent expensive mistakes, how to drive the CLI, a map of what else exists. Kept short because it costs context every time. |
| **Skills** | `.claude/skills/<name>/SKILL.md` | on demand, when the task matches the skill's `description` | Playbooks: which stage order, what to check, how to read results. Detail that only matters for one kind of task. |
| **Subagents** | `.claude/agents/<name>.md` | when Claude delegates | A separate worker with its own context, tools and (optionally) cheaper model. Keeps noisy output out of the main conversation. |
| **Settings** | `.claude/settings.json` | every session | Hard guardrails: what Claude may run without asking, must ask about, or may never touch. |

The CLI itself is the fifth piece and the most important: every stage is a
command with a `--describe` schema, JSON in, **exactly one JSON object out**,
and idempotent per `--output-dir`. That contract is what makes the workflow
reliable, because Claude never has to scrape prose. See
[Stages and contracts](stages_and_contracts.md).

## What is included

**Skills** (`chimes-*`), each pointing at `--describe` and the docs for exact
flags rather than duplicating them, so they don't go stale:

| Skill | Triggers on |
|---|---|
| `chimes-study` | an end-to-end goal ("I need a potential for X"): orchestrates the five study phases, layout, handoffs, approvals |
| `chimes-data-curation` | finding, fetching, generating and curating training data (the data agent's playbook) |
| `chimes-hyperparameter-search` | choosing cutoffs, Morse lambdas and 2/3/4-body orders (the tuner agent's playbook) |
| `chimes-benchmarking` | CPU-hours used, strong/weak scaling, sizing compute requests (the benchmark agent's playbook) |
| `chimes-study-report` | the final human-readable study report (the report writer's playbook) |
| `chimes-auto-build` | "build me a model from these configs", one-shot pipeline, QE labeling then fit |
| `chimes-build-model` | hands-on fitting, choosing cutoffs/orders, sweeps, diagnosing a bad fit |
| `chimes-hpc-jobs` | anything touching Slurm, QE, DLARS, lustre quota, or a job that seems stuck |
| `chimes-active-learning` | `al-select`, `al-run`, "improve/stabilize the model with more data" |
| `chimes-md-validation` | "is this model stable in MD", choosing between tied models, fingerprint coverage (the MD validator's playbook) |
| `chimes-literature` | published ChIMES practice (cutoffs, λ, smoothing, orders, weights, AL, validation, accuracy) with citations; consulted by every agent |

**Subagents:**

- `chimes-benchmark`: accounts for the CPU-hours a study used and measures
  the final model's strong/weak scaling, turning it into CPU-hour estimates for
  production runs (submits nothing itself).
- `chimes-report-writer`: collates the whole study into `REPORT.md`, every number
  traced to an artifact.
- `chimes-hyperparameter-tuner` — the hyperparameter agent. Analyzes the
  curated data, plans the cutoff/λ/order search, returns the Slurm job for
  approval, then interprets the result into `hyper_choice.json` and a
  written report.
- `chimes-data-curator` — the data agent. Plans the dataset from the
  request, searches open databases, fetches or generates structures,
  curates them, and returns `data_manifest.json` plus any decision it
  can't make (label source, QE submission, which it only dry-runs). See
  [The data phase](data_curation.md).
- `chimes-md-validator` — MD validation: `md-check` of the chosen and tied
  models (stability, close contacts, RDF vs DFT), fingerprint coverage, and
  a verdict: use a model, choose between models, or needs active learning.
- `chimes-active-learner` — active-learning rounds (harvest, fingerprint
  novelty, QE labeling for approval, `al-merge`, refit with n/I weight decay,
  re-validation) with an explicit stopping rule.
- `chimes-job-monitor` — read-only, runs on the small fast model. Claude
  hands it a job id or directory and gets back a ten-line status instead of
  raw `squeue`/`sacct`/log output. This is the right place for "wait and
  check" work.
- `chimes-fit-reviewer` — an independent, skeptical read of a finished
  sweep, `evaluate` or `auto-build` result: holdout integrity, grid-edge
  winners, cutoffs vs. box size, overfitting. It did not produce the result,
  so it has no reason to flatter it.

**Permissions** (`.claude/settings.json`):

| Rule | Effect |
|---|---|
| allow | `chimes-agent *`, `pytest`, `mkdocs build`, `squeue`/`sacct`/`sinfo`, `lfs quota`, read-only `git` |
| ask | `submit`, `qe-relabel`, `al-run`, `auto-build`, any `--machine` invocation, `scancel`, `sbatch` — things that spend allocation or affect running work |
| deny | `scancel -u` (all your jobs), `rm -rf codes/deps`, editing `codes/` or `deps/` |

Personal overrides go in `.claude/settings.local.json` (gitignored).

## How a session behaves

- Claude reads `CLAUDE.md` first, loads a skill when your request matches
  one, and calls `chimes-agent <stage> --describe` rather than guessing flags.
- Anything that submits to Slurm is previewed with `--dry-run` and you are
  asked before the real submission (both by the skill's instructions and by
  the `ask` permission rules).
- Long stages (`auto-build`, HPC `sweep`, DLARS solves) run in the
  background; Claude keeps working or delegates waiting to
  `chimes-job-monitor`, and reads the result file when it lands.
- Before recommending a model, Claude is instructed to have
  `chimes-fit-reviewer` look at it.
- It is a **tool-calling** design, not an autonomous loop: Claude proposes
  and runs steps, reports, and stops at decision points (submitting HPC
  work, choosing a model). The one stage that decides for you is
  `auto-build`, and it returns its full reasoning in `trace`.

## Design choices worth knowing

- **stdout is reserved for the JSON result.** Native libraries (chimes_calculator's
  C++ banner) and al_driver print freely; the CLI diverts all of it to
  `<output-dir>/<stage>.log` and returns the path as `stage_log`. Without this,
  one `al-select` call put ~1,600 lines ahead of the JSON.
- **Skills carry judgment; the CLI carries contracts.** Cutoff heuristics,
  when a sweep table is meaningless, what a DLARS cliff means — that is what
  a skill teaches. Flag lists are not duplicated there.
- **No MCP server.** The CLI plus the Bash tool already gives structured
  calls; a server would add a second interface to keep in sync. If you later
  want these stages callable from non-Claude agents, wrapping `--describe`
  schemas as MCP tools is straightforward, because the schemas already exist.
- **The `ask` rules can't see inside `--json-in` files.** A stage that gets
  `machine` from a JSON file bypasses the `--machine*` pattern. The stage
  names `submit`, `qe-relabel`, `al-run` and `auto-build` are gated by name
  for that reason, and the skills tell Claude to confirm before real
  submissions regardless.

## Keeping it healthy

`tests/unit/test_claude_setup.py` fails when: a skill or agent has broken
frontmatter (which makes it silently never load), a name doesn't match its
path, `CLAUDE.md` stops mentioning a stage, or a skill/`CLAUDE.md`
references a skill, agent or `chimes-agent <stage>` that doesn't exist.

When you add a stage: implement it, register it in `cli.STAGE_MODULE_NAMES`,
add it to `CLAUDE.md`'s stage list, and mention it in the relevant skill.
Changes to skills and agents take effect in new Claude Code sessions.

## Adding your own skill

Create `.claude/skills/<name>/SKILL.md` with `name` (same as the directory)
and a `description` that says *when* to use it — the description is the only
thing Claude sees when deciding whether to load the skill, so write it as a
list of situations, not a title. Keep the body to what Claude can't get from
`--describe`: order, judgment, gotchas.
