# Agentic ChIMES

Build [ChIMES](https://chimes-lsq.readthedocs.io/) machine-learned interatomic
potentials by describing what you need. Open the repository in
[Claude Code](https://claude.com/claude-code), say *"I need a ChIMES
potential for liquid Cu-Zr"*, and a team of specialist agents does the
following:

- finds and curates training data from open DFT databases, or generates
  structures and labels them with Quantum ESPRESSO;
- chooses cutoffs, Morse λ and polynomial orders;
- fits, reviews and validates the model in LAMMPS;
- measures what it costs to run and what it cost to build;
- writes the study up.

You approve every cluster job, and every step is a plain command
(`chimes-agent <stage>`) you can also run by hand.

<div class="grid cards" markdown>

- **[Getting started](getting_started.md)**: install, build the ChIMES
  toolchain, run your first study.
- **[Running a study](guide/running_a_study.md)**: what to ask for, what
  you will be asked, how to steer and resume.
- **[The agents](guide/agents.md)**: who does what, and what they never do.
- **[Worked example: Cu-Zr](guide/example_cuzr.md)**: a real study,
  numbers included.

</div>

## What a study produces

| | |
|---|---|
| **Model** | `params.txt` + `fm_setup.in` for LAMMPS `pair_style chimesFF` |
| **Model card** | accuracy ± uncertainty, the distances and compositions it is valid for, data provenance and licenses |
| **Cost model** | strong/weak scaling; CPU-hours for "N atoms for T ns" |
| **Compute ledger** | CPU-hours charged vs used, by phase |
| **Report** | `REPORT.md`: data → hyperparameters → model → MD → performance, with findings and caveats |

## The workflow

```
plan ─► data ─► hyperparameters ─► model check ─► active learning ─► MD ─► benchmark ─► deploy + report
```

Each phase ends with one file the next reads ([study layout](guide/study_layout.md)),
so a study can stop and resume anywhere. Guardrails:

- dry run and your approval before any cluster submission;
- one level of theory per fit;
- statistically tied models resolved in favor of the cheaper one;
- nothing written into the upstream ChIMES code.

## Using the stages directly

```bash
chimes-agent data-search --elements Cu,Zr          # open datasets for a chemical system
chimes-agent hyper-search --describe               # any stage's full input/output contract
```

Stdout is always one JSON object. See the [command reference](commands/index.md)
and the [stage-by-stage tutorial](tutorials/end_to_end_holdout_study.md).

## Repository layout

```
Agentic-ChIMES/
├── CLAUDE.md, .claude/       agent rules, subagents, playbooks (skills), permissions
├── src/agentic_chimes/       the chimes-agent CLI: stages, HPC layer, data sources, I/O
├── codes/                    gitignored: upstream ChIMES forks, cloned by `chimes-agent setup`
├── deps/                     gitignored: builds (LAMMPS, QE) and caches
├── tests/unit/               no HPC needed
└── docs/                     this site
```
