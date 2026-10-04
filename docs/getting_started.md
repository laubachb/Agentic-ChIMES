# Getting started

## 1. Install

```bash
git clone https://github.com/laubachb/Agentic-ChIMES.git
cd Agentic-ChIMES
pip install -e ".[data,al-select,plots,quests]"
```

- The core install is `numpy`, `pyyaml` and `jsonschema`.
- `data` adds ASE, pyarrow and the Hugging Face client for the open-data
  stages.
- `al-select` adds matplotlib, which al_driver's selector imports.
- `plots` adds matplotlib for the PNGs the validation stages and the report
  produce; `quests` adds QUESTS (information entropy of environments) for
  `quests`, `--selection quests` and `al-batch`.
- `dev` and `docs` add pytest and MkDocs. `chimes-agent doctor` reports
  which optional pieces are missing.

## 2. Build the ChIMES toolchain

```bash
export CHIMES_ACCOUNT=<your Slurm bank or allocation>   # read by the machine profiles
chimes-agent setup --machine dane --component all      # or --machine ./my_cluster.yaml
chimes-agent doctor --machine dane                     # verify: components, numerics, profile
```

`doctor` runs one real LAMMPS and one calculator evaluation against a
published reference, and checks the machine profile (account, shared scratch,
partitions). Fix anything it marks `fail` before starting a study. See
[machine profiles](concepts/machine_profiles.md) for other clusters.

This clones the three upstream ChIMES repositories (`al_driver`,
`chimes_lsq`, `chimes_calculator` forks) into `codes/` at pinned commits
(gitignored; see [vendored forks](concepts/vendored_forks.md)). It then runs
their own `install.sh` scripts under your machine's modules, and builds
LAMMPS with the ChIMES pair style and Quantum ESPRESSO. Build one piece at a
time with `--component chimes_lsq | chimes_calculator | lammps |
quantum_espresso`.

Machines: `dane` (LLNL LC) and `stampede3` (TACC) are built in. For another
Slurm cluster, `chimes-agent setup --init-profile ./my_cluster.yaml --scratch
<shared dir> --modules <compiler>,<mpi>` writes a profile from what the
scheduler reports (partitions, cores per node, your accounts) and lists what
is left to fill in; pass the file path to `--machine`
([machine profiles](concepts/machine_profiles.md)).

Optional but recommended: `export HF_TOKEN=...` (a free Hugging Face read
token). Anonymous downloads of open datasets are rate-limited per IP address,
and lab networks share one.

## 3. Run a study with Claude Code

Open Claude Code in the repository directory. It reads `CLAUDE.md` and the
agents and skills under `.claude/` automatically. Then describe what you
need:

> I need a ChIMES potential for Cu-Zr metallic glasses, liquid and amorphous,
> 300-2000 K. Label with Quantum ESPRESSO on Dane (bank pls2). Put the study
> in /p/lustre2/me/studies/cuzr.

What happens next is described in [Running a study](guide/running_a_study.md).
In short:

- You're asked only what changes the plan: which DFT settings the labels
  must match, compute budget, production MD sizes.
- Every cluster job is shown to you as a dry run first, and nothing is
  submitted without your "yes".
- The study directory fills phase by phase and ends with `REPORT.md` and a
  deployable model. You can stop at any point and ask to resume later.

## 4. Try the stages without HPC

The stages work without the agents. This fits a tiny model to a bundled
ChIMES test fixture in a few minutes on a login node:

```bash
export FM=codes/chimes_lsq-LLfork/test_suite-lsq/test_4atoms.2
export OUT=/tmp/$USER/chimes-quickstart

chimes-agent fm-setup-gen --trjfile "$(pwd)/$FM/dump2.xyzf" --nframes 250 \
  --elements C,H --order '{"2":6,"3":2}' \
  --pair-cutoffs '{"C-C":[1.29,5.0],"C-H":[1.29,5.0],"H-H":[0.9,5.0]}' --output-dir $OUT
chimes-agent amat-build --fm-setup-in $OUT/fm_setup.in --output-dir $OUT
chimes-agent solve --algorithm lassolars --alpha 1e-5 --A $OUT/A.txt --b $OUT/b.txt \
  --header $OUT/params.header --map $OUT/ff_groups.map --output-dir $OUT
chimes-agent evaluate --params $OUT/params.txt --holdout-xyzf "$FM/dump2.xyzf" --max-frames 25
```

Each command prints one JSON object. `chimes-agent <stage> --describe` prints
a stage's full input/output schema, and `chimes-agent --help` lists them all.
For the whole manual workflow, see the
[stage-by-stage tutorial](tutorials/end_to_end_holdout_study.md).

## Troubleshooting

- **`chimes-agent: command not found`**: pip installed to `~/.local/bin`;
  add it to `PATH`, or run `python3 -m agentic_chimes.cli <stage> ...`.
- **`ComponentNotInstalled`**: run `chimes-agent setup --component <name>
  --machine <name>`, or point the named `AGENTIC_CHIMES_*_BIN/_LIB`
  variable at an existing build.
- **`solve` import error inside `chimes_lsq.py`**: it runs with the same
  Python as `chimes-agent`, which needs numpy, scipy and scikit-learn.
- **"was written for different inputs"**: a stage's output directory
  remembers its inputs; pass `--force` to redo it, or use a new directory.
  (Dry runs record nothing, so a dry run never blocks the real run.)
- **Slurm job COMPLETED but no output**: its directory was on a node-local
  path such as `/tmp`, which compute nodes cannot see. Keep studies on
  `/p/lustre2` (or your cluster's shared scratch).
- **HTTP 429 from data stages**: set `HF_TOKEN` (see step 2).
- **`chimes_lsq` segfaults inside a Slurm job**: fixed. MPI binaries run
  as a plain process inside a job step used to inherit the step's MPI
  environment; stages now start them clean. If you launch one yourself inside
  a job, unset the `PMI_*` variables first.
- **Under-allocated Dane jobs**: a bare `-N 1` gets one core; stages always
  request cores explicitly ([machine profiles](concepts/machine_profiles.md)).
