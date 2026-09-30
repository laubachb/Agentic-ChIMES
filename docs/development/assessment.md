# Assessment: gaps, bugs and improvements for general users

*Audit of the repository as of 2026-09-29, after the benchmark and report
agents landed. The question it answers: what stops someone who is not
the author, on a machine that is not Dane, from getting a trustworthy
ChIMES model out of this toolkit, and what should be fixed first?*

Scope:

- the CLI stages;
- the Claude Code layer (CLAUDE.md, skills, agents, permissions);
- the docs;
- a comparison against the published ChIMES methodology
  ([literature](../concepts/literature.md)).

Items marked **verified** were reproduced or confirmed in the code. The
others are design gaps.

## Summary

The pipeline works end to end on Dane: data → hyperparameters → model →
MD → benchmark → report were all validated with real jobs on the Cu-Zr
example. The main risks for a general user are:

1. **Portability.** Machine profiles, environments and the Slurm path are
   tuned to one user on one cluster, and the dry run does not show what will
   actually be submitted.
2. **Silent staleness.** Caches and manifests key on paths and partial
   configuration, so changed inputs can return old results.
3. **Method gaps against the literature.** CUBIC smoothing for many-body
   terms, holdout-only model selection, and no fitting weights or stresses.
   Each is a place where the toolkit's answer can differ from what a ChIMES
   expert would choose.
4. **Missing phases.** There is no MD agent and no active-learning agent,
   so phases 5-6 of the intended workflow are still manual playbooks.

## Status (updated 2026-09-29, same day)

Fixed since the audit, validated by tests and on Dane:

- **Bugs:** B1-B4 and B6-B10. B5 (PDFs in git history) needs the
  repository owner.
- **Portability:**
  - `${VAR}` profile expansion (`CHIMES_ACCOUNT`);
  - jobs run the submitting interpreter;
  - `setup` takes profile paths.
- **Accuracy:**
  - group-aware holdout splits;
  - `weights` stage with the published presets;
  - `hyper-search --prefer richer`;
  - `md-check` (MD validation of candidates plus the active-learning
    harvest);
  - triclinic LAMMPS support;
  - automatic replication of thin cells in `lammps-run`. This was a newly
    found bug: LAMMPS energies were 30-50 kcal/mol off on 2-atom cells.

Still open:

- `doctor` / self-test;
- stresses through the data stages;
- per-pair 3-body cutoffs;
- hierarchical fitting and fingerprint wrappers;
- more data backends;
- the MD and active-learning *agents* (their stages now exist);
- a CI integration tier;
- slimming `auto-build`;
- the default smoothing. The CUBIC-vs-TERSOFF comparison on Cu-Zr
  (`docs/commands/hyper-search.md`) kept CUBIC for small data. TERSOFF made
  many-body terms active but overfit and did not improve forces on 126
  frames. Revisit on a larger dataset.

## 1. Bugs

Ranked by consequence.

| # | Severity | Bug | Where | Consequence | Fix |
|---|---|---|---|---|---|
| B1 ✅ | High | **Dry run renders a different script from the real submission.** Dry runs go through `hpc/dry_run.render_sbatch_script`, which adds `conda activate <env>`. Real submissions go through al_driver's `helpers.create_and_launch_job`, which does not (verified: the real `run.cmd` of a hyper-search job has no `conda activate`). | `hpc/slurm.py:93-117`, `hpc/dry_run.py:91` | The user approves one script and a different one runs. Jobs work on Dane only because the login environment is inherited. | Render once with the toolkit's renderer and `sbatch` that file in both modes. Test that dry run and real submission produce byte-identical scripts. |
| B2 ✅ | High | **Fit cache key ignores the data and solver.** `hyper-search` point caches key on `sha256(cfg)`; the solver/α come from the task, and the training/holdout files are not hashed. | `stages/_hyper.py:180-181` | Re-running with new data or another `--algorithm`/`--alpha` in the same output directory silently reuses old fits. | Hash the data file content (size + mtime at least), the solver, α and the evaluation settings into the key. |
| B3 ✅ | High | **Stage manifests hash paths, not content.** The short-circuit compares the input dict, including file *paths*. | `stages/_manifest.py:24-25` | Regenerating `train.xyzf` in place and re-running returns the previous result. | Include content hashes (or size + mtime) for file-valued inputs. |
| B4 ✅ | High | **Permission bypass.** `settings.json` allows `Bash(python3 -m agentic_chimes.cli *)` without asking, but the "ask" rules (qe-relabel, al-run, auto-build, `--machine`) only match `chimes-agent …`. | `.claude/settings.json:5,18-22` | An agent can submit Slurm jobs through the module form without the approval gate. | Remove the module-form allow, or mirror every ask rule for it. |
| B5 | High | **ChIMES papers were pushed publicly.** The PDFs in `chimes_papers/` were committed in `fe2f40c` and pushed. They are now untracked and gitignored, but remain in history. | git history | Publisher copyright exposure. | Rewrite history (`git filter-repo --path chimes_papers --invert-paths`) and force-push. This is the repository owner's decision. |
| B6 ✅ | Medium | **`setup --machine` rejects custom profiles.** `choices=available_profiles()` allows only the bundled names, although other stages accept a profile path. | `stages/setup_cmd.py:38` | A user on a new cluster cannot build with their own profile. | Accept a name or a path, as the other stages do. |
| B7 ✅ | Medium | **`--json-in` is not validated.** Arbitrary keys are `setattr` onto the args; typos are silently ignored, and wrong types fail deep inside the stage. | `cli.py:92-98` | "Why did my setting not apply?" | Validate against the stage `SCHEMA`: reject unknown keys, coerce or reject types. |
| B8 ✅ | Medium | **`--dry-run` is accepted but ignored by non-HPC stages.** | `cli.py` | A user expecting a preview gets real work. | Reject `--dry-run` on stages that do not implement it, or give them a plan-only mode. |
| B9 ✅ | Medium | **No shared-filesystem check.** `--output-dir` under `/tmp` with `--machine` submits a job whose output the login node never sees; it reports COMPLETED with nothing written. | all `--machine` stages | Lost jobs; this is only documented in CLAUDE.md. | Refuse (or warn) when the output dir is not under the profile's `scratch_root`. |
| B10 ✅ | Low | **Inconsistent α advice.** `chimes-build-model` says "1e-5 normalized, 1e-2 un-normalized" (the chimes_lsq docs); `hyper-search` uses 1e-5 raw `lassolars`, which Cu-Zr measurements supported. | skills | Agents may give contradictory advice. | Reconcile the text: state both sources and when each applies. |

## 2. Gaps for general users

### Portability and setup

- **Personal settings in bundled profiles.** `stampede3.yaml` hardcodes one
  user's allocation (`TG-CHM250118`) and scratch path, and `dane.yaml` the
  `pls2` bank and the `mat_mcts` conda env. Ship templates with `${ACCOUNT}`
  and `${USER}` placeholders, plus `chimes-agent setup --init-profile`, which
  asks for account, partition and scratch space.
- **Environment mismatch.** Local stages use `sys.executable`; Slurm jobs
  activate the profile's `conda_env` (dry run only, see B1). They should be
  the same interpreter. Record `sys.executable` in the profile at setup and
  use it everywhere.
- **No `doctor` / self-test.** Add a command that checks, in one place:
  - built components;
  - `chimescalc_lib` loads;
  - LAMMPS has ChIMES;
  - the MPI launcher works inside a Slurm step (the MVAPICH2 PMI problem
    found earlier);
  - the scratch root is shared;
  - `HF_TOKEN` is set.

  Also a 2-minute smoke study on the bundled example. Today these failures
  surface mid-study.
- **Single-scheduler, single-site assumptions.** Only Slurm, and the
  benchmark is tuned for 112-core nodes. Other sites need at least a
  documented porting checklist.

### Data

- **ColabFit is the only open-data backend.** Materials Project, OQMD,
  NOMAD and direct OMat24/MatPES downloads are not wrapped. Hugging Face
  rate limits (HTTP 429 on shared lab IPs) make even ColabFit fragile
  without `HF_TOKEN`.
- **No stresses from open data, and none fitted.** Force-only fits do not
  constrain pressure or density (Lindsey 2017, 2019). Carry stresses
  through `data-fetch`, `data-curate` and `fm-setup-gen --fitstrs` wherever
  a source provides them.
- **Holdout leakage risk.** Frames from the same MD trajectory or relaxation
  path are strongly correlated; a random split then overstates accuracy.
  Group-aware splitting by source trajectory exists in the solvers
  (`frame_groups`) but not in `data-curate` or `dataset-select`.
- **No quantitative coverage metric.** Cluster-graph fingerprints (Laubach
  2026) answer "does this data cover the target conditions?" and "is AL
  done?". The tool ships with chimes_calculator but is not wrapped. It is
  single-element for now.
- **Only DFT-MD-free data routes.** Open data is mostly relaxations. There
  is no stage to generate short DFT-MD (or DFTB-MD, Lindsey 2025)
  trajectories at target state points, which is how every published ChIMES
  base set was made.

### Fitting method (vs. the literature)

- **Smoothing.** Fits used CUBIC unconditionally.
  `hyper-search --smoothing 'TERSOFF 0.5'` now exists, but CUBIC is still
  the default. After a TERSOFF-vs-CUBIC comparison on Cu-Zr, consider making
  TERSOFF the default when 3-/4-body stages run. The Cu-Zr "4-body gives no
  gain" finding may be a smoothing artifact (Lindsey 2020 JCP).
- **Holdout-only selection.** Water showed holdout CV favoring overfit
  models that failed in MD (Lindsey 2019). Add an optional `md_check` stage
  to `hyper-search`: short `lammps-run` MD of the top tied candidates, with
  stability and an RDF comparison against the training data.
- **Parsimony before active learning.** The "cheapest tied model" rule
  suits a final model, but the literature says to err toward complexity
  before AL and prune after (Lindsey 2025). Expose
  `--prefer richer|cheaper` and default to `richer` for ALC-0 studies.
- **Global 3-body cutoff.** Per-pair 3-body cutoffs matched uniform
  accuracy at lower cost for water. Add a per-pair `s_maxim_3b` candidate
  set derived from each pair's shells.
- **No weighting schemes.** Uniform weights only. Two fixes:
  - a `weights` stage that writes the per-row file `solve --weights`
    already accepts, with the published presets (F/E/S 1/0.3/100; energy
    5; Boltzmann-style; AL decay n/I);
  - a plugin hook for group-specific schemes, which the intended workflow
    anticipates.
- **Hierarchical (element-block) fitting** is not exposed. al_driver 2.0
  supports it (`hierarch.py`, Lindsey 2026). It is the principled route for
  alloys and a natural alternative to exclusion sweeps.
- **`auto-build` duplicates `hyper-search`** with weaker logic (a single
  sweep, lowest-RMSE pick). Make it a thin wrapper:
  data-curate → hyper-search → optional AL.

### Phases 5-6: MD and active learning

- **No MD agent.** MD validation is a manual `lammps-run` plus reading
  logs. An agent (and stage) should:
  - run the target state points;
  - check stability, energy conservation and RDFs against DFT;
  - harvest close-contact and novel frames for AL.
- **No active-learning agent.** `al-select` and `al-run` exist, but nobody
  prepares al_driver's templates, `config.py` or the QE driver. The
  literature's recipe is concrete enough to automate: parallel state
  points, ≤20 close-contact + ≤20 other frames each, weight decay, and a
  fingerprint or MD-agreement stop.
- **Orthorhombic-only LAMMPS data and cutoff derivation.** Most
  open-database cells are triclinic. LAMMPS supports triclinic boxes, so
  write `xy xz yz` tilt factors.
- **Single-node benchmark.** Multi-node estimates are extrapolations.
  `--nodes` exists; recommend it for production-size requests.

### Claude Code layer

- **Agents depend on the author's context.** Several playbook numbers are
  Cu-Zr specific ("on Cu-Zr blocklasso was worse"). Mark them as examples
  and give the general rule first.
- **Nothing guards the approval gate except settings.json** (see B4).
  Consider a `PreToolUse` hook that blocks `sbatch` and `--machine` without
  `--dry-run` unless an approval file exists in the study.
- **No evaluation of the agents themselves.** Add a small scripted
  scenario, e.g. a toy system or "choose hyperparameters for this cached
  search", whose agent report is checked for required fields.

### Testing and CI

- CI runs unit tests only. Add:
  - an integration tier that builds `chimes_calculator` and `chimes_lsq`
    in CI (both are small) and runs one real fit and evaluate;
  - a nightly (or manual) HPC tier on Dane.
- Add the B1 dry-run vs real-script equality test and the B2/B3 cache-key
  tests.

## 3. Quality-of-life improvements

- `chimes-agent study --status` could print a one-screen dashboard: phases,
  what is done, what is waiting on the user, running jobs and CPU-hours so
  far.
- A `chimes-agent explain <artifact.json>` command that turns any stage
  output into three plain-language lines (useful to non-agent users too).
- Consistent units in every JSON (`*_kcal_mol_ang` suffixes everywhere;
  some keys still lack them).
- Progress output for long local stages (a status file in the output dir
  that `chimes-job-monitor` can read).
- Clearer error messages for the three most common failures: half-box
  cutoff violations, missing built components, and HF 429. Each should end
  with the fix command.
- Example studies beyond Cu-Zr: a single-element (carbon, to match the
  literature) and a molecular system (water), each with expected numbers,
  so users can check an installation against published values.
- Store a `STUDY.md` template with the decision log headings the
  orchestrator expects.

## 4. Literature integration (done in this pass)

- `chimes-literature` skill: 8 papers distilled by decision, with
  citations and published numbers. Every agent and playbook points to it,
  and CLAUDE.md requires citing it when making technical choices.
- `chimes_papers/` gitignored. `tools/index_papers.sh` builds a searchable
  text copy for agents; users can add their own PDFs.
- `evaluate` reports `reduced_force_rmse`, the literature's metric, so
  accuracy can be compared with published models (Cu-Zr: 0.46; water
  0.24-0.31; Carbon 2.0 0.28).
- `hyper-search --smoothing` enables TERSOFF, with a report note when
  many-body terms were fitted with CUBIC.
- Public summary: [ChIMES literature](../concepts/literature.md).

## 5. Roadmap

Ordered by value per effort for a new user.

1. **Safety and correctness (days):**
   - B1: one renderer, dry run equals real;
   - B4: close the permission bypass;
   - B2/B3: content-aware cache and manifest keys;
   - B9: shared-filesystem check;
   - B5: history rewrite (owner's decision).
2. **Portability (about a week):**
   - profile templates + `setup --init-profile`, and custom profile paths
     (B6);
   - `chimes-agent doctor`;
   - one interpreter everywhere;
   - `--json-in` validation (B7).
3. **Method alignment with the literature (1-2 weeks):**
   - TERSOFF-vs-CUBIC comparison on Cu-Zr, then choose the default;
   - MD check of tied candidates in `hyper-search`;
   - `--prefer richer` before AL;
   - a `weights` stage with published presets;
   - stresses carried through from data to fit;
   - group-aware holdout splits.
4. **MD agent**: stability, RDF and energy-conservation checks, and
   close-contact frame harvesting. Triclinic LAMMPS data.
5. **Active-learning agent**: al_driver study preparation from a study
   directory, the parallel state-point recipe, weight decay, and a stopping
   rule (fingerprints or MD agreement).
6. **Breadth:**
   - more data backends;
   - hierarchical element-block fitting;
   - fingerprint coverage reports;
   - carbon and water example studies checked against published numbers;
   - CI integration tier.
