# Troubleshooting

Start with `chimes-agent doctor --machine <m>` and, for any submitted
job, `chimes-agent job-status --work-dir <dir>`. Then:

| symptom | cause | fix |
|---|---|---|
| a stage returns `{"error": ...}` with `log_tail` | the native code or a library failed | read the named log; the message usually names the fix |
| `... was written for different inputs` | the output directory holds a run with other inputs or files that changed since | new `--output-dir`, or `--force` if the old run is disposable |
| `another <stage> run is still running here` | a live process uses that directory | wait, or `--force` if it died |
| a job writes nothing and runs out its walltime | a worker pool forked after numba/OpenMP threads started and deadlocked in `fork()` | fixed: pools spawn (`io/pool.py`). In your own scripts, use `process_pool`, not a bare `ProcessPoolExecutor` |
| `job directory ... is on node-local storage` | `--output-dir` under /tmp | use the profile's scratch root |
| `--json-in: unknown key` | a typo in the JSON | fix the key; the error lists valid ones |
| `does not match its training data: NFRAMES ...` | `fm_setup.in` disagrees with its trajectory | regenerate with `fm-setup-gen` |
| `mass of X is ... but the model was fitted with ...` | LAMMPS matches types by mass | drop `--masses` (taken from params.txt) |
| `frame(s) contain elements the model ... does not describe` | wrong model for the data | pick the right params.txt |
| `COMPLETED_WITHOUT_RESULTS` from job-status | the job wrote nowhere visible, or an error was swallowed | `log_tail`; shared filesystem |
| `TIMEOUT` | walltime too short | raise `--walltime-hours`; hyper-search and qe-relabel resume from caches |
| HTTP 429 from `data-*` | Hugging Face rate limit on a shared IP | `export HF_TOKEN=...` |
| `Object is remote` on Lustre | transient metadata error | stages retry; re-run if it persists |
| holdout error ≫ CV or training error on one group | a holdout frame inside the inner cutoff, or a mislabeled frame | `evaluate` lists `worst_frames` and `n_frames_below_inner_cutoff` |
| 3-/4-body terms "add nothing" | tiny columns under CUBIC smoothing, or data too sparse | `--smoothing 'TERSOFF 0.5'` (or the `smoothing` stage); learning curve |
| MD samples inside the inner cutoff | penalty too weak or coverage missing | `deploy` writes an explicit penalty; label `md-check`'s harvest |
| `quests not installed` | optional extra | `pip install -e ".[quests]"` |
| plots missing | matplotlib not installed | `pip install -e ".[plots]"` |
| lustre file-count quota | too many small files | archive to lustre3; stages write packed files, but studies accumulate |
