# `chimes-agent solve`

**Status: implemented** — local algorithms (`svd`, `fast_svd`, `ridge`,
`fast_ridge`, `ridgecv`, `lasso`, `lassolars`) run as a plain subprocess;
`dlars`/`dlasso` submit via `--machine` and are **validated against a real
Slurm job on Dane** (not just unit-tested), including the cliff monitor
actually cancelling a real pathological run -- see below.

Solves the least-squares fit via `chimes_lsq.py`. Local algorithms run it
subprocessed with `sys.executable` (make sure `numpy`/`scipy`/
`scikit-learn` are available there, e.g. via the `mat_mcts`-style conda env
your machine profile expects). The script prints the `params.txt` body to
stdout ending with a literal `ENDFILE` line, and writes `force.txt` itself
as a side-effect file in its working directory.

## Usage

```bash
# local
chimes-agent solve --algorithm svd \
  --A ./run1/A.txt --b ./run1/b.txt \
  --header ./run1/params.header --map ./run1/ff_groups.map \
  --output-dir ./run1

# DLARS/DLASSO via Slurm
chimes-agent solve --algorithm dlars --alpha 1e-5 \
  --A ./run1/A.txt --b ./run1/b.txt --dim ./run1/dim.txt \
  --header ./run1/params.header --map ./run1/ff_groups.map \
  --machine dane --queue batch --walltime-hours 2 --nodes 1 --ntasks-per-node 112 \
  --output-dir ./run1
```

`--output-dir` for the DLARS path **must be on a shared filesystem**
(`/p/lustre2/...`, not `/tmp`) -- see
[Machine profiles](../concepts/machine_profiles.md#shared-filesystem-required-for-real-hpc-submissions)
for why (found the hard way validating this exact path).

## Flags

- `--A`, `--b`, `--header`, `--map` PATH (required; from `amat-build`'s output)
- `--dim PATH` — `dim.txt` from `amat-build`; **required** for `dlars`/`dlasso`
- `--algorithm` (default `svd`)
- `--alpha` (default `1e-4` local / use `1e-5` for dlars — see below)
- `--eps` (default `1e-5`) — SVD regularization
- `--weights PATH` — optional per-equation weight file
- `--folds` (default `4`) — CV folds for `ridgecv`
- `--normalize` (dlars/dlasso only, **default false** — see below)
- `--split-files` (dlars/dlasso only; must match `fm_setup.in`'s `SPLITFI`)
- `--machine`, `--queue`, `--walltime-hours`, `--nodes`, `--ntasks-per-node` (dlars/dlasso only)
- `--poll-interval-s` (default 60) — how often the cliff monitor tails the live log
- `--dry-run` (generic flag) — render the job script, don't submit (dlars/dlasso only)

### `--normalize` defaults to false — confirmed on a real run

`dlars` itself prints `"Warning: normalize should not be used with
chimes_lsq"` and, when invoked with `--normalize true` through
`chimes_lsq.py` anyway, immediately hit repeated Intel MKL errors
(`Parameter 7 was incorrect on entry to cblas_dgemv`) and stalled at
iteration 0 forever (`RMS Error: -nan`, `Number of vars: 0`) — this is
not a hypothetical, it happened on a real Dane submission during
validation, and the cliff monitor's failure-signature detection
(`"Iteration failed: continuing"`) correctly caught it and cancelled the
job. With `--normalize false` (now the default), the same input solved
cleanly on the first try. Don't override this unless you have a specific
reason to.

## Output

```json
{
  "params": "./run1/params.txt",
  "force": "./run1/force.txt",
  "algorithm": "svd",
  "log": "./run1/solve.log"
}
```

DLARS/DLASSO output additionally includes `"job_id"`, `"cliff_detected"`
(bool), and `"cliff_report"` (`null` unless a cliff was caught).

## The DLARS cliff, and its live cancel-and-finalize path

Near the end of the LARS path (~85% of features active) the active-set
Gram matrix can become ill-conditioned, and DLARS starts limping while
emitting `Cholesky solution test failed`/similar log lines instead of
hanging outright. `stages/_cliff_monitor.py`'s `CliffMonitor` parses this
log format (`Finished iteration N`, the known failure signatures) and, on
iteration-rate collapse while a failure is present, decides to finalize
from `(iteration at first failure) - finalize_margin`.

`stages/_dlars_hpc.py` wires this to a *live* job: it submits the solve,
then every `--poll-interval-s` tails the growing `dlars.log` into the
monitor. On a "finalize" decision it cancels the Slurm job (`scancel`) and
submits a second, short job: a fresh `dlars` invocation capped at
`--iterations=<target>` (LARS' path is deterministic, so a fresh run
re-traces the same pre-cliff solution — no restart-file parsing needed),
then `chimes_lsq.py --read_output true` to emit the final `params.txt`.
`tests/unit/test_dlars_hpc.py` validates this whole control flow (submit →
poll → detect → cancel → finalize) against a mocked HPC boundary; the
detection/cancel half was also confirmed against the real pathological run
described above (not the finalize half specifically — that run's
target_iteration was 0, an edge case that real cliffs deep into a large
basis won't hit).
