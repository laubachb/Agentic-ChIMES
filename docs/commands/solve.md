# `chimes-agent solve`

**Status: implemented for local algorithms** (`svd`, `fast_svd`, `ridge`,
`fast_ridge`, `ridgecv`, `lasso`, `lassolars`). **`dlars`/`dlasso` are not
wired up yet** — they need the HPC submission layer (multi-node
`srun`/`ibrun`); requesting them raises a clear `NotImplementedError`
rather than silently doing the wrong thing.

Solves the least-squares fit via `chimes_lsq.py` (subprocessed with
`sys.executable`, so it runs in the same Python environment as
`chimes-agent` itself — make sure `numpy`/`scipy`/`scikit-learn` are
available there, e.g. by activating the `mat_mcts`-style conda env your
machine profile expects before running `chimes-agent`). The script prints
the `params.txt` body to stdout (captured to `params.txt` by this wrapper)
ending with a literal `ENDFILE` line, and writes `force.txt` itself as a
side-effect file in its working directory.

## Usage

```bash
chimes-agent solve --algorithm svd \
  --A ./run1/A.txt --b ./run1/b.txt \
  --header ./run1/params.header --map ./run1/ff_groups.map \
  --output-dir ./run1
```

## Flags

- `--A`, `--b`, `--header`, `--map` PATH (required; from `amat-build`'s output)
- `--algorithm` (default `svd`)
- `--alpha` (default `1e-4`) — regularization for ridge/lasso/lassolars
- `--eps` (default `1e-5`) — SVD regularization
- `--weights PATH` — optional per-equation weight file
- `--folds` (default `4`) — CV folds for `ridgecv`

## Output

```json
{
  "params": "./run1/params.txt",
  "force": "./run1/force.txt",
  "algorithm": "svd",
  "log": "./run1/solve.log"
}
```

## The DLARS cliff (design, not yet wired to a live solve)

Near the end of the LARS path (~85% of features active) the active-set
Gram matrix can become ill-conditioned, and DLARS starts limping (~1
iter/min instead of ~4/sec) while emitting `Cholesky solution test
failed`/similar log lines instead of hanging outright.
`stages/_cliff_monitor.py`'s `CliffMonitor` parses exactly this log format
(`Finished iteration N`, the four/five known Cholesky-failure signatures)
and, on iteration-rate collapse while a failure is present, decides to
finalize from `(iteration at first failure) - finalize_margin` — using
DLARS's own native `restart.txt` checkpointing (written every 10
iterations automatically) rather than any hand-rolled checkpoint file. It's
fully unit-tested against synthetic log fixtures today
(`tests/unit/test_cliff_monitor.py`) and will be wired into a live `solve
--algorithm dlars` HPC submission in a later phase.
