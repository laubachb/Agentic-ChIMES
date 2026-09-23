# `chimes-agent al-run`

**Status: implemented, deliberately minimal scope.**

Launches al_driver's own active-learning loop (`main.py`) — this stage
does not reimplement active-learning orchestration (dataset build, solve,
MD, QM labeling, convergence retry — al_driver's `main.py` already does
all of that), it correctly *calls* the existing, working driver as a
properly detached background process. See
`src/agentic_chimes/stages/al_run.py`.

## What this does and doesn't automate

You prepare an al_driver study the normal way al_driver itself documents:
`ALL_BASE_FILES/` (ALC-0 initial training data, MD input templates, QM
input templates) plus a `config.py`. This stage does **not** generate that
`config.py` for you — al_driver's config surface is large
(`codes/al_driver-LLfork/doc/source/options.rst`) and deeply
system-specific (QM code, basis, HPC settings, MD ensemble, cluster
selection criteria, ...); templating it generically here would mean either
a huge parameter surface duplicating al_driver's own, or a leaky
abstraction that breaks the moment your study needs something the
template didn't anticipate. What this stage *does* handle correctly:
copying your `config.py` into place and launching `main.py` as a properly
detached, logged background process — the actual "just call the driver"
mechanics.

**Fastest way to see it work end to end**: al_driver ships a real, complete
minimal example that uses the already-built ChIMES-patched LAMMPS binary
as *both* the MD engine and the "QM" reference method (no DFT code needed)
— `codes/al_driver-LLfork/examples/simple_iter_single_statepoint-lmp-test/`.
Copy that example's `config.py` + `ALL_BASE_FILES/`, edit the config's
hardcoded paths/HPC settings for your machine (it ships configured for a
different cluster), and point `al-run` at it.

## Usage

```bash
chimes-agent al-run \
  --work-dir ./my_al_study \
  --config-py ./my_al_study/config.py \
  --cycles 0,1,2,3 \
  --output-dir ignored  # al-run does not use --output-dir; work_dir is the state directory
```

(`al-run` sets `USES_OUTPUT_DIR = False` — `work_dir` already is
al_driver's own state directory, so there's no separate manifest.)

This returns **immediately** with a PID and log path; it does not wait for
`main.py` to finish (it can run for hours to days, submitting and polling
its own Slurm jobs internally — matching al_driver's own documented
recommendation to run it under `screen`/`tmux`/`nohup`).

```bash
tail -f ./my_al_study/driver.log          # watch progress
chimes-agent al-run --status-of <pid>     # is it still running?
chimes-agent al-run --stop <pid>          # SIGTERM it
```

## Flags

- `--work-dir DIR` (required unless `--status-of`/`--stop`) — an
  al_driver study directory
- `--config-py PATH` — copied to `work_dir/config.py`; omit if
  `work_dir/config.py` already exists
- `--cycles 0,1,2,3` (default `[0]`) — ALC cycle indices, passed as
  `main.py`'s argv
- `--python-bin PATH` — override which Python runs `main.py` (default:
  the same interpreter running `chimes-agent`)
- `--status-of PID` — check if a previously launched run is still alive
- `--stop PID` — SIGTERM a previously launched run

## Output (launch)

```json
{
  "work_dir": "./my_al_study", "config_py": "./my_al_study/config.py",
  "cycles": [0, 1, 2, 3], "pid": 123456, "log": "./my_al_study/driver.log",
  "status": "launched", "command": "python3 .../main.py 0 1 2 3",
  "note": "main.py is now running detached in the background ..."
}
```
