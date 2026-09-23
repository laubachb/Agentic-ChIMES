# `chimes-agent amat-build`

**Status: implemented (local only).** `--hpc`/`SPLITFI` support for large
DLARS-bound runs lands alongside `solve`'s HPC path.

Builds `A.txt`/`b.txt`/`dim.txt` (and friends) by subprocessing the
`chimes_lsq` C++ binary against an `fm_setup.in` whose `TRJFILE` already
points at a labeled `.xyzf`. The binary's CLI is exactly `chimes_lsq
<fm_setup.in>` (confirmed against `chimes_lsq.C`'s `argc != 2` check); all
output lands in the process's working directory.

## Usage

```bash
chimes-agent amat-build --fm-setup-in ./run1/fm_setup.in --output-dir ./run1
```

If `--output-dir` is omitted, output lands next to the `fm_setup.in` file.

## Flags

- `--fm-setup-in PATH` (required)
- `--chimes-lsq-bin PATH` — override the resolved `chimes_lsq` binary
  (default: resolved from `deps/installed.json` / `AGENTIC_CHIMES_LSQ_BIN`,
  see `chimes-agent setup`)

## Output

```json
{
  "work_dir": "./run1",
  "log": "./run1/fm_setup.log",
  "split": false,
  "missing_outputs": [],
  "A": "./run1/A.txt",
  "b": "./run1/b.txt",
  "b_labeled": "./run1/b-labeled.txt",
  "dim": "./run1/dim.txt",
  "natoms": "./run1/natoms.txt",
  "params_header": "./run1/params.header",
  "ff_groups_map": "./run1/ff_groups.map"
}
```

`"split": true` means `fm_setup.in` had `SPLITFI true` and output landed as
`A.0000.txt`/`b.0000.txt`/... instead (required for the DLARS solve path);
in that case the named keys above won't all be present.
