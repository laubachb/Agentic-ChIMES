# `chimes-agent model-build`

**Status: implemented.**

Complete ChIMES model build: `amat-build` then `solve`, sequentially, in
one call — composes those two stages' `run()` functions directly (see
`stages/_compose.py`) rather than requiring two separate invocations for
the common case of "I have an `fm_setup.in`, give me a `params.txt`."

Does not include `fm-setup-gen` (call that separately first) or `evaluate`
(holdout scoring is a deliberately separate decision, not bundled into
every build).

## Usage

```bash
chimes-agent model-build \
  --fm-setup-in ./run1/fm_setup.in \
  --algorithm lassolars --alpha 1e-5 \
  --output-dir ./run1
```

Same local-only scope as `amat-build`/`solve` today — `--algorithm
dlars`/`dlasso` isn't wired up yet (raises a clear error), and a
`SPLITFI true` `fm_setup.in` (needed for the DLARS path on a large basis)
also isn't supported here yet: run `amat-build` and `solve` separately in
that case.

## Flags

Union of `amat-build` and `solve`'s flags: `--fm-setup-in`,
`--chimes-lsq-bin`, `--algorithm`, `--alpha`, `--eps`, `--weights`,
`--folds`.

## Output

```json
{
  "work_dir": "./run1",
  "amat_build": { "...": "amat-build's full output" },
  "solve": { "...": "solve's full output" },
  "params": "./run1/params.txt"
}
```
