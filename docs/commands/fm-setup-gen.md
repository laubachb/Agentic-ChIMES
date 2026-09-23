# `chimes-agent fm-setup-gen`

**Status: implemented.**

Generates a `fm_setup.in` file from typed, high-level parameters instead of
hand-authoring the raw ChIMES grammar. See `src/agentic_chimes/io/fm_setup.py`
for the grammar (parser + renderer, round-trip tested against the golden
fixtures in `codes/chimes_lsq-LLfork/test_suite-lsq/`).

## Usage

```bash
chimes-agent fm-setup-gen \
  --trjfile /abs/path/to/training.xyzf --nframes 250 \
  --elements C,H \
  --order '{"2":12,"3":5,"4":4}' \
  --pair-cutoffs '{"C-C":[1.29,5.0],"C-H":[1.29,5.0],"H-H":[0.9,5.0]}' \
  --morse-lambda '{"C-C":1.54,"C-H":1.54,"H-H":1.54}' \
  --masses '{"C":12.011,"H":1.008}' \
  --fitener false --fitstrs false \
  --output-dir ./run1
```

Or via `--json-in` for the full schema (`--describe` to print it), including
`exclude_3b`/`exclude_4b` lists which have no flat-flag equivalent.

## Key flags

| Flag | Meaning |
|---|---|
| `--trjfile` | Written verbatim as `TRJFILE`; use an **absolute path** — `chimes_lsq` resolves it relative to its own working directory, not the fm_setup.in file's location |
| `--nframes` | `NFRAMES` |
| `--elements` | comma-separated, defines atom-type table order |
| `--order` | JSON `{"2": N, "3": N, "4": N}` — `"4"` is optional; omitting it renders `PAIRTYP` without a 4-body order or Chebyshev range, matching upstream's own convention (see `special3b/fm_setup.in`) |
| `--pair-cutoffs`, `--morse-lambda` | JSON, keyed `"El1-El2"` (order-insensitive lookup); unlisted pairs fall back to `--default-s-minim`/`--default-s-maxim`/`--default-morse-lambda` |
| `--fitener`, `--fitstrs` | pass through as raw strings (`false`, `true`, `ALL`, `FIRST <n>`) per upstream's grammar |

## Output

```json
{
  "fm_setup_in": "./run1/fm_setup.in",
  "params": { "...": "the fully-resolved parameter dict, including auto-generated pair rows" }
}
```

Feed `fm_setup_in` directly into `chimes-agent amat-build`.
