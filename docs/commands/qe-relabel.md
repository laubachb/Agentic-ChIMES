# `chimes-agent qe-relabel`

**Status: planned (Phase 4), not yet implemented.** The subcommand exists
today as a stub (`stages/qe_relabel.py`) that echoes its parsed input —
`--describe` works, the CLI contract below is fixed. See
`docs/concepts/qm_driver_plugins.md` for the full design (the QE driver,
the QM-driver registry it plugs into, and the `.xyzf` output contract it
must match).

`chimes-agent setup --component quantum_espresso` (already implemented)
only fetches and builds `pw.x` — it does not run or parse QM jobs; that's
this stage's job, once built.

## Planned schema

```json
{
  "frames": "path to unlabeled configs",
  "pseudopotentials": {"C": "/path/to/C.upf"},
  "ecutwfc": 60,
  "ecutrho": 480,
  "kspacing": 0.3,
  "smearing": "gaussian",
  "machine": "dane",
  "queue": "batch",
  "walltime_hours": 4,
  "collect": "path to a completed job dir, instead of submitting"
}
```
