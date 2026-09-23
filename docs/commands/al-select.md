# `chimes-agent al-select`

**Status: planned (Phase 5), not yet implemented.** The subcommand exists
today as a stub (`stages/al_select.py`) that echoes its parsed input —
`--describe` works, the CLI contract below is fixed.

## Intent

Diversity-based active-learning batch selection, wrapping al_driver's
existing Metropolis-MC energy-histogram selector
(`codes/al_driver-LLfork/src/gen_selections.py:gen_subset`). **This is
coverage/diversity selection, not uncertainty/query-by-committee** —
al_driver's own `doc/source/files_to_finish/committeeALmode.rst` is an
empty stub confirming committee-based AL was never finished upstream, and
this repo does not attempt to finish it here either.

The concrete future plug-in point for uncertainty-based selection is
`chimes-agent evaluate`'s `committee_spread` output (multiple `params.txt`
models diffed on the same unlabeled candidate frames, picking the
highest-disagreement ones) — see `docs/commands/evaluate.md`. Deliberately
not wired up in this stage yet.

## Planned schema

```json
{
  "candidate_frames": "path",
  "params": "path (for ChIMES-predicted energies)",
  "central_repo": "path",
  "n_select": 50,
  "histogram_bins": 20
}
```
