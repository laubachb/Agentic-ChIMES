# `chimes-agent lammps-run`

**Status: planned (Phase 3), not yet implemented.** The subcommand exists
today as a stub (`stages/lammps_run.py`) that echoes its parsed input —
`--describe` works, the CLI contract below is fixed.

`chimes-agent setup --component lammps` (already implemented) builds the
ChIMES-patched `lmp_mpi_chimes` binary this stage will subprocess (symlinked
at `deps/lammps-chimes/`) — see `docs/commands/setup.md`. Note this is a
from-source, patched-core LAMMPS build (not a stock LAMMPS + plugin), so
this stage will always run the `deps/lammps-chimes` binary, never a
site-installed LAMMPS module.

## Planned schema

```json
{
  "params": "path to params.txt",
  "structure": "path to a LAMMPS data file",
  "mode": "single_point | md",
  "md_settings": {"...": "..."},
  "machine": "dane",
  "queue": "batch",
  "walltime_hours": 2
}
```
