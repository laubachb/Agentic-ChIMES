"""Single-point/MD via the ChIMES-patched `lmp_mpi_chimes` build (see
`chimes-agent setup --component lammps`). Planned for Phase 3 -- see README
phasing table. Not yet implemented; this stub validates the CLI contract."""

from __future__ import annotations

from ._stub import add_json_passthrough, echo_run

NAME = "lammps-run"
SUMMARY = "Single-point/MD via lmp_mpi_chimes. [Phase 3 - not yet implemented]"
SCHEMA = {
    "type": "object",
    "properties": {
        "params": {"type": "string"},
        "structure": {"type": "string"},
        "mode": {"type": "string", "enum": ["single_point", "md"]},
        "md_settings": {"type": "object"},
        "machine": {"type": ["string", "null"]},
        "queue": {"type": "string"},
        "walltime_hours": {"type": "number"},
    },
}


def add_arguments(parser) -> None:
    add_json_passthrough(parser)


def run(args) -> dict:
    return echo_run(args)
