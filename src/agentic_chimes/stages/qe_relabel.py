"""Submit Quantum ESPRESSO single-point/relax jobs and (--collect) convert
finished output to .xyzf via converters.qe2xyzf. Planned for Phase 4, after
the QM-driver plugin registry lands -- see README phasing table. Not yet
implemented; this stub validates the CLI contract."""

from __future__ import annotations

from ._stub import add_json_passthrough, echo_run

NAME = "qe-relabel"
SUMMARY = "Submit QE single-point/relax jobs; --collect converts output to .xyzf. [Phase 4 - not yet implemented]"
SCHEMA = {
    "type": "object",
    "properties": {
        "frames": {"type": "string"},
        "pseudopotentials": {"type": "object", "additionalProperties": {"type": "string"}},
        "ecutwfc": {"type": "number"},
        "ecutrho": {"type": "number"},
        "kspacing": {"type": "number"},
        "smearing": {"type": "string"},
        "machine": {"type": "string"},
        "queue": {"type": "string"},
        "walltime_hours": {"type": "number"},
        "collect": {"type": "string", "description": "Path to a completed job dir to collect instead of submitting."},
    },
}


def add_arguments(parser) -> None:
    add_json_passthrough(parser)


def run(args) -> dict:
    return echo_run(args)
