"""Grid sweep over lambda/order/cutoffs, composing fm-setup-gen + amat-build
+ solve + evaluate and reporting a comparison table -- not auto-tuning; the
user picks from the table. Planned for Phase 3 -- see README phasing table.
Not yet implemented; this stub validates the CLI contract."""

from __future__ import annotations

from ._stub import add_json_passthrough, echo_run

NAME = "sweep"
SUMMARY = "Grid sweep over hyperparameters, reports a comparison table (not auto-tuning). [Phase 3 - not yet implemented]"
SCHEMA = {
    "type": "object",
    "properties": {
        "base_fm_setup": {"type": "string"},
        "grid": {"type": "object", "description": 'e.g. {"alpha": [1e-6,1e-5,1e-4], "order": [...]}'},
        "holdout_xyzf": {"type": "string"},
        "machine": {"type": ["string", "null"]},
        "max_parallel_jobs": {"type": "integer"},
    },
}


def add_arguments(parser) -> None:
    add_json_passthrough(parser)


def run(args) -> dict:
    return echo_run(args)
