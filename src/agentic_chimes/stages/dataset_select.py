"""FPS / holdout-split dataset sampling. Planned for Phase 3 -- see README
phasing table. Not yet implemented; this stub validates the CLI contract."""

from __future__ import annotations

from ._stub import add_json_passthrough, echo_run

NAME = "dataset-select"
SUMMARY = "FPS / holdout-split sampling over a frame pool. [Phase 3 - not yet implemented]"
SCHEMA = {
    "type": "object",
    "properties": {
        "frames": {"type": "string", "description": "Path to packed .xyzf/.npz frame pool."},
        "method": {"type": "string", "enum": ["fps", "random", "stratified_holdout"]},
        "n_select": {"type": "integer"},
        "holdout_fraction": {"type": "number"},
        "seed": {"type": "integer"},
    },
}


def add_arguments(parser) -> None:
    add_json_passthrough(parser)


def run(args) -> dict:
    return echo_run(args)
