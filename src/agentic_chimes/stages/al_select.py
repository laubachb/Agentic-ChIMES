"""Diversity-based active-learning batch selection, wrapping al_driver's
existing Metropolis-MC energy-histogram selector
(codes/al_driver-LLfork/src/gen_selections.py:gen_subset). This is coverage
selection, not uncertainty/query-by-committee -- al_driver's own
committeeALmode.rst doc stub confirms that mode was never finished upstream.
The concrete future plug-in point for uncertainty-based selection is
evaluate's `committee_spread` output (multiple params.txt models diffed on
the same frames), deliberately not wired up here.

Planned for Phase 5 -- see README phasing table. Not yet implemented; this
stub validates the CLI contract."""

from __future__ import annotations

from ._stub import add_json_passthrough, echo_run

NAME = "al-select"
SUMMARY = "Diversity (energy-histogram) active-learning batch selection. [Phase 5 - not yet implemented]"
SCHEMA = {
    "type": "object",
    "properties": {
        "candidate_frames": {"type": "string"},
        "params": {"type": "string"},
        "central_repo": {"type": "string"},
        "n_select": {"type": "integer"},
        "histogram_bins": {"type": "integer"},
    },
}


def add_arguments(parser) -> None:
    add_json_passthrough(parser)


def run(args) -> dict:
    return echo_run(args)
