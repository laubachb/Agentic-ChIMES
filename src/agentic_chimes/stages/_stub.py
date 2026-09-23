"""Shared scaffolding for stages not yet implemented (see the phasing plan in
the top-level README). Each stub still registers a real subcommand with a
real --describe summary and echoes its parsed input as JSON, so the CLI
surface and JSON-in/out ergonomics are validated end to end before a
stage's real logic lands -- swap `run` for real logic without changing the
subcommand's shape.
"""

from __future__ import annotations


def echo_run(args) -> dict:
    return {
        "status": "not_implemented",
        "echo": {k: v for k, v in vars(args).items() if not k.startswith("_")},
    }


def add_json_passthrough(parser) -> None:
    parser.add_argument(
        "--param",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Placeholder parameter (repeatable) until this stage is implemented; prefer --json-in for structured input.",
    )
