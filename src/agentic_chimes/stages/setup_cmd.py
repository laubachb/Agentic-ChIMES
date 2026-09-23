"""`chimes-agent setup` -- build/fetch chimes_lsq, chimes_calculator, LAMMPS,
and Quantum ESPRESSO for a given machine. See agentic_chimes.setup.bootstrap.
"""

from __future__ import annotations

from .. import setup as _setup
from ..machines import available_profiles

NAME = "setup"
SUMMARY = "Build/fetch chimes_lsq, chimes_calculator, LAMMPS, and Quantum ESPRESSO for a machine."
USES_OUTPUT_DIR = False

SCHEMA = {
    "type": "object",
    "properties": {
        "component": {
            "type": "array",
            "items": {"type": "string", "enum": _setup.ALL_COMPONENTS + ["all"]},
        },
        "machine": {"type": "string"},
        "force": {"type": "boolean"},
        "qe_version": {"type": ["string", "null"]},
        "status": {"type": "boolean"},
    },
}


def add_arguments(parser) -> None:
    parser.add_argument(
        "--component",
        action="append",
        choices=_setup.ALL_COMPONENTS + ["all"],
        help="Which component to build (repeatable). Default: all.",
    )
    parser.add_argument(
        "--machine",
        choices=available_profiles(),
        help="Machine profile to build for (required unless --status).",
    )
    parser.add_argument("--qe-version", dest="qe_version", default=None, help="Override the pinned QE release tag.")
    parser.add_argument(
        "--status",
        action="store_true",
        help="Report what's already installed (from deps/installed.json) and exit; no build.",
    )


def run(args) -> dict:
    if getattr(args, "status", False):
        return {"installed": _setup.status()}

    components = args.component or ["all"]
    if not args.machine and components != ["codes"]:
        raise SystemExit(
            "chimes-agent setup: --machine is required unless --status is given, or "
            "--component is exactly 'codes' (cloning the vendored forks needs no machine profile)"
        )
    results = _setup.run_setup(
        components,
        machine=args.machine,
        force=bool(getattr(args, "force", False)),
        qe_version=getattr(args, "qe_version", None),
    )
    return {"machine": args.machine, "results": results}
