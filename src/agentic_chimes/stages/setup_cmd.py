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
        "ref": {"type": ["array", "null"], "items": {"type": "string"},
                "description": "REPO=COMMIT (repeatable): check a vendored fork out at this commit, branch or tag instead of its pin, for this call."},
        "init_profile": {"type": ["string", "null"], "description": "Write a machine profile YAML at this path (partitions, cores and accounts detected from Slurm) and exit."},
        "account": {"type": ["string", "null"], "description": "--init-profile: fallback Slurm account (CHIMES_ACCOUNT still wins)."},
        "debug_partition": {"type": ["string", "null"]},
        "batch_partition": {"type": ["string", "null"]},
        "cores_per_node": {"type": ["integer", "null"]},
        "scratch": {"type": ["string", "null"], "description": "--init-profile: shared scratch directory for job I/O."},
        "modules": {"type": ["array", "null"], "items": {"type": "string"}, "description": "--init-profile: modules to load in jobs and builds."},
        "hosttype": {"type": ["string", "null"], "description": "--init-profile: the forks' modfile name (codes/*/modfiles/<hosttype>.mod); default none."},
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
        help=f"Machine profile to build for: a built-in name ({', '.join(available_profiles())}) or a path to "
        "your own profile YAML (required unless --status).",
    )
    parser.add_argument("--qe-version", dest="qe_version", default=None, help="Override the pinned QE release tag.")
    parser.add_argument(
        "--status",
        action="store_true",
        help="Report what's already installed (from deps/installed.json) and exit; no build.",
    )
    parser.add_argument("--ref", action="append", default=None,
                        help="REPO=COMMIT (repeatable): check a vendored fork out at this ref instead of its pin.")
    parser.add_argument("--init-profile", dest="init_profile", default=None,
                        help="Write a machine profile YAML here (detected from Slurm) and exit.")
    parser.add_argument("--account", default=None)
    parser.add_argument("--debug-partition", dest="debug_partition", default=None)
    parser.add_argument("--batch-partition", dest="batch_partition", default=None)
    parser.add_argument("--cores-per-node", dest="cores_per_node", type=int, default=None)
    parser.add_argument("--scratch", default=None)
    parser.add_argument("--modules", type=lambda s: [x for x in s.split(",") if x], default=None)
    parser.add_argument("--hosttype", default=None)


def run(args) -> dict:
    if getattr(args, "status", False):
        return {"installed": _setup.status()}
    if getattr(args, "init_profile", None):
        from ..setup import init_profile

        return init_profile.write(args.init_profile, account=getattr(args, "account", None),
                                  debug_partition=getattr(args, "debug_partition", None),
                                  batch_partition=getattr(args, "batch_partition", None),
                                  cores_per_node=getattr(args, "cores_per_node", None), scratch=getattr(args, "scratch", None),
                                  modules=getattr(args, "modules", None), hosttype=getattr(args, "hosttype", None),
                                  overwrite=bool(getattr(args, "force", False)))
    from ..setup import clone_codes

    refs = clone_codes.parse_refs(getattr(args, "ref", None))

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
        refs=refs,
    )
    return {"machine": args.machine, "results": results}
