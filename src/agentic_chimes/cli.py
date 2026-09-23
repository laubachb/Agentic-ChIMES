"""`chimes-agent`: the tool-calling CLI entry point.

Every stage is a subcommand with a uniform JSON contract: `--json-in FILE`
(or discrete flags) in, one JSON object on stdout (or `--json-out FILE`)
out, plus `--describe` to print the stage's schema/help without running it.
Stages that write files accept `--output-dir` and are idempotent there via a
manifest (agentic_chimes.stages._manifest) -- re-invoking with the same
inputs short-circuits instead of re-running; different inputs at the same
path require --force.

No stage chains into another here -- composing stages is the caller's
(human's or agent's) job. See docs/concepts/stages_and_contracts.md.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from .stages import _manifest

STAGE_MODULE_NAMES = [
    "setup_cmd",
    "dataset_select",
    "qe_relabel",
    "fm_setup_gen",
    "amat_build",
    "solve",
    "sweep",
    "evaluate",
    "lammps_run",
    "submit",
    "al_select",
]


def _load_stage_modules():
    mods = []
    for name in STAGE_MODULE_NAMES:
        mods.append(importlib.import_module(f"agentic_chimes.stages.{name}"))
    return mods


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chimes-agent",
        description=(
            "Tool-calling CLI for building ChIMES machine-learned interatomic "
            "potentials: one discrete, structured-I/O command per pipeline "
            "stage. Run `chimes-agent <stage> --describe` for a stage's full "
            "input/output contract."
        ),
    )
    sub = parser.add_subparsers(dest="stage", required=True)

    for mod in _load_stage_modules():
        p = sub.add_parser(mod.NAME, help=mod.SUMMARY, description=mod.SUMMARY)
        p.add_argument("--json-in", dest="json_in", default=None, help="Read stage input from this JSON file.")
        p.add_argument("--json-out", dest="json_out", default=None, help="Write stage output JSON here instead of stdout.")
        p.add_argument("--describe", action="store_true", help="Print this stage's schema/help and exit without running it.")
        p.add_argument("--force", action="store_true", help="Re-run even if a prior --output-dir run with matching inputs is already done.")
        p.add_argument("--dry-run", dest="dry_run", action="store_true", help="For HPC-submitting stages: render the sbatch script without submitting.")
        uses_output_dir = getattr(mod, "USES_OUTPUT_DIR", True)
        if uses_output_dir:
            p.add_argument("--output-dir", dest="output_dir", default=None, help="Directory for output files + manifest.json.")
        mod.add_arguments(p)
        p.set_defaults(_module=mod, _uses_output_dir=uses_output_dir)

    return parser


def _apply_json_in(args: argparse.Namespace) -> None:
    if not getattr(args, "json_in", None):
        return
    with open(args.json_in) as f:
        data = json.load(f)
    for key, value in data.items():
        setattr(args, key, value)


def _emit(result: dict, args: argparse.Namespace) -> None:
    text = json.dumps(result, indent=2, default=str)
    if getattr(args, "json_out", None):
        Path(args.json_out).write_text(text + "\n")
    else:
        print(text)


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    mod = args._module

    if args.describe:
        _emit(
            {
                "stage": mod.NAME,
                "summary": mod.SUMMARY,
                "uses_output_dir": args._uses_output_dir,
                "schema": getattr(mod, "SCHEMA", None),
            },
            args,
        )
        return 0

    _apply_json_in(args)

    _NON_INPUT_KEYS = ("json_in", "json_out", "describe", "stage", "force", "dry_run", "output_dir")
    input_echo = {k: v for k, v in vars(args).items() if not k.startswith("_") and k not in _NON_INPUT_KEYS}

    if args._uses_output_dir and args.output_dir:
        try:
            decision, prior = _manifest.begin(args.output_dir, mod.NAME, input_echo, force=args.force)
        except _manifest.InputMismatch as exc:
            _emit({"error": str(exc)}, args)
            return 1
        if decision == "short_circuit":
            _emit(prior["outputs"], args)
            return 0

    try:
        result = mod.run(args)
    except Exception as exc:  # noqa: BLE001 - surface as structured error, not a traceback the caller has to parse
        if args._uses_output_dir and args.output_dir:
            _manifest.finish(args.output_dir, mod.NAME, input_echo, {"error": str(exc)}, status="failed")
        _emit({"error": str(exc)}, args)
        return 1

    if args._uses_output_dir and args.output_dir:
        _manifest.finish(args.output_dir, mod.NAME, input_echo, result, status="done")

    _emit(result, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
