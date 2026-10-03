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
import contextlib
import importlib
import json
import os
import sys
import tempfile
from pathlib import Path

from .stages import _manifest
from .io import fs

STAGE_MODULE_NAMES = [
    "setup_cmd",
    "doctor",
    "study",
    "usage",
    "data_search",
    "data_fetch",
    "data_generate",
    "data_curate",
    "fingerprint",
    "dataset_select",
    "qe_relabel",
    "hyper_analyze",
    "hyper_search",
    "fm_setup_gen",
    "amat_build",
    "solve",
    "weights",
    "hierarch",
    "model_build",
    "sweep",
    "auto_build",
    "evaluate",
    "lammps_run",
    "md_check",
    "benchmark",
    "deploy",
    "study_report",
    "submit",
    "job_status",
    "al_select",
    "al_merge",
    "al_run",
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


_JSON_TYPES = {"string": str, "number": (int, float), "integer": int, "boolean": bool, "array": list, "object": dict,
               "null": type(None)}


def _json_type_ok(value, name: str) -> bool:
    if name not in _JSON_TYPES:
        return True
    if isinstance(value, bool) and name != "boolean":  # bool is an int subclass in Python
        return False
    return isinstance(value, _JSON_TYPES[name])


def _apply_json_in(args: argparse.Namespace, schema: dict | None = None) -> None:
    """Merge a --json-in file into args. Unknown keys (usually typos) and
    values of the wrong JSON type are refused, so a setting can never be
    silently ignored."""
    if not getattr(args, "json_in", None):
        return
    with open(args.json_in) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"--json-in {args.json_in}: expected a JSON object, got {type(data).__name__}")
    props = (schema or {}).get("properties", {})
    known = {k for k in vars(args) if not k.startswith("_")} | set(props)
    unknown = sorted(k.replace("-", "_") for k in data if k.replace("-", "_") not in known)
    if unknown:
        raise ValueError(f"--json-in {args.json_in}: unknown key(s) {unknown}; valid keys: {sorted(known - {'json_in', 'json_out', 'describe', 'stage'})}")
    for key, value in data.items():
        key = key.replace("-", "_")
        declared = props.get(key, {}).get("type")
        if declared is not None and value is not None:  # null = "use the default", like omitting the key
            allowed = declared if isinstance(declared, list) else [declared]
            ok = any(_json_type_ok(value, name) for name in allowed)
            if not ok:
                raise ValueError(f"--json-in {args.json_in}: {key!r} must be {' or '.join(allowed)}, got {type(value).__name__} ({value!r})")
        setattr(args, key, value)


def _emit(result: dict, args: argparse.Namespace) -> None:
    text = json.dumps(result, indent=2, default=str)
    if getattr(args, "json_out", None):
        Path(args.json_out).write_text(text + "\n")
    else:
        print(text)


@contextlib.contextmanager
def _stage_output_to_log(log_path: Path):
    """Route everything a stage writes to fd 1 into `log_path`, so stdout
    stays reserved for the one JSON result object. Works at the file-
    descriptor level on purpose: chimes_calculator's C++ library and
    al_driver's `print`s write to stdout directly (~1,600 lines of banner/
    progress for one al-select call), which would otherwise land ahead of
    the JSON and make it unparseable for an agent."""
    sys.stdout.flush()
    saved_fd = os.dup(1)
    log_fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.dup2(log_fd, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved_fd, 1)
        os.close(saved_fd)
        os.close(log_fd)


def _log_tail(log_path: Path, n: int = 20) -> str:
    try:
        return "\n".join(log_path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def _cpu_snapshot():
    import resource
    import time

    s, c = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return time.time(), s.ru_utime + s.ru_stime + c.ru_utime + c.ru_stime


def _record_local_usage(args, stage_name: str, t0: float, cpu0: float) -> None:
    """Append this stage's login-node CPU time to <study>/usage/local.jsonl when
    it wrote inside a study. Skipped inside Slurm jobs (sacct accounts for
    those) and for sub-second runs; never allowed to fail the stage."""
    try:
        if os.environ.get("SLURM_JOB_ID") or not getattr(args, "output_dir", None):
            return
        from .stages import study as study_stage

        root = study_stage.find_study(args.output_dir)
        if root is None:
            return
        t1, cpu1 = _cpu_snapshot()
        if cpu1 - cpu0 < 1.0:
            return
        import datetime

        rec = {"stage": stage_name, "output_dir": str(Path(args.output_dir).resolve()), "wall_s": round(t1 - t0, 2),
               "cpu_s": round(cpu1 - cpu0, 2), "host": os.uname().nodename,
               "at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
        ledger = root / "usage" / "local.jsonl"
        ledger.parent.mkdir(exist_ok=True)
        with open(ledger, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:  # noqa: BLE001 - accounting must never break a stage
        pass


def _stage_log_path(args, stage_name: str) -> Path:
    if args._uses_output_dir and args.output_dir:
        out = Path(args.output_dir)
        fs.ensure_dir(out)
        return out / f"{stage_name}.log"
    fd, path = tempfile.mkstemp(prefix=f"chimes-agent-{stage_name}-", suffix=".log")
    os.close(fd)
    return Path(path)


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

    try:
        _apply_json_in(args, getattr(mod, "SCHEMA", None))
    except (ValueError, json.JSONDecodeError, OSError) as exc:
        _emit({"error": str(exc)}, args)
        return 1
    if getattr(args, "dry_run", False) and not getattr(mod, "SUPPORTS_DRY_RUN", False):
        _emit({"error": f"{mod.NAME} has no --dry-run mode (it submits nothing); run it without --dry-run, "
                        "or use --describe to see its inputs"}, args)
        return 1

    _NON_INPUT_KEYS = ("json_in", "json_out", "describe", "stage", "force", "dry_run", "output_dir")
    input_echo = {k: v for k, v in vars(args).items() if not k.startswith("_") and k not in _NON_INPUT_KEYS}

    # A dry run records nothing: otherwise the real run that follows, with
    # identical inputs, would short-circuit to the dry run's result.
    use_manifest = bool(args._uses_output_dir and args.output_dir and not getattr(args, "dry_run", False))

    if use_manifest:
        try:
            decision, prior = _manifest.begin(args.output_dir, mod.NAME, input_echo, force=args.force)
        except _manifest.InputMismatch as exc:
            _emit({"error": str(exc)}, args)
            return 1
        except Exception as exc:  # noqa: BLE001 - e.g. a filesystem error creating the output dir: still one JSON object
            _emit({"error": f"could not prepare {args.output_dir}: {type(exc).__name__}: {exc}"}, args)
            return 1
        if decision == "short_circuit":
            _emit(prior["outputs"], args)
            return 0

    log_path = _stage_log_path(args, mod.NAME)
    try:
        t0, cpu0 = _cpu_snapshot()
        with _stage_output_to_log(log_path):
            result = mod.run(args)
        _record_local_usage(args, mod.NAME, t0, cpu0)
    except Exception as exc:  # noqa: BLE001 - surface as structured error, not a traceback the caller has to parse
        if use_manifest:
            _manifest.finish(args.output_dir, mod.NAME, input_echo, {"error": str(exc)}, status="failed")
        error = {"error": str(exc), "log": str(log_path)}
        tail = _log_tail(log_path)
        if tail:
            error["log_tail"] = tail
        _emit(error, args)
        return 1

    if use_manifest:
        _manifest.finish(args.output_dir, mod.NAME, input_echo, result, status="done")

    emitted = dict(result)
    if log_path.exists() and log_path.stat().st_size > 0:
        emitted["stage_log"] = str(log_path)
    else:
        log_path.unlink(missing_ok=True)
    _emit(emitted, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
