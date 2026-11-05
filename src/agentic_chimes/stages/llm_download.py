"""Download open-weight model(s) from EZRA's gallery.

Runs where it is called, normally a login node: many clusters give compute
nodes no internet access. With `--for-machine`, weights go under the
profile's shared scratch (`<scratch_root>/ezra/models`) so a batch job can
read them; node-local directories are refused there. Gated models need
`HF_TOKEN` and an accepted license on Hugging Face.
"""

from __future__ import annotations

from pathlib import Path

from .. import machines
from ..hpc import slurm
from . import _llm

NAME = "llm-download"
SUMMARY = "Download open-weight LLM weights via EZRA onto shared storage (login node; --dry-run lists what would be fetched)."
USES_OUTPUT_DIR = False
SUPPORTS_DRY_RUN = True
SCHEMA = {
    "type": "object",
    "required": ["models"],
    "properties": {
        "models": {"type": "array", "items": {"type": "string"}, "description": "EZRA gallery names (see llm-recommend)."},
        "models_dir": {"type": ["string", "null"], "description": "Where weights go (EZRA_MODELS_DIR). Default: <profile scratch_root>/ezra/models with --for-machine, else EZRA's own default."},
        "for_machine": {"type": ["string", "null"], "description": "Machine profile whose shared scratch holds the weights."},
        "quant": {"type": ["string", "null"], "description": "GGUF quantization to prefer, e.g. Q4_K_M."},
        "ezra": {"type": ["string", "null"], "description": "Path to the ezra executable (default: CHIMES_EZRA or PATH)."},
    },
}


def add_arguments(parser) -> None:
    parser.add_argument("--model", dest="models", action="append", default=None, help="Gallery name (repeatable).")
    parser.add_argument("--models-dir", dest="models_dir", default=None)
    parser.add_argument("--for-machine", dest="for_machine", default=None)
    parser.add_argument("--quant", default=None)
    parser.add_argument("--ezra", default=None)


def models_dir_for(args) -> str:
    if getattr(args, "models_dir", None):
        return str(Path(args.models_dir).expanduser())
    if getattr(args, "for_machine", None):
        scratch = machines.load_profile(args.for_machine).filesystem.get("scratch_root")
        if not scratch:
            raise ValueError(f"profile {args.for_machine!r} has no filesystem.scratch_root; pass --models-dir")
        return str(Path(scratch) / "ezra" / "models")
    return ""


def run(args) -> dict:
    if not getattr(args, "models", None):
        raise ValueError("llm-download needs at least one --model (an EZRA gallery name; see llm-recommend)")
    models_dir = models_dir_for(args)
    if models_dir and getattr(args, "for_machine", None):
        slurm.check_shared_dir(models_dir)
    if models_dir:
        Path(models_dir).mkdir(parents=True, exist_ok=True)

    cmd = ["models", "download", *args.models, "--json"]
    if getattr(args, "quant", None):
        cmd += ["--quant", args.quant]
    if getattr(args, "dry_run", False):
        cmd.append("--dry-run")
    ezra = getattr(args, "ezra", None)
    results = _llm.run_ezra(cmd, ezra=ezra, models_dir=models_dir or None, check=False)

    installed = {}
    for name in args.models:
        try:
            info = _llm.run_ezra(["models", "info", name, "--json"], ezra=ezra, models_dir=models_dir or None)
        except RuntimeError as exc:
            installed[name] = {"installed": False, "error": str(exc)}
            continue
        installed[name] = {k: info.get(k) for k in ("installed", "path", "format", "bytes_on_disk", "license", "gated")}

    failed = [r for r in results if r.get("status") == "failed"]
    out = {
        "models_dir": models_dir or "(EZRA default)",
        "dry_run": bool(getattr(args, "dry_run", False)),
        "results": results,
        "models": installed,
    }
    if failed:
        out["error"] = "; ".join(f"{r.get('name')}: {r.get('message')}" for r in failed)
        if any("gated" in (r.get("message") or "").lower() or "401" in (r.get("message") or "") for r in failed):
            out["fix"] = "Gated model: set HF_TOKEN and accept the license on its Hugging Face page."
        raise RuntimeError(out["error"] + (f" ({out['fix']})" if "fix" in out else ""))
    return out
