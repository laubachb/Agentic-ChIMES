"""Install an EZRA personal agent for driving chimes-agent.

EZRA agents (https://github.com/Awwal41/EZRA, docs/agents.md) are folders
`agents/<name>/` holding `agent.yaml` (preferred / fallback gallery models,
tags, science focus), `prompt.md` (the system prompt) and `NOTES.md`. This
stage follows EZRA's own steps - init from a template, pick models that fit
the machine, check the aliases - for the `chimes-assistant` profile shipped
in this package:

1. copy it to `<ezra-root>/agents/<name>/` (default `~/.chimes-agent/ezra`,
   handed to EZRA as EZRA_ROOT so your EZRA checkout is not modified);
2. re-pick `models.preferred` / `fallback` with llm-recommend's ranking for
   this host or `--for-machine`'s compute node;
3. check every alias against the gallery and that EZRA itself loads the
   profile (`ezra agents show`).

`llm-serve --agent <name>` then serves the first downloaded model from that
list; with the `ezra` engine it also applies the prompt server-side.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from . import _llm
from .llm_download import models_dir_for
from .llm_recommend import _target

NAME = "llm-agent"
SUMMARY = "Install an EZRA agent profile (prompt + models picked for this host or a profile's node) for driving chimes-agent."
USES_OUTPUT_DIR = False
SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "default": _llm.DEFAULT_AGENT, "description": "Agent name (EZRA naming: letters, digits, . _ -; optional local/ prefix)."},
        "template": {"type": ["string", "null"], "description": "Agent folder to copy (default: the shipped chimes-assistant)."},
        "ezra_root": {"type": ["string", "null"], "description": "EZRA_ROOT holding agents/ (default CHIMES_EZRA_ROOT or ~/.chimes-agent/ezra)."},
        "for_machine": {"type": ["string", "null"], "description": "Pick models for one compute node of this machine profile."},
        "models_dir": {"type": ["string", "null"], "description": "Where llm-download put weights (default: profile scratch with --for-machine)."},
        "memory_gb": {"type": ["number", "null"]},
        "gpus": {"type": ["integer", "null"]},
        "gpu_memory_gb": {"type": ["number", "null"]},
        "ctx_size": {"type": "integer", "default": _llm.DEFAULT_CTX},
        "keep_models": {"type": "boolean", "default": False, "description": "Keep the template's model lists instead of re-picking."},
        "replace": {"type": "boolean", "default": False, "description": "Overwrite an existing agent folder of that name."},
        "ezra": {"type": ["string", "null"]},
    },
}


def add_arguments(parser) -> None:
    parser.add_argument("--name", default=_llm.DEFAULT_AGENT)
    parser.add_argument("--template", default=None)
    parser.add_argument("--ezra-root", dest="ezra_root", default=None)
    parser.add_argument("--for-machine", dest="for_machine", default=None)
    parser.add_argument("--models-dir", dest="models_dir", default=None)
    parser.add_argument("--memory-gb", dest="memory_gb", type=float, default=None)
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--gpu-memory-gb", dest="gpu_memory_gb", type=float, default=None)
    parser.add_argument("--ctx-size", dest="ctx_size", type=int, default=_llm.DEFAULT_CTX)
    parser.add_argument("--keep-models", dest="keep_models", action="store_true")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--ezra", default=None)


def pick_models(ranked: list) -> tuple:
    """Two strongest fits as preferred; as fallback, the two strongest of the
    3-9B class (still useful when the big ones are not downloaded), else the
    next two in the ranking."""
    preferred = [r["name"] for r in ranked[:2]]
    rest = [r for r in ranked if r["name"] not in preferred]
    mid = [r["name"] for r in rest if 3.0 <= float(r.get("size_b") or 0) <= 9.0]
    fallback = (mid or [r["name"] for r in rest])[:2]
    return preferred, fallback


def run(args) -> dict:
    name = args.name.strip().strip("/")
    if not name or name.startswith("_") or ".." in name or name.count("/") > 1:
        raise ValueError(f"invalid agent name {args.name!r}: use e.g. 'chimes-assistant' or 'local/my-chimes'")
    template = Path(args.template).expanduser().resolve() if getattr(args, "template", None) else _llm.SHIPPED_AGENTS / _llm.DEFAULT_AGENT
    if not (template / "agent.yaml").is_file():
        raise FileNotFoundError(f"{template} has no agent.yaml")
    root = _llm.ezra_root(getattr(args, "ezra_root", None))
    dest = root / "agents" / name
    if dest.exists():
        if not getattr(args, "replace", False):
            raise FileExistsError(f"agent {name!r} already exists at {dest}; pass --replace to overwrite it")
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(template, dest)

    spec = yaml.safe_load((dest / "agent.yaml").read_text()) or {}
    spec["name"] = name.split("/")[-1]
    ezra = getattr(args, "ezra", None)
    gallery = _llm.run_ezra(["models", "list", "--json"], ezra=ezra, models_dir=models_dir_for(args) or None)
    hardware = None
    if not getattr(args, "keep_models", False):
        target = _target(args)
        budget = _llm.budget_gb(target["memory_gb"], target["gpus"], target["gpu_memory_gb"], target["context"])
        ranked = _llm.rank_for_agents(gallery, budget=budget, has_gpu=bool(target["gpus"]), ctx_size=args.ctx_size)
        if not ranked:
            raise ValueError(f"no gallery chat model fits {budget} GB with a {args.ctx_size}-token context; "
                             "lower --ctx-size or size a bigger node (--for-machine)")
        preferred, fallback = pick_models(ranked)
        spec.setdefault("models", {})
        spec["models"]["preferred"], spec["models"]["fallback"] = preferred, fallback
        hardware = {**target, "budget_gb": budget}
    header = (f"# EZRA agent installed by `chimes-agent llm-agent` from {template}.\n"
              "# Edit freely; re-run llm-agent --replace to re-pick models for other hardware.\n")
    (dest / "agent.yaml").write_text(header + yaml.safe_dump(spec, sort_keys=False))

    models = spec.get("models") or {}
    aliases = list(dict.fromkeys(list(models.get("preferred") or []) + list(models.get("fallback") or [])))
    known_names = {m.get("name") for m in gallery}
    installed = {m.get("name") for m in gallery if m.get("installed")}
    unknown = [a for a in aliases if a not in known_names]
    try:
        _llm.run_ezra(["agents", "show", name], ezra=ezra, ezra_root=str(root), parse=False)
        load_error = None
    except RuntimeError as exc:
        load_error = str(exc)
    missing = [a for a in aliases if a in known_names and a not in installed]

    return {
        "agent": name,
        "path": str(dest),
        "ezra_root": str(root),
        "prompt": str(dest / (spec.get("prompt") or "prompt.md")),
        "preferred": list(models.get("preferred") or []),
        "fallback": list(models.get("fallback") or []),
        "hardware": hardware,
        "check": {"ok": not unknown and load_error is None, "unknown_aliases": unknown,
                  "not_downloaded": missing, "ezra_load_error": load_error},
        "next": ([f"chimes-agent llm-download --model {missing[0]}"] if missing else [])
                + [f"chimes-agent llm-serve --agent {name} --work-dir <dir>"],
        "note": (f"Tell EZRA where this agent lives with `export EZRA_ROOT={root}` before running "
                 f"`ezra chat --agent {name}` yourself; llm-serve sets it for you."),
    }
