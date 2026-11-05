"""Which open-weight LLM can drive the agents on this hardware.

Probes the host through EZRA (RAM, NVIDIA GPUs), or takes a compute node's
resources from a machine profile's `node:` block (`--for-machine`) or flags,
and ranks EZRA's gallery chat models by whether weights plus a full agent
context fit. A login node is capped at a small budget whatever it has: it is
shared. Submits nothing.
"""

from __future__ import annotations

from .. import machines
from . import _llm

NAME = "llm-recommend"
SUMMARY = "Hardware check + open-weight chat models (EZRA gallery) that fit this host or a profile's compute node."
USES_OUTPUT_DIR = False
SCHEMA = {
    "type": "object",
    "properties": {
        "keyword": {"type": ["string", "null"], "description": "Only models whose name, tags or description contain this."},
        "for_machine": {"type": ["string", "null"], "description": "Size for one compute node of this machine profile instead of this host."},
        "memory_gb": {"type": ["number", "null"], "description": "Node RAM in GB (overrides the probe / profile)."},
        "gpus": {"type": ["integer", "null"], "description": "GPUs per node (overrides the probe / profile)."},
        "gpu_memory_gb": {"type": ["number", "null"], "description": "VRAM per GPU in GB (overrides the probe / profile)."},
        "ctx_size": {"type": "integer", "default": _llm.DEFAULT_CTX, "description": "Context the agent needs, in tokens; its KV cache counts against memory."},
        "limit": {"type": "integer", "default": 5},
        "include_gated": {"type": "boolean", "default": False, "description": "Include models that need an accepted Hugging Face license."},
        "ezra": {"type": ["string", "null"], "description": "Path to the ezra executable (default: CHIMES_EZRA or PATH)."},
    },
}


def add_arguments(parser) -> None:
    parser.add_argument("--keyword", default=None)
    parser.add_argument("--for-machine", dest="for_machine", default=None)
    parser.add_argument("--memory-gb", dest="memory_gb", type=float, default=None)
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--gpu-memory-gb", dest="gpu_memory_gb", type=float, default=None)
    parser.add_argument("--ctx-size", dest="ctx_size", type=int, default=_llm.DEFAULT_CTX)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--include-gated", dest="include_gated", action="store_true")
    parser.add_argument("--ezra", default=None)


def _target(args) -> dict:
    overrides = {k: getattr(args, k, None) for k in ("memory_gb", "gpus", "gpu_memory_gb")}
    if getattr(args, "for_machine", None):
        profile = machines.load_profile(args.for_machine)
        node = dict(profile.node)
        if not node.get("memory_gb") and overrides["memory_gb"] is None:
            raise ValueError(f"machine profile {profile.name!r} has no node.memory_gb; add a `node:` block or pass --memory-gb")
        node.update({k: v for k, v in overrides.items() if v is not None})
        return {"context": "compute-node", "machine": profile.name, "memory_gb": float(node.get("memory_gb") or 0),
                "cores": node.get("cores") or profile.default_ntasks_per_node, "gpus": int(node.get("gpus") or 0),
                "gpu_memory_gb": float(node.get("gpu_memory_gb") or 0), "gpu_names": []}
    system = _llm.run_ezra(["models", "recommend", "--json", "--limit", "1"], ezra=getattr(args, "ezra", None))["system"]
    gpus = system.get("gpus") or []
    target = {"context": _llm.host_context(), "machine": None, "memory_gb": float(system.get("ram_total_gb") or 0),
              "cores": system.get("cpu_count"), "gpus": len(gpus),
              "gpu_memory_gb": min((g.get("vram_gb") or 0 for g in gpus), default=0.0),
              "gpu_names": [g.get("name") for g in gpus]}
    for k, v in overrides.items():
        if v is not None:
            target[k] = v
    return target


def run(args) -> dict:
    target = _target(args)
    budget = _llm.budget_gb(target["memory_gb"], target["gpus"], target["gpu_memory_gb"], target["context"])
    models = _llm.run_ezra(["models", "list", "--json"], ezra=getattr(args, "ezra", None))
    ranked = _llm.rank_for_agents(models, budget=budget, has_gpu=bool(target["gpus"]), ctx_size=args.ctx_size,
                                  include_gated=bool(getattr(args, "include_gated", False)),
                                  keyword=getattr(args, "keyword", None))
    limit = getattr(args, "limit", 5) or 0
    picks = ranked[:limit] if limit > 0 else ranked
    engines = _llm.available_engines()

    notes = []
    if target["context"] == "login-node":
        notes.append(f"Login node: capped at {_llm.LOGIN_NODE_BUDGET_GB} GB and {_llm.LOGIN_NODE_THREADS} threads because it "
                     "is shared. For a larger model, size a compute node with --for-machine and serve it with "
                     "`llm-serve --where batch` or from inside an idev/salloc allocation.")
    if target["context"] == "workstation" and not target["gpus"]:
        notes.append("No GPU detected: GGUF models on CPU with llama-server.")
    if picks and not engines.get(picks[0]["engine"]):
        notes.append(f"`{picks[0]['engine']}` is not on PATH here; install it (llama.cpp release binaries ship "
                     "llama-server) or pass --engine-bin to llm-serve.")
    if not picks:
        notes.append("Nothing fits. Lower --ctx-size, use a compute node (--for-machine), or a GPU node.")

    top = picks[0]["name"] if picks else None
    return {
        "hardware": {**target, "budget_gb": budget},
        "ctx_size": args.ctx_size,
        "engines_on_path": engines,
        "recommendations": picks,
        "top_pick": top,
        "next": (f"chimes-agent llm-download --model {top}" if top else None),
        "notes": notes,
    }
