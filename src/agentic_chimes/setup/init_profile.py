"""Write a machine profile for a Slurm cluster from a few answers, filling
what the scheduler can tell us (partitions, cores per node, the user's
accounts). The result is a YAML file to pass as `--machine <path>`; nothing
is built or submitted. See docs/concepts/machine_profiles.md.
"""

from __future__ import annotations

import getpass
import os
import subprocess
from pathlib import Path

import yaml


def _run(cmd) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def detect() -> dict:
    """What sinfo/sacctmgr report: partitions (name, cores per node, time limit, default) and the user's accounts."""
    parts = []
    for ln in _run(["sinfo", "-h", "-o", "%P|%c|%l"]).splitlines():
        t = ln.split("|")
        if len(t) == 3 and t[0]:
            try:
                cores = int(t[1].rstrip("+"))
            except ValueError:
                cores = None
            parts.append({"name": t[0].rstrip("*"), "default": t[0].endswith("*"), "cores_per_node": cores, "time_limit": t[2]})
    accounts = sorted({a.strip() for a in _run(["sacctmgr", "-nP", "show", "assoc", f"user={getpass.getuser()}",
                                                "format=account"]).splitlines() if a.strip()})
    return {"partitions": parts, "accounts": accounts, "slurm": bool(parts)}


def _pick_partitions(parts, debug, batch):
    names = [p["name"] for p in parts]
    if not debug:
        debug = next((n for n in names if "debug" in n.lower() or "dev" in n.lower()), None)
    if not batch:
        batch = next((p["name"] for p in parts if p["default"] and p["name"] != debug), None) \
            or next((n for n in names if "batch" in n.lower() or "normal" in n.lower()), None) \
            or next((n for n in names if n != debug), None)
    return debug, batch


def write(path, *, name=None, account=None, debug_partition=None, batch_partition=None, cores_per_node=None,
          scratch=None, modules=None, hosttype=None, launcher="srun", overwrite=False) -> dict:
    path = Path(path).expanduser().resolve()
    if path.exists() and not overwrite:
        raise ValueError(f"{path} exists; pass --force to overwrite it")
    found = detect()
    debug_partition, batch_partition = _pick_partitions(found["partitions"], debug_partition, batch_partition)
    if not cores_per_node:
        by_name = {p["name"]: p["cores_per_node"] for p in found["partitions"]}
        cores_per_node = by_name.get(batch_partition) or by_name.get(debug_partition)
    todo = []
    if not account and len(found["accounts"]) != 1:
        todo.append("account: set CHIMES_ACCOUNT (or --account)" + (f"; yours are {found['accounts']}" if found["accounts"] else ""))
    if not debug_partition or not batch_partition:
        todo.append("partitions: give --debug-partition and --batch-partition (sinfo found none to choose from)")
    if not cores_per_node:
        todo.append("default_ntasks_per_node: give --cores-per-node (cores on one node of the batch partition)")
    if not scratch:
        todo.append("filesystem.scratch_root: give --scratch, a directory compute nodes can see (not /tmp or $HOME on most clusters)")
    if not modules and (hosttype or "none") == "none":
        todo.append("modules: list the compiler and MPI modules to load (--modules a,b), or load them before `setup`")
    default_account = account or (found["accounts"][0] if len(found["accounts"]) == 1 else "")
    profile = {
        "name": name or path.stem,
        "job_system": "slurm",
        "launcher": launcher,
        # CHIMES_ACCOUNT wins, so the file can be shared; the detected or given account is only the fallback
        "account": "${CHIMES_ACCOUNT" + (f":-{default_account}" if default_account else "") + "}",
        "hosttype": hosttype or "none",
        "partitions": {"debug": debug_partition or "debug", "batch": batch_partition or "batch"},
        "default_ntasks_per_node": int(cores_per_node or 16),
        "require_ntasks_per_node_or_exclusive": True,
        "poll_interval_s": 60,
        "modules": list(modules or []),
        "filesystem": {"scratch_root": scratch or "${CHIMES_SCRATCH}", "max_small_files_per_dir": 5000, "bulk_archive_root": None},
        "qe": {"configure_extra_args": []},
    }
    header = ("# Machine profile written by `chimes-agent setup --init-profile`.\n"
              "# Edit freely; see docs/concepts/machine_profiles.md for every key.\n"
              + "".join(f"# TODO {t}\n" for t in todo))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + yaml.safe_dump(profile, sort_keys=False))

    from .. import machines

    loaded = machines.load_profile(str(path))      # the file must load as a profile, or this is a bug here
    return {"profile": str(path), "name": loaded.name, "account": loaded.account or None,
            "partitions": loaded.partitions, "default_ntasks_per_node": loaded.default_ntasks_per_node,
            "scratch_root": loaded.filesystem.get("scratch_root") or None, "modules": loaded.modules,
            "detected": found, "todo": todo,
            "next": [f"chimes-agent setup --machine {path} --component all", f"chimes-agent doctor --machine {path}"]}
