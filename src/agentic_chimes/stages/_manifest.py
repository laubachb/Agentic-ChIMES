"""Per-output-dir manifest: the idempotency/resume mechanism every stage uses
in place of al_driver's append-only, substring-parsed `restart.dat`.

A stage invoked with --output-dir writes `<output-dir>/.manifest-<stage>.json`
(namespaced by stage name, since chaining stages into the same --output-dir
is the normal pattern -- e.g. fm-setup-gen and amat-build both writing into
one run directory -- and they must not collide on a single shared file)
recording an input hash + status. Re-invoking with the same --output-dir and
matching inputs short-circuits (reprints the prior outputs) unless --force;
re-invoking with *different* inputs at the same path is refused (not
silently overwritten) unless --force, so a caller can't accidentally clobber
a differently-parameterized run.

Input *files* count by content, not just path: every string input naming an
existing file adds its size and modification time to the hash, so
regenerating `train.xyzf` in place makes the next run refuse (and say which
file changed) instead of reprinting results computed from the old file.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import time
from pathlib import Path
from typing import Optional

from ..io import atomic
from ..io import fs


def _file_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _file_strings(v)
    elif isinstance(value, dict):
        for v in value.values():
            yield from _file_strings(v)


def input_files(input_dict: dict) -> dict:
    """{resolved path: "size:mtime_ns"} for every input string that names an existing file."""
    out = {}
    for s in _file_strings(input_dict):
        if not s or len(s) > 4096 or "\n" in s:
            continue
        try:
            p = Path(s)
            if p.is_file():
                st = p.stat()
                out[str(p.resolve())] = f"{st.st_size}:{st.st_mtime_ns}"
        except (OSError, ValueError):
            continue
    return out


def _hash_inputs(input_dict: dict, files: Optional[dict] = None) -> str:
    files = input_files(input_dict) if files is None else files
    blob = json.dumps({"inputs": input_dict, "files": files}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def _manifest_path(output_dir: Path, stage: str) -> Path:
    safe_stage = stage.replace("/", "_")
    return Path(output_dir) / f".manifest-{safe_stage}.json"


def load(output_dir: Path, stage: str) -> Optional[dict]:
    """The stage's manifest, or None. An unreadable (e.g. truncated) manifest
    is treated as absent: it describes no completed run worth protecting."""
    from ..io import atomic

    return atomic.read_json(_manifest_path(output_dir, stage))


class InputMismatch(RuntimeError):
    pass


STALE_AFTER_S = 6 * 3600  # a "running" manifest from another host older than this is assumed dead


def _still_running(prior: dict) -> str:
    """Why a manifest marked "running" still looks live ("" if not)."""
    host, pid, started = prior.get("host"), prior.get("pid"), prior.get("started_at") or 0
    if not host or not pid:
        return ""  # written before host/pid were recorded
    if host == socket.gethostname():
        try:
            os.kill(int(pid), 0)
        except ProcessLookupError:
            return ""
        except PermissionError:
            pass
        return f"still running here (pid {pid})"
    age = time.time() - started
    if age < STALE_AFTER_S:
        return f"marked running on {host} (pid {pid}, started {age / 60:.0f} min ago; cannot check another host)"
    return ""


def begin(output_dir: Path, stage: str, input_dict: dict, *, force: bool = False):
    """Returns ("short_circuit", prior_manifest) or ("run", None)."""
    output_dir = Path(output_dir)
    fs.ensure_dir(output_dir)
    files = input_files(input_dict)
    input_hash = _hash_inputs(input_dict, files)
    prior = load(output_dir, stage)

    if prior is not None and "input_files" not in prior:
        # Manifest from before file fingerprints: compare on the old, path-only hash.
        legacy = hashlib.sha256(json.dumps(input_dict, sort_keys=True, default=str).encode()).hexdigest()
        if prior.get("input_hash") == legacy:
            prior = {**prior, "input_hash": input_hash}
    if prior is not None and not force and prior.get("status") == "running":
        busy = _still_running(prior)
        if busy:
            raise InputMismatch(f"{_manifest_path(output_dir, stage)}: another {stage} run is {busy}. Two runs in one "
                                "--output-dir overwrite each other; wait for it, use another --output-dir, or pass "
                                "--force if that run has died.")
    if prior is not None and not force:
        if prior.get("input_hash") == input_hash and prior.get("status") == "done":
            return "short_circuit", prior
        if prior.get("input_hash") != input_hash and prior.get("status") != "failed":  # a failed run protects nothing
            changed = sorted(p for p, sig in (prior.get("input_files") or {}).items() if files.get(p, sig) != sig)
            why = f"input file(s) changed since that run: {changed}" if changed else "different inputs"
            raise InputMismatch(
                f"{_manifest_path(output_dir, stage)} was written for {why} "
                f"(hash {prior.get('input_hash')} != {input_hash} here). Pass "
                "--force to overwrite, or use a different --output-dir."
            )

    manifest = {
        "stage": stage,
        "input_hash": input_hash,
        "input_files": files,
        "status": "running",
        "started_at": time.time(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
    }
    atomic.write_json(_manifest_path(output_dir, stage), manifest, indent=2)
    return "run", None


def finish(output_dir: Path, stage: str, input_dict: dict, outputs: dict, *, status: str = "done") -> dict:
    output_dir = Path(output_dir)
    # Keep the hash taken when the run began: a stage may legitimately touch
    # its own input files (e.g. `study --register` updates study.json).
    prior = load(output_dir, stage)
    if prior and prior.get("status") == "running":
        input_hash, files = prior["input_hash"], prior.get("input_files", {})
    else:
        files = input_files(input_dict)
        input_hash = _hash_inputs(input_dict, files)
    manifest = {
        "stage": stage,
        "input_hash": input_hash,
        "input_files": files,
        "status": status,
        "finished_at": time.time(),
        "outputs": outputs,
    }
    atomic.write_json(_manifest_path(output_dir, stage), manifest, indent=2)
    return manifest
