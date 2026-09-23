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
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Optional


def _hash_inputs(input_dict: dict) -> str:
    return hashlib.sha256(json.dumps(input_dict, sort_keys=True, default=str).encode()).hexdigest()


def _manifest_path(output_dir: Path, stage: str) -> Path:
    safe_stage = stage.replace("/", "_")
    return Path(output_dir) / f".manifest-{safe_stage}.json"


def load(output_dir: Path, stage: str) -> Optional[dict]:
    p = _manifest_path(output_dir, stage)
    if p.is_file():
        with open(p) as f:
            return json.load(f)
    return None


class InputMismatch(RuntimeError):
    pass


def begin(output_dir: Path, stage: str, input_dict: dict, *, force: bool = False):
    """Returns ("short_circuit", prior_manifest) or ("run", None)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_hash = _hash_inputs(input_dict)
    prior = load(output_dir, stage)

    if prior is not None and not force:
        if prior.get("input_hash") == input_hash and prior.get("status") == "done":
            return "short_circuit", prior
        if prior.get("input_hash") != input_hash:
            raise InputMismatch(
                f"{_manifest_path(output_dir, stage)} was written for different inputs "
                f"(hash {prior.get('input_hash')} != {input_hash} here). Pass "
                "--force to overwrite, or use a different --output-dir."
            )

    manifest = {
        "stage": stage,
        "input_hash": input_hash,
        "status": "running",
        "started_at": time.time(),
    }
    _manifest_path(output_dir, stage).write_text(json.dumps(manifest, indent=2))
    return "run", None


def finish(output_dir: Path, stage: str, input_dict: dict, outputs: dict, *, status: str = "done") -> dict:
    output_dir = Path(output_dir)
    manifest = {
        "stage": stage,
        "input_hash": _hash_inputs(input_dict),
        "status": status,
        "finished_at": time.time(),
        "outputs": outputs,
    }
    _manifest_path(output_dir, stage).write_text(json.dumps(manifest, indent=2, default=str))
    return manifest
