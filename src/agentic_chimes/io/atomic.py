"""Crash-safe JSON files.

A job killed mid-write (walltime, OOM, a Lustre hiccup) used to leave a
truncated JSON file. The next run then crashed reading it: a truncated stage
manifest broke the CLI's one-JSON-object contract with a raw traceback, and
a truncated hyper-search point cache would break resuming the search.
Writes now go to a temporary file in the same directory and are renamed
into place (atomic on POSIX filesystems, Lustre included); readers treat an
unreadable file as missing.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def write_text(path, text: str) -> Path:
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def write_json(path, obj, **dump_kw) -> Path:
    dump_kw.setdefault("default", str)
    return write_text(path, json.dumps(obj, **dump_kw))


def read_json(path, default=None):
    """Parsed JSON, or `default` when the file is missing or unreadable (truncated, corrupt)."""
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default
