"""Filesystem helpers that survive Lustre's quirks.

On /p/lustre2, creating a nested directory (`mkdir -p a/b`) intermittently
fails with EREMOTE ("Object is remote", a DNE multi-metadata-server effect),
while creating one level at a time works. Seen repeatedly on 2026-09-29, in
our own stages and in `git filter-branch`.
"""

from __future__ import annotations

import errno
import time
from pathlib import Path

_RETRY = (errno.EREMOTE, errno.EAGAIN, errno.EINTR)


def ensure_dir(path, *, attempts: int = 6, delay_s: float = 0.5) -> Path:
    """mkdir -p, one level at a time, retrying transient Lustre errors."""
    path = Path(path)
    if path.is_dir():
        return path
    missing = []
    p = path
    while not p.exists():
        missing.append(p)
        p = p.parent
    for d in reversed(missing):
        for k in range(attempts):
            try:
                d.mkdir(exist_ok=True)
                break
            except OSError as exc:
                if exc.errno not in _RETRY or k == attempts - 1:
                    raise
                time.sleep(delay_s * (2 ** k))
    return path
