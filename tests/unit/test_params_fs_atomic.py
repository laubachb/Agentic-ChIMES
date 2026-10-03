"""params.txt type/mass checks, Lustre-safe mkdir, crash-safe JSON."""

import errno
from pathlib import Path

import pytest

from agentic_chimes.io import atomic, fs
from agentic_chimes.io import params as params_io

PARAMS = """PAIRTYP: CHEBYSHEV  6 4 0 -1 1
ATOM TYPES: 2

# TYPEIDX #	# ATM_TYP #	# ATMCHRG #	# ATMMASS #
0		Cu		0		63.546
1		Zr		0		91.224

ATOM PAIRS: 3
"""


def test_types_and_masses_come_from_the_model(tmp_path):
    p = tmp_path / "params.txt"
    p.write_text(PARAMS)
    els, masses = params_io.resolve_types(p)
    assert els == ["Cu", "Zr"] and masses == {"Cu": 63.546, "Zr": 91.224}
    assert params_io.resolve_types(p, ["Zr", "Cu"], {"Cu": 63.5462})[0] == ["Cu", "Zr"]   # within tolerance
    with pytest.raises(ValueError, match="mass of Cu is 63.5"):
        params_io.resolve_types(p, masses={"Cu": 63.5})
    with pytest.raises(ValueError, match="'Al' is not in the model"):
        params_io.resolve_types(p, ["Cu", "Al"])
    with pytest.raises(ValueError, match="does not describe"):
        params_io.frame_elements_check(p, ["Cu", "O"])


def test_ensure_dir_retries_eremote(tmp_path, monkeypatch):
    calls = {"n": 0}
    real = Path.mkdir

    def flaky(self, *a, **k):
        calls["n"] += 1
        if calls["n"] == 2:                       # second level fails once, like Lustre DNE
            raise OSError(errno.EREMOTE, "Object is remote")
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "mkdir", flaky)
    target = fs.ensure_dir(tmp_path / "a" / "b" / "c", delay_s=0.0)
    assert target.is_dir() and calls["n"] == 4


def test_atomic_json_round_trip_and_tolerant_read(tmp_path):
    p = atomic.write_json(tmp_path / "x.json", {"a": 1})
    assert atomic.read_json(p) == {"a": 1}
    assert not list(tmp_path.glob(".x.json.*.tmp"))       # no temp file left behind
    p.write_text('{"a": ')
    assert atomic.read_json(p, default="gone") == "gone"
    assert atomic.read_json(tmp_path / "missing.json") is None
