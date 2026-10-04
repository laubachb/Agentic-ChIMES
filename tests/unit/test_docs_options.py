"""docs/commands/options.md is generated from the stages; it must match the code."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_options_reference_is_current():
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "gen_options_doc.py"), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr + "\n(run: python3 tools/gen_options_doc.py)"
