"""weights stage: al_driver row classification and methods, presets, AL decay."""

import numpy as np
import pytest

from agentic_chimes.stages import weights


def _work(tmp_path, frames=((2, True), (1, True))):
    """frames: (natoms, has_energy). Writes b-labeled.txt + natoms.txt like chimes_lsq."""
    rows, nat = [], []
    for n, energy in frames:
        for _ in range(3 * n):
            rows.append("Cu 1.5")
            nat.append(n)
        if energy:
            rows.append(f"+1 {-100.0 * n}")
            nat.append(n)
        rows.append("s_xx 0.2")
        nat.append(n)
    (tmp_path / "b-labeled.txt").write_text("\n".join(rows) + "\n")
    (tmp_path / "natoms.txt").write_text("\n".join(str(x) for x in nat) + "\n")
    return tmp_path


def test_presets_and_row_classification(tmp_path):
    res = weights.build(_work(tmp_path), preset="hierarchical2026")
    w = np.loadtxt(res["weights"])
    assert len(w) == res["rows"] == 6 + 1 + 1 + 3 + 1 + 1
    assert res["by_row_type"]["force"] == {"rows": 9, "min": 1.0, "max": 1.0}
    assert res["by_row_type"]["energy"]["max"] == pytest.approx(0.3)
    assert res["by_row_type"]["stress"]["max"] == pytest.approx(100.0)


def test_al_driver_methods_match_their_definitions():
    assert weights.method_weight(["B", [2.0, -1.0]], 0.0, 10, cycle=4) == pytest.approx(0.5)
    assert weights.method_weight(["B", [2.0, -1.0]], 0.0, 10, cycle=0) == pytest.approx(2.0)   # cycle 0 -> 1
    assert weights.method_weight(["C", [1.0, -1.0, 10.0]], -5.0, 1, 0) == pytest.approx(np.exp(-0.5))
    assert weights.method_weight(["E", [0.5]], 0.0, 16, 0) == pytest.approx(4.0)
    with pytest.raises(ValueError):
        weights.method_weight(["A", [1.0, 2.0]], 0, 1, 0)


def test_active_learning_decay_by_frame(tmp_path):
    res = weights.build(_work(tmp_path), preset="uniform", frame_cycles=[0, 4], decay_cycles=4)
    w = np.loadtxt(res["weights"])
    assert np.allclose(w[:8], 4.0)            # frame 0 (cycle 0 -> 1): 4/1
    assert np.allclose(w[8:], 1.0)            # frame 1 (cycle 4): 4/4
    with pytest.raises(ValueError, match="frames"):
        weights.build(tmp_path, frame_cycles=[0], decay_cycles=4)
