"""Cluster-graph fingerprints: exact agreement with chimes_calculator's shipped
tool, periodic correctness, and the Mahalanobis statistics."""

import numpy as np
import pytest

from agentic_chimes import config
from agentic_chimes.io import fingerprint as fp
from agentic_chimes.io import xyzf as xyzf_io

EX = config.CHIMES_CALCULATOR_ROOT / "etc" / "lmp" / "tests" / "example-fingerprint"


def _water_frame():
    lines = (EX / "data.in").read_text().splitlines()
    box = float(lines[5].split()[1])
    start = lines.index("Atoms") + 2
    rows = [ln.split() for ln in lines[start:start + 96]]
    return xyzf_io.Frame(symbols=["O" if r[1] == "1" else "H" for r in rows],
                         positions=[[float(x) for x in r[2:5]] for r in rows], forces=[[0, 0, 0]] * 96, box=[box] * 3)


@pytest.mark.skipif(not (EX / "expected_output").is_dir(), reason="chimes_calculator fork not cloned")
def test_matches_the_shipped_fingerprint_tool():
    model = fp.ModelCutoffs(EX / "params.txt")
    cl = fp.clusters(_water_frame(), model)
    assert {o: len(v) for o, v in cl.items()} == {2: 1886, 3: 1618, 4: 119}
    for o in (2, 3, 4):
        ref = np.loadtxt(EX / "expected_output" / f"0-0.{o}b_clu-s.hist")[:, 1]
        hist, _ = fp.histogram(cl[o], o, max_clusters=None)
        assert np.abs(hist - ref).max() < 1e-5          # the reference is printed to 6 digits


@pytest.mark.skipif(not (EX / "params.txt").is_file(), reason="chimes_calculator fork not cloned")
def test_periodic_images_counted_once():
    model = fp.ModelCutoffs(EX / "params.txt")
    f = _water_frame()
    cell = np.diag(f.box)
    pos = np.asarray(f.positions)
    shifts = [i * cell[0] + j * cell[1] for i in range(2) for j in range(2)]
    big = xyzf_io.Frame(symbols=f.symbols * 4, positions=np.concatenate([pos + s for s in shifts]).tolist(),
                        forces=[[0, 0, 0]] * (4 * f.natoms), box=[2 * f.box[0], 2 * f.box[1], f.box[2]])
    a, b = fp.clusters(f, model, (2, 3)), fp.clusters(big, model, (2, 3))
    assert all(len(b[o]) == 4 * len(a[o]) for o in (2, 3))


def test_mahalanobis_separates_shifted_sets_and_not_resamples():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(60, 6))
    same = rng.normal(size=(60, 6))
    shifted = rng.normal(size=(60, 6)) + 3.0      # the paper's D2 measures separation of distributions, not of means
    assert not fp.mahalanobis_sets(base, same)["distinguishable"]
    assert fp.mahalanobis_sets(base, shifted)["distinguishable"]
    nov = fp.novelty(base, np.vstack([base[:5], base[:5] + 10.0]))
    assert nov["novel"] == [False] * 5 + [True] * 5
