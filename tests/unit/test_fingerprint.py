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


def _two_element_frames():
    """A rocksalt-like AB cell and the same geometry with half the B atoms swapped to A."""
    from ase.build import bulk

    at = bulk("NaCl", "rocksalt", a=5.6, cubic=True)
    syms = at.get_chemical_symbols()
    swapped = ["Na" if (s == "Cl" and i % 2) else s for i, s in enumerate(syms)]
    mk = lambda ss: xyzf_io.Frame(symbols=ss, positions=at.get_positions().tolist(), forces=[[0, 0, 0]] * len(at), box=[5.6] * 3)
    return mk(syms), mk(swapped)


def _model_for(elements, tmp_path, masses):
    rows = "\n".join(f"{i}\t{e}\t0\t{masses[e]}" for i, e in enumerate(elements))
    pairs = [(a, b) for i, a in enumerate(elements) for b in elements[i:]]
    prow = "\n".join(f"{k}\t{a}\t{b}\t1.0\t6.0\tMORSE\t1.5" for k, (a, b) in enumerate(pairs))
    (tmp_path / "params.txt").write_text(f"""ATOM TYPES: {len(elements)}

# TYPEIDX #	# ATM_TYP #	# ATMCHRG #	# ATMMASS #
{rows}

ATOM PAIRS: {len(pairs)}

# PAIRIDX #	# ATM_TY1 #	# ATM_TY1 #	# S_MINIM #	# S_MAXIM #	# CHBDIST #	# MORSE_LAMBDA #
{prow}
""")
    return fp.ModelCutoffs(tmp_path / "params.txt")


def test_hybrid_metric_sees_composition_that_structure_cannot(tmp_path):
    a, b = _two_element_frames()
    model = _model_for(["Na", "Cl"], tmp_path, {"Na": 22.99, "Cl": 35.45})
    assert model.descriptor == {"Na": 22.99, "Cl": 35.45}
    fa1, fb1 = (fp.fingerprint(f, model, (2, 3), alpha=1.0) for f in (a, b))
    fa0, fb0 = (fp.fingerprint(f, model, (2, 3), alpha=0.0) for f in (a, b))
    assert np.allclose(fa1, fb1)                       # same geometry: structure only cannot tell them apart
    assert not np.allclose(fa0, fb0)                   # composition only can
    assert np.isclose(fa0.sum(), 2.0) and np.isclose(fb0.sum(), 2.0)   # one normalized histogram per order
    # alpha=0 on a pure-A cell: every composition distance is 0 -> all mass in the first bin
    pure = xyzf_io.Frame(symbols=["Na"] * a.natoms, positions=a.positions, forces=a.forces, box=a.box)
    f0 = fp.fingerprint(pure, model, (2,), alpha=0.0)
    assert f0[0] == 1.0


def test_centrality_ordering_puts_the_central_atom_first():
    # sorted edges e = [0.1, 0.2, 0.9]: S_1 = 0.3, S_2 = 1.0, S_3 = 1.1 -> atom 1 most central
    edges = np.array([[0.1, 0.2, 0.9]])
    types = np.array([["B", "A", "A"]], dtype=object)
    d = fp.composition_descriptors(edges, types, 3, {"A": 1.0, "B": 2.0})
    assert d.tolist() == [[2.0, 1.0, 1.0]]
    d2 = fp.composition_descriptors(np.array([[0.5]]), np.array([["B", "A"]], dtype=object), 2, {"A": 1.0, "B": 2.0})
    assert d2.tolist() == [[1.0, 2.0]]


def test_stage_sweep_reports_each_weight_and_the_baseline(tmp_path):
    from types import SimpleNamespace

    from agentic_chimes.stages import fingerprint as stage

    a, b = _two_element_frames()
    model_path = _model_for(["Na", "Cl"], tmp_path, {"Na": 22.99, "Cl": 35.45})
    rng = np.random.default_rng(0)

    def jitter(f, n):
        return [xyzf_io.Frame(symbols=f.symbols, positions=(np.asarray(f.positions) + rng.normal(0, 0.05, (f.natoms, 3))).tolist(),
                              forces=f.forces, box=f.box) for _ in range(n)]

    xyzf_io.write_xyzf(jitter(a, 8), tmp_path / "ref.xyzf")
    xyzf_io.write_xyzf(jitter(b, 6), tmp_path / "cand.xyzf")
    r = stage.run(SimpleNamespace(params=str(tmp_path / "params.txt"), reference_xyzf=str(tmp_path / "ref.xyzf"),
                                  candidates_xyzf=str(tmp_path / "cand.xyzf"), orders=[2], max_frames=None, max_clusters=500,
                                  alpha=0.1, structure_weight=0.5, structure_weights=[0.0, 1.0], descriptor="mass", workers=1,
                                  machine=None, output_dir=str(tmp_path / "out")))
    assert set(r["by_structure_weight"]) == {"0.5", "0", "1", "type-agnostic"}
    assert r["element_descriptor"] == {"Na": 22.99, "Cl": 35.45}
    # same geometry, different chemical order: composition must separate the sets more than structure
    assert any("composition separates" in n for n in r["notes"])
    with pytest.raises(ValueError, match="descriptor"):
        (tmp_path / "one").mkdir()
        _model_for(["Na"], tmp_path / "one", {"Na": 22.99})
        stage.run(SimpleNamespace(params=str(tmp_path / "one" / "params.txt"),
                                  reference_xyzf=str(tmp_path / "ref.xyzf"), candidates_xyzf=None, orders=[2], max_frames=None,
                                  max_clusters=500, alpha=0.1, structure_weight=0.5, structure_weights=None, descriptor="mass",
                                  workers=1, machine=None, output_dir=str(tmp_path / "out1")))
