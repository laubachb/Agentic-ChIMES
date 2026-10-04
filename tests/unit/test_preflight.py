"""fm_setup.in pre-flight against the trajectory."""

from agentic_chimes.io import fm_setup
from agentic_chimes.io import xyzf as xyzf_io
from agentic_chimes.stages import _preflight


def _setup(tmp_path, nframes=3, nlayers=1, smin=2.0, smax=4.0, fitstrs="false", elements=("Cu",)):
    frames = [xyzf_io.Frame(symbols=["Cu", "Cu"], positions=[[0, 0, 0], [1.8, 1.8, 0.0]], forces=[[0, 0, 0]] * 2,
                            box=[3.6, 3.6, 3.6], energy=-1.0) for _ in range(3)]
    xyzf_io.write_xyzf(frames, tmp_path / "t.xyzf")
    text = fm_setup.render({"trjfile": str(tmp_path / "t.xyzf"), "nframes": nframes, "nlayers": nlayers, "order2": 6,
                            "order3": 0, "fitstrs": fitstrs, "fitener": "true",
                            "atom_types": [{"idx": i + 1, "symbol": e, "charge": 0.0, "mass": 63.5} for i, e in enumerate(elements)],
                            "pairs": [{"idx": 1, "type1": elements[0], "type2": elements[0], "s_minim": smin, "s_maxim": smax,
                                       "s_delta": 0.01, "morse_lambda": 2.5}]})
    (tmp_path / "fm_setup.in").write_text(text)
    return tmp_path / "fm_setup.in"


def test_clean_setup_passes(tmp_path):
    r = _preflight.check(_setup(tmp_path, smin=2.52))
    assert r["errors"] == [] and r["warnings"] == []


def test_errors_are_reported(tmp_path):
    r = _preflight.check(_setup(tmp_path, nframes=5, nlayers=0, smax=4.0, fitstrs="ALL", elements=("Zr",)))
    msg = " ".join(r["errors"])
    assert "NFRAMES is 5" in msg and "not ATOM TYPES" in msg and "outer cutoff" in msg and "stress" in msg


def test_inner_cutoff_warnings(tmp_path):
    close = 2.546  # 1.8*sqrt(2)
    r = _preflight.check(_setup(tmp_path, smin=2.6))
    assert any("inside S_MINIM" in w for w in r["warnings"])
    r = _preflight.check(_setup(tmp_path, smin=2.0))
    assert any("below the closest" in w for w in r["warnings"])
    assert close > 2.5


def test_worker_pools_spawn_rather_than_fork():
    """Forking after numba/OpenMP threads start can deadlock in fork() (al-batch on Dane); pools must spawn."""
    from agentic_chimes.io.pool import process_pool

    with process_pool(2) as pool:
        assert pool._mp_context.get_start_method() == "spawn"
        assert list(pool.map(abs, [-1, 2])) == [1, 2]
