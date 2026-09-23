"""Validates al-select end to end against the real, known-good CHON fixture
already confirmed working by test_evaluate.py
(`codes/chimes_calculator-LLfork/serial_interface/tests/force_fields/test_params.CHON.txt`)
-- a candidate pool is synthesized by jittering that fixture's one reference
structure (same box/composition, small random position perturbations,
deterministic seed) since the vendored fixture itself only ships one frame.
This exercises the full real pipeline (ctypes energy evaluation + al_driver's
own `gen_subset` Metropolis-MC selector out of gen_selections.py), not a mock.

A separately-checked h2o-invr params.txt (INVRSE_R basis) was tried first and
rejected by chimes_calculator's serial C++ parser ("Incorrect input in
line...Expect 7 or 8 entries") -- that fixture's basis format isn't
compatible with the vendored serial_interface build here, hence sticking
with the fixture test_evaluate.py already validated.

Skips (does not fail) if chimes_calculator hasn't been built, or if
matplotlib/cycler (gen_selections.py's own unconditional imports) aren't
installed -- both are environment preconditions, not code defects.
"""

import importlib.util
import random

import pytest

from agentic_chimes import config
from agentic_chimes.io import xyzf as xyzf_io
from agentic_chimes.stages import al_select

FIXTURES_DIR = config.CHIMES_CALCULATOR_ROOT / "serial_interface" / "tests"
REFERENCE_XYZ = FIXTURES_DIR / "configurations" / "CHON.testfile_#000.xyz"
PARAMS_TXT = FIXTURES_DIR / "force_fields" / "test_params.CHON.txt"

N_CANDIDATES = 40


def _chimescalc_lib_available() -> bool:
    return config.resolve_component("chimescalc_lib", required=False) is not None


def _matplotlib_available() -> bool:
    return importlib.util.find_spec("matplotlib") is not None and importlib.util.find_spec("cycler") is not None


pytestmark = pytest.mark.skipif(
    not (_chimescalc_lib_available() and _matplotlib_available()),
    reason="requires chimes_calculator built + matplotlib/cycler installed (pip install agentic-chimes[al-select])",
)


def _read_reference():
    lines = REFERENCE_XYZ.read_text().splitlines()
    natoms = int(lines[0].split()[0])
    box9 = [float(x) for x in lines[1].split()]
    assert box9[1] == 0 and box9[2] == 0 and box9[3] == 0 and box9[5] == 0 and box9[6] == 0 and box9[7] == 0, "expected an orthorhombic box"
    box = [box9[0], box9[4], box9[8]]
    symbols, positions = [], []
    for row in lines[2 : 2 + natoms]:
        toks = row.split()
        symbols.append(toks[0])
        positions.append([float(toks[1]), float(toks[2]), float(toks[3])])
    return symbols, positions, box


def _write_synthetic_pool(path, n_frames, seed=7):
    symbols, positions, box = _read_reference()
    natoms = len(symbols)
    rng = random.Random(seed)
    frames = []
    for _ in range(n_frames):
        jittered = [[c + rng.uniform(-0.1, 0.1) for c in pos] for pos in positions]
        frames.append(
            xyzf_io.Frame(
                symbols=list(symbols),
                positions=jittered,
                forces=[[0.0, 0.0, 0.0]] * natoms,
                box=list(box),
            )
        )
    xyzf_io.write_xyzf(frames, path)


class _Args:
    def __init__(self, **kwargs):
        self.output_dir = None
        for k, v in kwargs.items():
            setattr(self, k, v)


def test_al_select_picks_a_diverse_subset(tmp_path):
    assert REFERENCE_XYZ.is_file() and PARAMS_TXT.is_file(), "vendored fixture missing"

    candidate_xyzf = tmp_path / "candidates.xyzf"
    _write_synthetic_pool(candidate_xyzf, N_CANDIDATES)

    args = _Args(
        candidate_frames=str(candidate_xyzf),
        params=str(PARAMS_TXT),
        n_select=10,
        histogram_bins=10,
        nsweep=2,
        seed=1,
        output_dir=str(tmp_path / "out"),
    )
    result = al_select.run(args)

    assert result["n_candidates"] == N_CANDIDATES
    assert result["n_selected"] == 10
    assert len(set(result["selected_indices"])) == 10
    assert all(0 <= i < N_CANDIDATES for i in result["selected_indices"])
    assert result["selected_indices"] == sorted(result["selected_indices"])

    selected = xyzf_io.read_xyzf(result["selected_xyzf"])
    assert len(selected) == 10
    assert all(fr.natoms == 9 for fr in selected)

    central_repo_lines = (tmp_path / "out" / "central_repo_energies_normed.txt").read_text().split()
    assert len(central_repo_lines) == 10


def test_al_select_rejects_n_select_over_pool_size(tmp_path):
    candidate_xyzf = tmp_path / "candidates.xyzf"
    _write_synthetic_pool(candidate_xyzf, N_CANDIDATES)

    args = _Args(
        candidate_frames=str(candidate_xyzf),
        params=str(PARAMS_TXT),
        n_select=1000,
        output_dir=str(tmp_path / "out"),
    )
    with pytest.raises(ValueError, match="exceeds candidate pool size"):
        al_select.run(args)


def test_al_select_chains_via_central_repo(tmp_path):
    candidate_xyzf = tmp_path / "candidates.xyzf"
    _write_synthetic_pool(candidate_xyzf, N_CANDIDATES)

    args1 = _Args(
        candidate_frames=str(candidate_xyzf),
        params=str(PARAMS_TXT),
        n_select=10,
        histogram_bins=10,
        nsweep=2,
        seed=1,
        output_dir=str(tmp_path / "first"),
    )
    result1 = al_select.run(args1)

    args2 = _Args(
        candidate_frames=str(candidate_xyzf),
        params=str(PARAMS_TXT),
        n_select=10,
        histogram_bins=10,
        nsweep=2,
        seed=2,
        central_repo=result1["central_repo_out"],
        output_dir=str(tmp_path / "second"),
    )
    al_select.run(args2)

    combined_lines = (tmp_path / "second" / "central_repo_energies_normed.txt").read_text().split()
    assert len(combined_lines) == 20
