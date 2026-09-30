"""Round-trip fm_setup.in parsing/rendering against the real golden fixtures
shipped in codes/chimes_lsq-LLfork/test_suite-lsq/. Semantic round-trip
(re-parsing a rendering reproduces the same structured dict), not byte-for-
byte, since real fm_setup.in files vary in whitespace/alignment.

codes/ is gitignored (see docs/concepts/vendored_forks.md) and not cloned
by CI (.github/workflows/ci.yml runs no `chimes-agent setup`), so these
tests skip cleanly rather than failing when the fixtures aren't present --
same pattern as test_evaluate.py's chimescalc_lib check. Run
`chimes-agent setup --component codes` locally to exercise them for real.
"""

import pytest

from agentic_chimes import config
from agentic_chimes.io import fm_setup

FIXTURES = [
    config.CHIMES_LSQ_ROOT / "test_suite-lsq" / "test_4atoms.2" / "fm_setup.in",
    config.CHIMES_LSQ_ROOT / "test_suite-lsq" / "special3b" / "fm_setup.in",
]

pytestmark = pytest.mark.skipif(
    not all(p.is_file() for p in FIXTURES),
    reason="codes/chimes_lsq-LLfork not cloned; run `chimes-agent setup --component codes`",
)


def test_fixtures_exist():
    for path in FIXTURES:
        assert path.is_file(), f"golden fixture missing: {path}"


def test_round_trip():
    for path in FIXTURES:
        original = fm_setup.parse(path.read_text())
        rendered = fm_setup.render(original)
        reparsed = fm_setup.parse(rendered)
        assert reparsed == original, f"round-trip mismatch for {path}"


def test_pairtyp_without_4body_omits_range():
    # special3b's "CHEBYSHEV 12 5" (no 4-body order) must not gain a
    # fabricated order4/cheby_min/cheby_max on render.
    path = config.CHIMES_LSQ_ROOT / "test_suite-lsq" / "special3b" / "fm_setup.in"
    data = fm_setup.parse(path.read_text())
    assert "order4" not in data
    rendered = fm_setup.render(data)
    assert "CHEBYSHEV 12 5\n" in rendered or "CHEBYSHEV 12 5" in rendered.splitlines()[rendered.splitlines().index("# PAIRTYP #") + 1]


def test_exclude_and_special_blocks_preserved():
    d1 = fm_setup.parse((config.CHIMES_LSQ_ROOT / "test_suite-lsq" / "test_4atoms.2" / "fm_setup.in").read_text())
    assert len(d1["exclude_3b"]) == 2
    assert len(d1["exclude_4b"]) == 4

    d2 = fm_setup.parse((config.CHIMES_LSQ_ROOT / "test_suite-lsq" / "special3b" / "fm_setup.in").read_text())
    assert len(d2["special_blocks"]) == 1
    assert d2["special_blocks"][0]["mode"] == "SPECIFIC"
    assert len(d2["special_blocks"][0]["rows"]) == 4


def test_specific_3b_rows_names_values_and_exclusions():
    from agentic_chimes.io.fm_setup import specific_3b_rows

    rows = specific_3b_rows(["Cu", "Zr"], {"Cu-Cu": 5.0, "Zr-Cu": 6.33, "Zr-Zr": 5.5})
    assert rows == [
        ["CuCuCuCuCuCu", "CuCu", "CuCu", "CuCu", "5", "5", "5"],
        ["CuCuCuZrCuZr", "CuCu", "CuZr", "CuZr", "5", "6.33", "6.33"],
        ["CuZrCuZrZrZr", "CuZr", "CuZr", "ZrZr", "6.33", "6.33", "5.5"],
        ["ZrZrZrZrZrZr", "ZrZr", "ZrZr", "ZrZr", "5.5", "5.5", "5.5"],
    ]
    # an excluded triplet type must not get a row (chimesFF maps it to -1)
    kept = specific_3b_rows(["Cu", "Zr"], {"Cu-Cu": 5.0, "Cu-Zr": 6.33, "Zr-Zr": 5.5}, exclude=[["Zr", "Cu", "Cu"]])
    assert [r[0] for r in kept] == ["CuCuCuCuCuCu", "CuZrCuZrZrZr", "ZrZrZrZrZrZr"]


def test_hierarchical_excludes_round_trip():
    from agentic_chimes.io import fm_setup

    p = fm_setup.parse(fm_setup.render({
        "trjfile": "t.xyzf", "nframes": 1, "order2": 6, "order3": 4, "hierarc": True,
        "atom_types": [{"idx": 1, "symbol": "Cu", "charge": 0.0, "mass": 63.5}, {"idx": 2, "symbol": "Zr", "charge": 0.0, "mass": 91.2}],
        "pairs": [{"idx": 1, "type1": "Cu", "type2": "Cu", "s_minim": 2.2, "s_maxim": 8.0, "s_delta": 0.01, "morse_lambda": 2.5}],
        "exclude_1b": [["Cu"], ["Zr"]], "exclude_2b": [["Cu", "Cu"], ["Zr", "Zr"]], "exclude_3b": [["Cu", "Cu", "Cu"]],
    }))
    assert p["hierarc"] is True
    assert p["exclude_1b"] == [["Cu"], ["Zr"]]
    assert p["exclude_2b"] == [["Cu", "Cu"], ["Zr", "Zr"]]
    assert p["exclude_3b"] == [["Cu", "Cu", "Cu"]]
