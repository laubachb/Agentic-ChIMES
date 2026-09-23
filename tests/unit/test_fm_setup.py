"""Round-trip fm_setup.in parsing/rendering against the real golden fixtures
shipped in codes/chimes_lsq-LLfork/test_suite-lsq/. Semantic round-trip
(re-parsing a rendering reproduces the same structured dict), not byte-for-
byte, since real fm_setup.in files vary in whitespace/alignment.
"""

from agentic_chimes import config
from agentic_chimes.io import fm_setup

FIXTURES = [
    config.CHIMES_LSQ_ROOT / "test_suite-lsq" / "test_4atoms.2" / "fm_setup.in",
    config.CHIMES_LSQ_ROOT / "test_suite-lsq" / "special3b" / "fm_setup.in",
]


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
