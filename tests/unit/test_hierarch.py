"""Hierarchical fitting helpers."""

from agentic_chimes.stages import hierarch

PARAMS = """! header
PAIRTYP: CHEBYSHEV  6 4 0 -1 1
ATOM TYPES: 1
# TYPEIDX #	# ATM_TYP #	# ATMCHRG #	# ATMMASS #
0		Cu		0		63.546

FCUT TYPE: CUBIC
PAIR CHEBYSHEV PENALTY DIST:    0.02
PAIR CHEBYSHEV PENALTY SCALING: 10000.0

ENDFILE
"""


def test_zero_penalty_copy_and_element_detection(tmp_path):
    src = tmp_path / "Cu.params.txt"
    src.write_text(PARAMS)
    out = hierarch.zero_penalty_copy(src, tmp_path / "z.txt").read_text()
    assert "PENALTY DIST:    0.0" in out and "PENALTY SCALING: 0.0" in out
    assert "0.02" not in out and "10000" not in out
    assert out.index("FCUT TYPE") < out.index("PENALTY DIST")
    assert hierarch._elements_of(src) == ["Cu"]
