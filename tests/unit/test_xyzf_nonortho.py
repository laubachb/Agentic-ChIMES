"""chimes_lsq only recognises triclinic frames by the exact token
`NON_ORTHO` (ClassDefs.C). The writer once emitted `NON-ORTHO`, which
chimes_lsq would parse as an orthorhombic box with a garbage first length."""

import pytest

from agentic_chimes import config
from agentic_chimes.io import xyzf

FIXTURE = config.CHIMES_LSQ_ROOT / "test_suite-lsq" / "nonorth2" / "input.xyzf"


def test_writer_uses_chimes_lsq_token(tmp_path):
    fr = xyzf.Frame(symbols=["C"], positions=[[0, 0, 0]], forces=[[0, 0, 0]],
                    box=[[3.0, 0, 0], [1.0, 3.0, 0], [0, 0, 3.0]], non_ortho=True, energy=-1.0)
    out = tmp_path / "t.xyzf"
    xyzf.write_xyzf([fr], out)
    assert out.read_text().splitlines()[1].split()[0] == "NON_ORTHO"
    back = xyzf.read_xyzf(out)[0]
    assert back.non_ortho and back.box[1] == [1.0, 3.0, 0.0] and back.energy == -1.0


@pytest.mark.skipif(not FIXTURE.is_file(), reason="codes/ not cloned")
def test_reads_real_chimes_lsq_nonortho_fixture():
    frames = xyzf.read_xyzf(FIXTURE)
    assert frames and all(f.non_ortho for f in frames)
    assert frames[0].box[1][0] != 0.0  # tilted b vector survives


def test_mixed_pool_is_written_uniformly_non_ortho(tmp_path):
    """chimes_lsq segfaults on a plain orthorhombic frame after a NON_ORTHO
    one (found on curated MatPES data); the writer must never mix formats."""
    tri = xyzf.Frame(symbols=["C"], positions=[[0, 0, 0]], forces=[[0, 0, 0]],
                     box=[[3.0, 0, 0], [1.0, 3.0, 0], [0, 0, 3.0]], non_ortho=True)
    ortho = xyzf.Frame(symbols=["C"], positions=[[0, 0, 0]], forces=[[0, 0, 0]], box=[4.0, 5.0, 6.0])
    out = tmp_path / "mixed.xyzf"
    xyzf.write_xyzf([tri, ortho], out)
    headers = out.read_text().splitlines()[1::3]
    assert all(h.startswith("NON_ORTHO") for h in headers)
    back = xyzf.read_xyzf(out)
    assert back[0].non_ortho and not back[1].non_ortho and back[1].box == [4.0, 5.0, 6.0]
    only_ortho = tmp_path / "ortho.xyzf"
    xyzf.write_xyzf([ortho], only_ortho)
    assert only_ortho.read_text().splitlines()[1] == "4.0 5.0 6.0"
