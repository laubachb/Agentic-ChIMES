"""qe-converge scan layout and --collect on synthetic pw.x output; triclinic pw.in."""

import json
from pathlib import Path
from types import SimpleNamespace

from agentic_chimes.io import xyzf as xyzf_io
from agentic_chimes.stages import qe_converge
from agentic_chimes.stages.qe_relabel import _render_pw_in


def test_scan_points_two_ladders():
    pts = qe_converge.scan_points([30, 60, 45], [0.4, 0.25, 0.12])
    names = [p[0] for p in pts]
    assert names[:3] == ["ecut_30", "ecut_45", "ecut_60"]
    assert all(p[2] == 0.25 for p in pts[:3])              # ecut ladder at the middle kspacing
    assert {p[1] for p in pts[3:]} == {60} and len(pts) == 5   # kspacing ladder at the largest ecut, no duplicate point


def test_triclinic_cell_is_written_in_full():
    fr = xyzf_io.Frame(symbols=["Cu"], positions=[[0, 0, 0]], forces=[[0, 0, 0]],
                       box=[[3.0, 0, 0], [1.0, 2.9, 0], [0.5, 0.4, 2.8]], non_ortho=True)
    text = _render_pw_in(fr, ["Cu"], {"Cu": 63.5}, {"Cu": "/pp/Cu.upf"}, ecutwfc=40, ecutrho=None, kpoints=[2, 2, 2],
                         smearing="gaussian", degauss=0.01, conv_thr=1e-8)
    assert "1.0000000000 2.9000000000 0.0000000000" in text and "ibrav = 0" in text


def _pw_out(e_ry, fx, p_kbar):
    s = (f"!    total energy              =   {e_ry:.8f} Ry\n     convergence has been achieved in   5 iterations\n"
         "     Forces acting on atoms (cartesian axes, Ry/au):\n"
         f"     atom    1 type  1   force =    {fx:.8f}   0.00000000   0.00000000\n"
         f"     atom    2 type  1   force =   {-fx:.8f}   0.00000000   0.00000000\n\n"
         f"          total   stress  (Ry/bohr**3)                   (kbar)     P=      {p_kbar:.2f}\n")
    for i in range(3):
        row = [0.0] * 3
        row[i] = p_kbar
        s += f"   0.0 0.0 0.0  {row[0]:10.2f} {row[1]:10.2f} {row[2]:10.2f}\n"
    return s


def test_collect_recommends_cheapest_converged_setting(tmp_path):
    pts = [("ecut_30", 30, 0.25), ("ecut_45", 45, 0.25), ("ecut_60", 60, 0.25), ("kspacing_0.4", 60, 0.4), ("kspacing_0.12", 60, 0.12)]
    man = {"frame": {"natoms": 2, "symbols": ["Cu", "Cu"]}, "points": [], "ecutwfc_values": [30, 45, 60], "kspacing_values": [0.4, 0.25, 0.12]}
    for name, e, k in pts:
        d = tmp_path / name
        d.mkdir()
        de = {30: 0.004, 45: 0.00005, 60: 0.0}[e]        # Ry per 2 atoms: 30 Ry ~27 meV/atom off, 45 Ry ~0.3 meV/atom
        dk = {0.4: 0.002, 0.25: 0.0, 0.12: 0.0}[k]
        (d / "pw.out").write_text(_pw_out(-100.0 + de + dk, 0.01 + de, 10.0 + 100 * de))
        man["points"].append({"name": name, "dir": str(d), "ecutwfc": e, "kspacing": k})
    (tmp_path / "qe_converge_manifest.json").write_text(json.dumps(man))
    r = qe_converge.run(SimpleNamespace(collect=str(tmp_path), tol_energy_mev_atom=1.0, tol_force_mev_ang=10.0, tol_pressure_gpa=0.1))
    assert r["recommended"] == {"ecutwfc": 45, "kspacing": 0.25, "ecutrho": 180}
    assert r["ecutwfc"]["table"][0]["converged_vs_finest"] is False and r["status_counts"]["converged"] == 5
