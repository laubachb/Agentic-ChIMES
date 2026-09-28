"""Offline tests for the data phase: data-search, data-fetch (ColabFit via
local parquet files in the real schema, and local ASE files), data-generate,
data-curate, and qe-relabel's provenance/kspacing. Labels come from ASE's EMT
calculator, so every energy and force can be recomputed and checked."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("ase")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from ase.build import bulk  # noqa: E402
from ase.calculators.emt import EMT  # noqa: E402
from ase.calculators.singlepoint import SinglePointCalculator  # noqa: E402
from ase.io import write  # noqa: E402

from agentic_chimes.converters import units  # noqa: E402
from agentic_chimes.data_sources import colabfit, convert  # noqa: E402
from agentic_chimes.io import xyzf as xyzf_io  # noqa: E402
from agentic_chimes.stages import data_curate, data_fetch, data_generate, data_search, qe_relabel  # noqa: E402


def _labeled_atoms(n, rng, symbols=("Cu", "Au"), shear=False):
    out = []
    for k in range(n):
        a = bulk("Cu", "fcc", a=3.7, cubic=True).repeat((2, 2, 2))
        syms = [symbols[1] if rng.random() < (k % 3) * 0.25 else symbols[0] for _ in range(len(a))]
        a.set_chemical_symbols(syms)
        a.rattle(0.05, rng=rng)
        if shear and k % 2:
            a.set_cell(a.cell[:] @ np.array([[1, 0.07, 0], [0, 1, 0.03], [0, 0, 1]]), scale_atoms=True)
        a.calc = EMT()
        e, f = a.get_potential_energy(), a.get_forces()
        a.calc = SinglePointCalculator(a, energy=e, forces=f)
        out.append(a)
    return out


def _check_labels(frames):
    for fr in frames:
        a = convert.frame_to_atoms(fr)
        a.calc = EMT()
        assert a.get_potential_energy() == pytest.approx(convert.energy_ev(fr), abs=1e-5)
        assert np.abs(a.get_forces() - convert.forces_ev_ang(fr)).max() < 1e-5


# ---------------------------------------------------------------- convert

def test_to_frame_rotates_triclinic_cells_consistently():
    rng = np.random.default_rng(0)
    a = _labeled_atoms(1, rng)[0]
    R = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    R *= np.sign(np.linalg.det(R))
    cell = a.cell[:] @ np.array([[1, 0.1, 0], [0, 1, 0], [0.05, 0, 1]]) @ R.T
    frac = a.get_scaled_positions()
    a.set_cell(cell)
    a.set_scaled_positions(frac)
    a.calc = EMT()
    fr = convert.to_frame(a.get_chemical_symbols(), a.cell[:], a.positions, a.get_potential_energy(), a.get_forces())
    assert fr.non_ortho and np.allclose(np.triu(np.asarray(fr.box), 1), 0)
    _check_labels([fr])
    assert fr.energy == pytest.approx(units.ev_to_kcal_per_mol(a.get_potential_energy()))


# ---------------------------------------------------------------- search

CATALOG = [
    {"repo_id": "cf/CuZr", "name": "CuZr_md", "elements": ["Cu", "Zr"], "nconfigurations": 100, "atomic_forces_count": 100, "energy_count": 100, "methods": ["DFT-PBE"], "dimension_types": [[1, 1, 1]]},
    {"repo_id": "cf/Cu", "name": "Cu_only", "elements": ["Cu"], "nconfigurations": 50, "atomic_forces_count": 50, "energy_count": 50, "methods": ["DFT-PBE"], "dimension_types": [[1, 1, 1]]},
    {"repo_id": "cf/Big", "name": "OMat24_x", "elements": ["Cu", "Zr", "O", "Fe"] + [f"X{i}" for i in range(40)], "nconfigurations": 10**7, "atomic_forces_count": 1, "energy_count": 1, "methods": ["DFT-PBE+U"], "dimension_types": [[1, 1, 1]]},
    {"repo_id": "cf/Tern", "name": "CuZrAl", "elements": ["Cu", "Zr", "Al"], "nconfigurations": 10, "atomic_forces_count": 10, "energy_count": 10, "methods": ["DFT-PBE", "DFT-PBE+D3"], "dimension_types": [[1, 1, 1]]},
    {"repo_id": "cf/Mol", "name": "CuZr_molecules", "elements": ["Cu", "Zr"], "nconfigurations": 5, "atomic_forces_count": 5, "methods": ["DFT-PBE"], "dimension_types": [[0, 0, 0]]},
    {"repo_id": "cf/NoF", "name": "CuZr_energies", "elements": ["Cu", "Zr"], "nconfigurations": 5, "atomic_forces_count": 0, "methods": ["DFT-PBE"], "dimension_types": [[1, 1, 1]]},
    {"repo_id": "cf/Other", "name": "SiO", "elements": ["Si", "O"], "nconfigurations": 5, "atomic_forces_count": 5, "methods": ["DFT-PBE"], "dimension_types": [[1, 1, 1]]},
    {"repo_id": "cf/Broken", "catalog_error": "404"},
]


def test_search_classifies_and_ranks(monkeypatch, tmp_path):
    monkeypatch.setattr(colabfit, "load_catalog", lambda refresh=False: CATALOG)
    monkeypatch.setattr(colabfit, "_cache_dir", lambda: tmp_path)
    res = data_search.run(SimpleNamespace(elements=["Cu", "Zr"], methods=None, require_forces=True,
                                          periodic_only=True, include_overlap=False, limit=25, refresh=False))
    names = [r["name"] for r in res["results"]]
    assert names == ["CuZr_md", "Cu_only", "CuZrAl", "OMat24_x"]  # exact, subsystem, fewest extra elements first
    tern = res["results"][2]
    assert tern["n_extra_elements"] == 1 and any("mixed levels" in n for n in tern["notes"])
    assert any("far-from-equilibrium" in n for n in res["results"][3]["notes"])
    assert res["catalog"]["unreadable"] == 1

    only_u = data_search.run(SimpleNamespace(elements=["Cu", "Zr"], methods=["DFT-PBE+U"], require_forces=True,
                                             periodic_only=True, include_overlap=False, limit=25, refresh=False))
    assert [r["name"] for r in only_u["results"]] == ["OMat24_x"]


# ---------------------------------------------------------------- fetch (ColabFit schema, local parquet)

def _write_colabfit_parquet(path, atoms_list, methods):
    rows = {k: [] for k in ("configuration_id", "method", "software", "energy", "atomic_forces", "cell",
                            "positions", "pbc", "atomic_numbers", "elements")}
    for i, (a, m) in enumerate(zip(atoms_list, methods)):
        rows["configuration_id"].append(f"CO_{i}")
        rows["method"].append(m)
        rows["software"].append("VASP")
        rows["energy"].append(a.get_potential_energy())
        rows["atomic_forces"].append(a.get_forces().tolist())
        rows["cell"].append(a.cell[:].tolist())
        rows["positions"].append(a.positions.tolist())
        rows["pbc"].append([True, True, True])
        rows["atomic_numbers"].append(a.numbers.tolist())
        rows["elements"].append(sorted(set(a.get_chemical_symbols())))
    pq.write_table(pa.table(rows), path, row_group_size=7)


@pytest.fixture
def fake_colabfit(monkeypatch, tmp_path):
    rng = np.random.default_rng(3)
    cuau = _labeled_atoms(30, rng, shear=True)
    ni = _labeled_atoms(6, rng, symbols=("Ni", "Ni"))
    files = {"co/co_0.parquet": tmp_path / "co_0.parquet", "co/co_1.parquet": tmp_path / "co_1.parquet"}
    _write_colabfit_parquet(files["co/co_0.parquet"], cuau[:20] + ni, ["DFT-PBE"] * 26)
    _write_colabfit_parquet(files["co/co_1.parquet"], cuau[20:], ["DFT-PBE"] * 8 + ["DFT-PBE+U"] * 2)
    (tmp_path / "cache").mkdir()
    monkeypatch.setattr(colabfit, "_cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setattr(colabfit, "config_files", lambda repo: sorted(files))
    monkeypatch.setattr(colabfit, "_open_parquet", lambda repo, rel: pq.ParquetFile(files[rel]))
    monkeypatch.setattr(colabfit, "catalog_entry", lambda repo: {"repo_id": repo, "name": "fake", "license": "CC-BY-4.0", "doi": "10.x/y"})
    return tmp_path


def _fetch_args(out, **kw):
    base = dict(source="colabfit:cf/fake", elements=["Cu", "Au"], composition_rule="subset", method=None,
                max_frames=None, count_only=False, max_scan_files=None, label_policy="source",
                level_of_theory=None, seed=1, output_dir=str(out))
    base.update(kw)
    return SimpleNamespace(**base)


def test_fetch_count_only_reports_methods(fake_colabfit):
    res = data_fetch.run(_fetch_args(fake_colabfit / "o", count_only=True))
    assert res["n_matching"] == 30 and res["matching_by_method"] == {"DFT-PBE": 28, "DFT-PBE+U": 2}
    assert not (fake_colabfit / "o" / "pool.xyzf").exists()


def test_fetch_refuses_mixed_methods_then_fetches_one(fake_colabfit):
    with pytest.raises(ValueError, match="several levels of theory"):
        data_fetch.run(_fetch_args(fake_colabfit / "o"))
    res = data_fetch.run(_fetch_args(fake_colabfit / "o", method="DFT-PBE", max_frames=12))
    assert res["n_frames"] == 12 and res["n_matching"] == 28
    frames = xyzf_io.read_xyzf(res["pool_xyzf"])
    _check_labels(frames)
    prov = json.loads((fake_colabfit / "o" / "provenance.json").read_text())
    assert prov["level_of_theory"]["methods"] == {"DFT-PBE": 12}
    assert len(prov["frame_ids"]) == 12 and prov["source_metadata"]["license"] == "CC-BY-4.0"


def test_fetch_sampling_is_seeded_and_spans_files(fake_colabfit):
    a = data_fetch.run(_fetch_args(fake_colabfit / "a", method="DFT-PBE", max_frames=10, seed=5))
    b = data_fetch.run(_fetch_args(fake_colabfit / "b", method="DFT-PBE", max_frames=10, seed=5))
    ida = json.loads((fake_colabfit / "a" / "provenance.json").read_text())["frame_ids"]
    idb = json.loads((fake_colabfit / "b" / "provenance.json").read_text())["frame_ids"]
    assert ida == idb and a["n_frames"] == b["n_frames"] == 10


def test_fetch_exact_rule_and_relabel_policy(fake_colabfit):
    res = data_fetch.run(_fetch_args(fake_colabfit / "o", method="DFT-PBE", composition_rule="exact", label_policy="relabel"))
    frames = xyzf_io.read_xyzf(res["pool_xyzf"])
    assert all(set(f.symbols) == {"Cu", "Au"} for f in frames)
    assert all(f.energy is None and not np.any(f.forces) for f in frames)


def test_scan_skips_unreadable_files(fake_colabfit, monkeypatch):
    real_open = colabfit._open_parquet

    def flaky(repo, rel):
        if rel == "co/co_1.parquet":
            raise OSError("Couldn't deserialize thrift")
        return real_open(repo, rel)

    monkeypatch.setattr(colabfit, "_open_parquet", flaky)
    res = data_fetch.run(_fetch_args(fake_colabfit / "o", count_only=True))
    assert res["n_matching"] == 20 and res["unreadable_files"] == ["co/co_1.parquet"]
    assert "co/co_1.parquet" in colabfit.known_unreadable()["cf/fake"]


def test_fetch_local_extxyz(tmp_path):
    atoms = _labeled_atoms(8, np.random.default_rng(4))
    write(tmp_path / "d.extxyz", atoms)
    res = data_fetch.run(_fetch_args(tmp_path / "o", source=str(tmp_path / "d.extxyz"), level_of_theory="EMT"))
    assert res["n_frames"] == 8 and res["level_of_theory"]["methods"] == {"EMT": 8}
    _check_labels(xyzf_io.read_xyzf(res["pool_xyzf"]))


# ---------------------------------------------------------------- generate

def test_generate_orthorhombic_wide_cells_and_compositions(tmp_path):
    res = data_generate.run(SimpleNamespace(
        prototypes=[{"name": "Cu", "crystalstructure": "fcc", "a": 3.61}], seeds=None,
        compositions=[{"Cu": 0.75, "Au": 0.25}], min_width_ang=8.0, max_atoms=200,
        volume_strains=[-0.05, 0.05], rattle_std_ang=[0.1], n_rattle=3, seed=1, output_dir=str(tmp_path)))
    frames = xyzf_io.read_xyzf(res["structures_xyzf"])
    assert res["n_frames"] == 6 and res["n_non_orthorhombic"] == 0
    assert all(min(f.box) >= 8.0 * 0.95 ** (1 / 3) for f in frames)
    for f in frames:
        assert f.symbols.count("Au") == round(0.25 * f.natoms)
    prov = json.loads((tmp_path / "provenance.json").read_text())
    assert prov["label_policy"] == "relabel" and len(prov["frame_ids"]) == 6


def test_generate_skips_oversized_supercells(tmp_path):
    # a 3.61 A cubic Cu cell needs 3x3x3 = 108 atoms to reach 8 A widths
    with pytest.raises(ValueError, match="exceeds max_atoms"):
        data_generate.run(SimpleNamespace(
            prototypes=[{"name": "Cu", "crystalstructure": "fcc", "a": 3.61, "cubic": True}], seeds=None,
            compositions=None, min_width_ang=8.0, max_atoms=40, volume_strains=[0.0], rattle_std_ang=[0.1],
            n_rattle=1, seed=1, output_dir=str(tmp_path)))


# ---------------------------------------------------------------- curate

def _pool(tmp_path, name, atoms_list, method="DFT-PBE", source="colabfit:cf/fake", policy="source"):
    d = tmp_path / name
    d.mkdir()
    frames = [convert.to_frame(a.get_chemical_symbols(), a.cell[:], a.positions,
                               None if policy == "relabel" else a.get_potential_energy(),
                               None if policy == "relabel" else a.get_forces()) for a in atoms_list]
    xyzf_io.write_xyzf(frames, d / "pool.xyzf")
    (d / "provenance.json").write_text(json.dumps({
        "source": source, "label_policy": policy, "level_of_theory": {"methods": {method: len(frames)}, "software": {"VASP": len(frames)}},
        "frame_ids": [f"{name}_{i}" for i in range(len(frames))]}))
    return str(d / "pool.xyzf")


def _curate_args(frames, out, **kw):
    base = dict(frames=frames, elements=["Cu", "Au"], min_distance_ang=0.5, max_force_ev_ang=50.0,
                energy_outlier_mad=8.0, energy_outlier_floor=1.0, dedupe=True, natoms_min=None, natoms_max=None, require_orthorhombic=False,
                max_volume_ratio=3.0, coverage_cutoff_ang=6.0, target_size=None, holdout_fraction=0.25,
                allow_mixed_theory=False, seed=1, output_dir=str(out))
    base.update(kw)
    return SimpleNamespace(**base)


def _with_label(a, energy, forces):
    a.calc = SinglePointCalculator(a, energy=energy, forces=forces)
    return a


def test_curate_filters_every_planted_problem(tmp_path):
    rng = np.random.default_rng(7)
    good = _labeled_atoms(40, rng)
    close = good[0].copy()
    close.positions[1] = close.positions[0] + [0.3, 0, 0]
    close = _with_label(close, good[0].get_potential_energy(), good[0].get_forces())
    dup = _with_label(good[1].copy(), good[1].get_potential_energy(), good[1].get_forces())
    hot, shifted = good[2].copy(), good[3].copy()
    hot.positions += 0.01  # distinct geometry, or dedupe (correctly) removes them first
    shifted.positions += 0.01
    hot = _with_label(hot, good[2].get_potential_energy(), good[2].get_forces() * 0 + 80.0)
    shifted = _with_label(shifted, good[3].get_potential_energy() + 60.0, good[3].get_forces())
    cluster = bulk("Cu", "fcc", a=3.6, cubic=True)
    cluster.set_cell([20, 20, 20])
    cluster = _with_label(cluster, -10.0, np.zeros((4, 3)))
    lone = bulk("Cu", "sc", a=12.0)
    lone = _with_label(lone, -0.1, np.zeros((1, 3)))
    wrong_el = _with_label(bulk("Ni", "fcc", a=3.5, cubic=True), -20.0, np.zeros((4, 3)))

    pool = _pool(tmp_path, "p", good + [close, dup, hot, shifted, cluster, lone, wrong_el])
    res = data_curate.run(_curate_args([pool], tmp_path / "out"))
    assert res["removed_by_reason"] == {"min_distance": 1, "duplicate_of": 1, "max_force": 1, "energy_outlier": 1,
                                        "low_density": 1, "isolated_atom": 1, "elements_outside_target": 1}
    assert res["n_frames"] == 40 and res["n_train"] + res["n_holdout"] == 40

    manifest = json.loads((tmp_path / "out" / "data_manifest.json").read_text())
    assert manifest["labeled"] and manifest["level_of_theory"] == ["DFT-PBE (VASP)"]
    assert set(manifest["pairs"]) == {"Au-Au", "Au-Cu", "Cu-Cu"}
    assert manifest["fit_hints"]["fitener"] is True
    assert manifest["fit_hints"]["nlayers_required"]["8.0"] == 1  # 7.4 A cells
    report = json.loads((tmp_path / "out" / "curation_report.json").read_text())
    assert {r["frame_id"] for r in report["removed"]} >= {"p_40", "p_41", "p_46"}
    _check_labels(xyzf_io.read_xyzf(manifest["train_xyzf"]))


def test_curate_refuses_mixing_sources_and_theories(tmp_path):
    rng = np.random.default_rng(8)
    a = _pool(tmp_path, "a", _labeled_atoms(6, rng))
    b = _pool(tmp_path, "b", _labeled_atoms(6, rng), source="colabfit:cf/other")
    c = _pool(tmp_path, "c", _labeled_atoms(6, rng), method="DFT-PBE+U")
    with pytest.raises(ValueError, match="different datasets"):
        data_curate.run(_curate_args([a, b], tmp_path / "o1"))
    with pytest.raises(ValueError, match="different levels of theory"):
        data_curate.run(_curate_args([a, c], tmp_path / "o2"))
    res = data_curate.run(_curate_args([a, b], tmp_path / "o3", allow_mixed_theory=True, holdout_fraction=None))
    assert res["n_frames"] == 12


def test_curate_structure_only_pool_and_fps_target(tmp_path):
    pool = _pool(tmp_path, "s", _labeled_atoms(30, np.random.default_rng(9)), policy="relabel")
    res = data_curate.run(_curate_args([pool], tmp_path / "o", target_size=10, holdout_fraction=None))
    assert res["n_frames"] == 10
    assert any("label with qe-relabel" in w for w in res["warnings"])
    manifest = json.loads((tmp_path / "o" / "data_manifest.json").read_text())
    assert manifest["labeled"] is False and manifest["fit_hints"]["fitener"] is False


def test_nlayers_required():
    assert data_curate.nlayers_required(3.0, 7.0) == 0
    assert data_curate.nlayers_required(8.0, 2.0) == 4  # (2*4+1)*1.0 = 9 >= 8
    assert data_curate.nlayers_required(8.0, 7.4) == 1


# ---------------------------------------------------------------- qe-relabel provenance + kspacing

def test_kspacing_grid_scales_with_cell():
    small = xyzf_io.Frame(symbols=["Cu"], positions=[[0, 0, 0]], forces=[[0, 0, 0]], box=[2.5, 2.5, 5.0])
    big = xyzf_io.Frame(symbols=["Cu"], positions=[[0, 0, 0]], forces=[[0, 0, 0]], box=[10.0, 10.0, 10.0])
    assert qe_relabel.kgrid_from_spacing(small, 0.25) == [11, 11, 6]
    assert qe_relabel.kgrid_from_spacing(big, 0.25) == [3, 3, 3]


def test_qe_collect_writes_mergeable_provenance(tmp_path):
    from test_qe_relabel import SYNTHETIC_PW_OUT

    frames = [xyzf_io.Frame(symbols=["C", "H"], positions=[[0, 0, 0], [1.1 + 0.1 * i, 0, 0]], forces=[[0, 0, 0]] * 2,
                            box=[10.0, 10.0, 10.0]) for i in range(2)]
    pseudos = {}
    for el in ("C", "H"):
        (tmp_path / f"{el}.upf").write_text(f"pp {el}")
        pseudos[el] = str(tmp_path / f"{el}.upf")
    results = []
    for batch in ("r1", "r2"):
        (tmp_path / batch).mkdir()
        xyzf_io.write_xyzf(frames, tmp_path / batch / "structures.xyzf")
        out = tmp_path / batch / "qe"
        qe_relabel.run(SimpleNamespace(
            structure_xyzf=str(tmp_path / batch / "structures.xyzf"), frame_indices=None, elements=["C", "H"],
            masses={"C": 12.0, "H": 1.0}, pseudopotentials=pseudos, ecutwfc=60.0, ecutrho=None, kpoints=[1, 1, 1],
            kspacing=0.3, smearing="gaussian", degauss=0.01, conv_thr=1e-8, machine="dane", queue="debug",
            walltime_hours=1.0, nodes=1, ntasks_per_node=None, collect=None, dry_run=True, output_dir=str(out)))
        for fd in ("frame_0000", "frame_0001"):
            (out / fd / "pw.out").write_text(SYNTHETIC_PW_OUT)
        results.append(qe_relabel.run(SimpleNamespace(collect=str(out))))

    provs = [json.loads(open(r["provenance"]).read()) for r in results]
    assert provs[0]["source"] == provs[1]["source"]  # same settings -> same key
    assert provs[0]["dft_settings"]["kspacing_inv_ang"] == 0.3
    res = data_curate.run(_curate_args([r["labeled_xyzf"] for r in results], tmp_path / "cur", elements=["C", "H"],
                                       holdout_fraction=None, energy_outlier_mad=None, dedupe=False,
                                       max_volume_ratio=None))
    assert res["n_frames"] == 4 and res["level_of_theory"][0].startswith("DFT-QE")
