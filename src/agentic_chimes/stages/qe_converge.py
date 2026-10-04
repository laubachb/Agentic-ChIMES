"""Convergence tests for the Quantum ESPRESSO settings a study will label with.

Labels are only as good as the DFT settings behind them, and every
active-learning round must reuse the base set's settings, so they have to be
right before the first batch. This stage runs two one-dimensional scans on
one representative frame:

- `ecutwfc` ladder (at the middle `kspacing`);
- `kspacing` ladder (at the largest `ecutwfc`).

`--collect` compares each setting with the finest one in its scan on the
quantities ChIMES fits: energy per atom, the largest force-component change,
and pressure. It recommends the cheapest setting that meets every threshold
(defaults 1 meV/atom, 10 meV/Å, 0.1 GPa, the usual MLIP labeling tolerances)
and reports the tables and a plot. Submission mirrors `qe-relabel` (one
Slurm job, `pw.x` per directory), so `--dry-run` previews the exact job.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .. import config, hpc, machines
from ..converters import qe2xyzf, units
from ..io import atomic, fs
from ..io import xyzf as xyzf_io

NAME = "qe-converge"
SUMMARY = "Converge QE ecutwfc and k-point spacing on one frame (energy/atom, forces, pressure) before labeling with them."
SUPPORTS_DRY_RUN = True
SCHEMA = {
    "type": "object",
    "properties": {
        "structure_xyzf": {"type": ["string", "null"], "description": "A representative frame (the densest / most compressed one is the harder test)."},
        "frame_index": {"type": "integer", "default": 0},
        "elements": {"type": ["array", "null"], "items": {"type": "string"}},
        "masses": {"type": ["object", "null"]},
        "pseudopotentials": {"type": ["object", "null"], "description": "element -> UPF path, as for qe-relabel."},
        "ecutwfc_values": {"type": "array", "items": {"type": "number"}, "default": [30, 40, 50, 60, 80], "description": "Ry"},
        "kspacing_values": {"type": "array", "items": {"type": "number"}, "default": [0.5, 0.35, 0.25, 0.18, 0.12], "description": "1/Å (smaller = denser)"},
        "ecutrho_factor": {"type": "number", "default": 4.0},
        "smearing": {"type": "string", "default": "gaussian"},
        "degauss": {"type": "number", "default": 0.01},
        "conv_thr": {"type": "number", "default": 1.0e-8},
        "machine": {"type": ["string", "null"]},
        "queue": {"type": "string", "default": "debug"},
        "walltime_hours": {"type": "number", "default": 2.0},
        "nodes": {"type": "integer", "default": 1},
        "ntasks_per_node": {"type": ["integer", "null"]},
        "collect": {"type": ["string", "null"], "description": "A submitted qe-converge directory: parse and recommend."},
        "tol_energy_mev_atom": {"type": "number", "default": 1.0},
        "tol_force_mev_ang": {"type": "number", "default": 10.0},
        "tol_pressure_gpa": {"type": "number", "default": 0.1},
    },
}
_MANIFEST = "qe_converge_manifest.json"


def add_arguments(parser) -> None:
    parser.add_argument("--structure-xyzf", dest="structure_xyzf", default=None)
    parser.add_argument("--frame-index", dest="frame_index", type=int, default=0)
    parser.add_argument("--elements", type=lambda s: [x for x in s.split(",") if x], default=None)
    parser.add_argument("--masses", type=json.loads, default=None)
    parser.add_argument("--pseudopotentials", type=json.loads, default=None)
    parser.add_argument("--ecutwfc-values", dest="ecutwfc_values", type=lambda s: [float(x) for x in s.split(",") if x], default=None)
    parser.add_argument("--kspacing-values", dest="kspacing_values", type=lambda s: [float(x) for x in s.split(",") if x], default=None)
    parser.add_argument("--ecutrho-factor", dest="ecutrho_factor", type=float, default=4.0)
    parser.add_argument("--smearing", default="gaussian")
    parser.add_argument("--degauss", type=float, default=0.01)
    parser.add_argument("--conv-thr", dest="conv_thr", type=float, default=1.0e-8)
    parser.add_argument("--machine", default=None)
    parser.add_argument("--queue", default="debug")
    parser.add_argument("--walltime-hours", dest="walltime_hours", type=float, default=2.0)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--ntasks-per-node", dest="ntasks_per_node", type=int, default=None)
    parser.add_argument("--collect", default=None)
    parser.add_argument("--tol-energy-mev-atom", dest="tol_energy_mev_atom", type=float, default=1.0)
    parser.add_argument("--tol-force-mev-ang", dest="tol_force_mev_ang", type=float, default=10.0)
    parser.add_argument("--tol-pressure-gpa", dest="tol_pressure_gpa", type=float, default=0.1)


def scan_points(ecuts, kspacings) -> list:
    """[(name, ecutwfc, kspacing)]: the ecutwfc ladder at the middle kspacing, the kspacing ladder at the largest ecutwfc."""
    ecuts, ks = sorted(set(ecuts)), sorted(set(kspacings), reverse=True)
    k_mid = ks[len(ks) // 2]
    pts = [(f"ecut_{e:g}", e, k_mid) for e in ecuts]
    pts += [(f"kspacing_{k:g}", ecuts[-1], k) for k in ks if (ecuts[-1], k) != (ecuts[-1], k_mid) or True]
    seen, out = set(), []
    for name, e, k in pts:
        if (e, k) in seen:
            continue
        seen.add((e, k))
        out.append((name, e, k))
    return out


def _submit(args) -> dict:
    from .qe_relabel import _render_pw_in, kgrid_from_spacing

    for req in ("structure_xyzf", "elements", "masses", "pseudopotentials", "machine"):
        if not getattr(args, req, None):
            raise ValueError(f"qe-converge requires --{req.replace('_', '-')}")
    frame = xyzf_io.read_xyzf(args.structure_xyzf)[getattr(args, "frame_index", 0) or 0]
    work = fs.ensure_dir(Path(getattr(args, "output_dir", None) or ".").resolve())
    ecuts = getattr(args, "ecutwfc_values", None) or [30, 40, 50, 60, 80]
    ks = getattr(args, "kspacing_values", None) or [0.5, 0.35, 0.25, 0.18, 0.12]
    points = scan_points(ecuts, ks)
    commands, dirs = [], []
    for name, e, k in points:
        d = fs.ensure_dir(work / name)
        dirs.append({"name": name, "dir": str(d), "ecutwfc": e, "kspacing": k, "kpoints": kgrid_from_spacing(frame, k)})
        (d / "pw.in").write_text(_render_pw_in(frame, args.elements, args.masses, args.pseudopotentials, ecutwfc=e,
                                                ecutrho=e * float(getattr(args, "ecutrho_factor", 4.0) or 4.0),
                                                kpoints=kgrid_from_spacing(frame, k), smearing=args.smearing,
                                                degauss=args.degauss, conv_thr=args.conv_thr))
        for el in args.elements:
            src = Path(args.pseudopotentials[el]).resolve()
            if not (d / src.name).exists():
                (d / src.name).symlink_to(src)
        commands += [f"cd {d}", "pw.x -in pw.in > pw.out 2> pw.err", "cd -"]
    manifest = {"frame": {"natoms": frame.natoms, "symbols": frame.symbols}, "points": dirs, "ecutwfc_values": ecuts,
                "kspacing_values": ks, "structure_xyzf": str(Path(args.structure_xyzf).resolve())}
    atomic.write_json(work / _MANIFEST, manifest, indent=1)
    profile = machines.load_profile(args.machine)
    dry = bool(getattr(args, "dry_run", False))
    pw_bin = config.resolve_component("qe_pw_bin", required=not dry)
    pw_dir = Path(pw_bin).parent if pw_bin else "<qe_pw_bin not yet built; run chimes-agent setup --component quantum_espresso>"
    handle = hpc.submit_job(profile, job_name="qe-converge", commands=[f"export PATH={pw_dir}:$PATH"] + commands, work_dir=work,
                            nodes=getattr(args, "nodes", 1) or 1, ntasks_per_node=getattr(args, "ntasks_per_node", None),
                            walltime_hours=getattr(args, "walltime_hours", 2.0) or 2.0, queue=getattr(args, "queue", "debug") or "debug",
                            dry_run=dry, expect=[str(Path(p["dir"]) / "pw.out") for p in dirs])
    return {"work_dir": str(work), "n_points": len(points), "points": dirs, "job_id": handle.job_id, "dry_run": handle.dry_run,
            "job_file": str(handle.job_file), "next": f"when the job is done: chimes-agent qe-converge --collect {work}"}


def _collect(args) -> dict:
    work = Path(args.collect).resolve()
    manifest = atomic.read_json(work / _MANIFEST)
    if not manifest:
        raise ValueError(f"{work} holds no {_MANIFEST}; is this a qe-converge directory?")
    natoms = manifest["frame"]["natoms"]
    rows = []
    for p in manifest["points"]:
        out = Path(p["dir"]) / "pw.out"
        row = {**p, "status": "missing"}
        if out.is_file():
            r = qe2xyzf.parse_pwx_output(out.read_text())
            if r.energy_ry is None or r.forces_ry_bohr is None:
                row["status"] = "failed"
            else:
                row.update({"status": "converged" if r.converged else "not_converged",
                            "energy_ev_atom": units.RY_TO_EV * r.energy_ry / natoms,
                            "forces_ev_ang": (np.asarray(r.forces_ry_bohr) * units.RY_TO_EV / 0.52917721067).tolist(),
                            "pressure_gpa": (sum(r.stress_kbar[i][i] for i in range(3)) / 3 * units.KBAR_TO_GPA) if r.stress_kbar else None})
        rows.append(row)
    tol_e, tol_f, tol_p = (float(getattr(args, k, d) or d) for k, d in (("tol_energy_mev_atom", 1.0), ("tol_force_mev_ang", 10.0), ("tol_pressure_gpa", 0.1)))

    e_max = max(manifest["ecutwfc_values"])
    ks_sorted = sorted(set(manifest["kspacing_values"]), reverse=True)
    k_mid = ks_sorted[len(ks_sorted) // 2]

    def scan(prefix, key):
        # select by setting values, not names: the point shared by both ladders carries one name only
        fixed = (lambda r: r["kspacing"] == k_mid) if key == "ecutwfc" else (lambda r: r["ecutwfc"] == e_max)
        pts = [r for r in rows if fixed(r) and r["status"] == "converged"]
        pts.sort(key=lambda r: r[key], reverse=(key == "kspacing"))
        if len(pts) < 2:
            return {"points": pts, "recommended": None, "note": f"fewer than two converged {prefix} points"}
        ref = pts[-1]
        table, rec = [], None
        for r in pts:
            de = abs(r["energy_ev_atom"] - ref["energy_ev_atom"]) * 1000
            df = float(np.abs(np.asarray(r["forces_ev_ang"]) - np.asarray(ref["forces_ev_ang"])).max()) * 1000
            dp = abs(r["pressure_gpa"] - ref["pressure_gpa"]) if r.get("pressure_gpa") is not None and ref.get("pressure_gpa") is not None else None
            ok = de <= tol_e and df <= tol_f and (dp is None or dp <= tol_p)
            table.append({key: r[key], "dE_meV_atom": round(de, 3), "dF_max_meV_ang": round(df, 2),
                          "dP_GPa": None if dp is None else round(dp, 4), "converged_vs_finest": ok})
            if ok and rec is None and r is not ref:
                rec = r[key]
        if rec is None:
            rec = ref[key]
            note = f"only the finest {key} ({rec}) meets the thresholds: extend the scan"
        else:
            note = f"{key} {rec} is the cheapest setting within {tol_e} meV/atom, {tol_f} meV/Å and {tol_p} GPa of the finest"
        return {"table": table, "recommended": rec, "reference": ref[key], "note": note}

    result = {"work_dir": str(work), "natoms": natoms, "n_points": len(rows),
              "status_counts": {s: sum(r["status"] == s for r in rows) for s in ("converged", "not_converged", "failed", "missing")},
              "ecutwfc": scan("ecut_", "ecutwfc"), "kspacing": scan("kspacing_", "kspacing"),
              "thresholds": {"energy_mev_atom": tol_e, "force_mev_ang": tol_f, "pressure_gpa": tol_p}}
    result["recommended"] = {"ecutwfc": result["ecutwfc"].get("recommended"), "kspacing": result["kspacing"].get("recommended"),
                             "ecutrho": None if result["ecutwfc"].get("recommended") is None else 4 * result["ecutwfc"]["recommended"]}
    from ..io import plots

    series = {}
    for key in ("ecutwfc", "kspacing"):
        t = result[key].get("table") or []
        if t:
            series[f"{key}: dE (meV/atom)"] = ([r[key] for r in t], [r["dE_meV_atom"] for r in t])
    if series:
        png = plots.lines(series, work / "convergence.png", xlabel="setting", ylabel="change vs finest", title="QE convergence")
        if png:
            result["plot"] = png
    atomic.write_json(work / "qe_converge.json", result, indent=1)
    return result


def run(args) -> dict:
    if getattr(args, "collect", None):
        return _collect(args)
    return _submit(args)
