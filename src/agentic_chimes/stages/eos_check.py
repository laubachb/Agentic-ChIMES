"""Equation of state and elastic constants of a structure under a model.

Published ChIMES validation goes beyond MD stability to the equation of
state and elastic response (Lindsey et al. 2017, 2019, 2025). This stage
measures both for an ideal structure, with chimes_calculator on exact
supercells; it is local and takes seconds:

- **E(V)**: isotropic strain of the cell over +-`volume_range`, fitted with
  third-order Birch-Murnaghan, giving V0 (A^3/atom), E0, B0 (GPa) and B0'.
  The pressure from the model's stress tensor at each volume is reported
  next to -dE/dV as an internal consistency check.
- **Elastic constants**: the cell is relaxed isotropically to V0, then
  +-`strain` is applied in each Voigt direction. C_ij = d sigma_i / d eps_j
  from the model's stress (Cauchy sign), in GPa. These are clamped-ion
  constants (no internal relaxation). They are exact for lattices whose
  atoms all sit on inversion centres (fcc, bcc, B2), approximate otherwise.
- **Mechanical stability**: the 6x6 tensor must be positive definite (Born).
  Cubic shortcuts are reported when the structure is cubic.

With `reference_xyzf`, DFT frames of the same composition are compared on
pressure (model vs DFT stress), when they carry stresses.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..io import atomic, fs
from ..io import xyzf as xyzf_io

NAME = "eos-check"
SUMMARY = "Equation of state (Birch-Murnaghan V0, B0, B0') and clamped-ion elastic constants of a structure under a model."
SCHEMA = {
    "type": "object",
    "required": ["params"],
    "properties": {
        "params": {"type": "string"},
        "structure_xyzf": {"type": ["string", "null"]},
        "frame_index": {"type": "integer", "default": 0},
        "prototype": {"type": ["object", "null"], "description": "ase.build.bulk kwargs, e.g. {\"name\":\"Cu\",\"crystalstructure\":\"fcc\",\"a\":3.61}."},
        "volume_range": {"type": "number", "default": 0.08, "description": "Fractional volume change each way."},
        "n_points": {"type": "integer", "default": 11},
        "strain": {"type": "number", "default": 0.005},
        "reference_xyzf": {"type": ["string", "null"], "description": "DFT frames (with stresses) to compare pressures against."},
        "plot": {"type": "boolean", "default": True},
    },
}

GPA = 6.9479  # kcal/mol/A^3 -> GPa (converters.units.CHIMES_STRESS_TO_GPA)


def add_arguments(parser) -> None:
    import json

    parser.add_argument("--params", default=None)
    parser.add_argument("--structure-xyzf", dest="structure_xyzf", default=None)
    parser.add_argument("--frame-index", dest="frame_index", type=int, default=0)
    parser.add_argument("--prototype", type=json.loads, default=None)
    parser.add_argument("--volume-range", dest="volume_range", type=float, default=0.08)
    parser.add_argument("--n-points", dest="n_points", type=int, default=11)
    parser.add_argument("--strain", type=float, default=0.005)
    parser.add_argument("--reference-xyzf", dest="reference_xyzf", default=None)
    parser.add_argument("--no-plot", dest="plot", action="store_false", default=True)


def birch_murnaghan(v, e):
    """Fit E(V) = E0 + 9 V0 B0/16 [ (x-1)^3 B0' + (x-1)^2 (6 - 4 x) ], x = (V0/V)^(2/3).
    Returns (V0, E0, B0 [energy/volume units], B0p)."""
    v, e = np.asarray(v, float), np.asarray(e, float)
    # initial guess from a parabola in V
    c = np.polyfit(v, e, 2)
    v0 = -c[1] / (2 * c[0])
    b0 = 2 * c[0] * v0
    p = np.array([v0, float(np.polyval(c, v0)), b0, 4.0])

    def model(p, v):
        x = (p[0] / v) ** (2.0 / 3.0)
        return p[1] + 9.0 * p[0] * p[2] / 16.0 * ((x - 1) ** 3 * p[3] + (x - 1) ** 2 * (6 - 4 * x))

    for _ in range(200):  # Gauss-Newton with numeric Jacobian
        r = model(p, v) - e
        J = np.empty((len(v), 4))
        for k in range(4):
            dp = np.zeros(4)
            dp[k] = 1e-6 * max(abs(p[k]), 1e-3)
            J[:, k] = (model(p + dp, v) - model(p - dp, v)) / (2 * dp[k])
        step, *_ = np.linalg.lstsq(J, -r, rcond=None)
        p = p + step
        if np.max(np.abs(step) / np.maximum(np.abs(p), 1e-9)) < 1e-10:
            break
    rms = float(np.sqrt(np.mean((model(p, v) - e) ** 2)))
    return p, rms


def _frame(args):
    from ..data_sources import convert

    if getattr(args, "prototype", None):
        from .benchmark import _base_atoms

        at = _base_atoms(args)
        return convert.to_frame(at.get_chemical_symbols(), at.cell.array, at.positions)
    if getattr(args, "structure_xyzf", None):
        return xyzf_io.read_xyzf(args.structure_xyzf)[getattr(args, "frame_index", 0) or 0]
    raise ValueError("eos-check needs --prototype or --structure-xyzf")


def _deform(frame, F):
    cell = np.asarray(frame.box if frame.non_ortho else np.diag(frame.box), dtype=float)
    new_cell = cell @ F.T
    pos = np.asarray(frame.positions, float) @ F.T
    return xyzf_io.Frame(symbols=list(frame.symbols), positions=pos.tolist(), forces=[[0.0] * 3] * frame.natoms,
                         box=new_cell.tolist(), non_ortho=True)


def _volume(frame):
    cell = np.asarray(frame.box if frame.non_ortho else np.diag(frame.box), dtype=float)
    return abs(np.linalg.det(cell))


def run(args) -> dict:
    from . import evaluate

    if not getattr(args, "params", None):
        raise ValueError("eos-check needs --params")
    params = str(Path(args.params).resolve())
    frame = _frame(args)
    evaluate.check_frames_against_models([frame], [params])
    w = evaluate._load_wrapper()
    ptr = w.chimes_open_instance()
    w.set_chimes_instance(ptr, small=False)
    w.init_chimes_instance(ptr, params, 0)
    cut = evaluate.max_outer_cutoff(params)
    n = frame.natoms

    def calc(fr):
        e, _f, s = evaluate.predict(w, ptr, fr, cut, with_stress=True)  # s: xyzf order, GPa, pressure sign
        return e, np.array(s)

    try:
        # ---- E(V)
        vr = float(getattr(args, "volume_range", 0.08) or 0.08)
        npts = max(5, int(getattr(args, "n_points", 11) or 11))
        rows = []
        for f in np.linspace(1 - vr, 1 + vr, npts):
            fr = _deform(frame, np.eye(3) * f ** (1 / 3))
            e, s = calc(fr)
            rows.append((_volume(fr) / n, e / n, float(s[:3].mean())))
        v, e, p = (np.array(c) for c in zip(*rows))
        bm, rms = birch_murnaghan(v, e)
        v0, e0, b0, b0p = bm
        dedv = np.gradient(e, v)                       # kcal/mol/A^3 per atom-volume, -dE/dV = P
        p_from_e = -dedv * GPA
        eos = {"V0_A3_per_atom": round(float(v0), 4), "E0_kcal_mol_per_atom": round(float(e0), 4),
               "B0_GPa": round(float(b0 * GPA), 2), "B0_prime": round(float(b0p), 2), "fit_rms_kcal_mol_atom": rms,
               "points": [{"V_per_atom": round(float(a), 4), "E_per_atom": round(float(b), 5), "P_GPa_stress": round(float(c), 3),
                           "P_GPa_dEdV": round(float(d), 3)} for a, b, c, d in zip(v, e, p, p_from_e)],
               "stress_vs_dEdV_max_abs_GPa": round(float(np.max(np.abs(p - p_from_e)[1:-1])), 3)}
        notes = []
        if not (v.min() < v0 < v.max()):
            notes.append("V0 lies outside the scanned volumes: the structure is far from equilibrium for this "
                         "model; widen volume_range or check the structure")
        if b0 <= 0:
            notes.append("non-positive bulk modulus: the model's E(V) has no minimum here")

        # ---- elastic constants at V0
        f0 = (v0 / (_volume(frame) / n)) ** (1 / 3)
        fr0 = _deform(frame, np.eye(3) * f0)
        eps = float(getattr(args, "strain", 0.005) or 0.005)
        voigt = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
        order = [0, 1, 2, 5, 4, 3]  # xyzf stress order xx yy zz xy xz yz -> Voigt xx yy zz yz xz xy
        C = np.zeros((6, 6))
        for j, (a, b) in enumerate(voigt):
            sig = []
            for sgn in (1, -1):
                E_ = np.zeros((3, 3))
                if a == b:
                    E_[a, a] = sgn * eps
                else:
                    E_[a, b] = E_[b, a] = sgn * eps / 2  # engineering shear strain = eps
                _, s = calc(_deform(fr0, np.eye(3) + E_))
                sig.append(-s[order])                   # Cauchy stress, GPa
            C[:, j] = (sig[0] - sig[1]) / (2 * eps)
        C = 0.5 * (C + C.T)
        eig = np.linalg.eigvalsh(C)
        elastic = {"C_GPa": np.round(C, 2).tolist(), "eigenvalues_GPa": np.round(eig, 2).tolist(),
                   "born_stable": bool(eig.min() > 0), "bulk_modulus_voigt_GPa": round(float(C[:3, :3].sum() / 9), 2)}
        if np.allclose(np.diag(C)[:3], C[0, 0], rtol=0.05) and np.allclose(np.diag(C)[3:], C[3, 3], rtol=0.05):
            elastic["cubic"] = {"C11": round(float(C[0, 0]), 2), "C12": round(float(np.mean([C[0, 1], C[0, 2], C[1, 2]])), 2),
                                "C44": round(float(C[3, 3]), 2)}
        if not elastic["born_stable"]:
            notes.append("elastic tensor is not positive definite: the structure is mechanically unstable under this model")
        result = {"natoms": n, "composition": "-".join(sorted(set(frame.symbols))), "eos": eos, "elastic": elastic,
                  "notes": notes + ["clamped-ion elastic constants (no internal relaxation)"]}

        ref = getattr(args, "reference_xyzf", None)
        if ref:
            comp = set(frame.symbols)
            pairs = []
            for rf in xyzf_io.read_xyzf(ref):
                if set(rf.symbols) == comp and rf.stress is not None:
                    _e, s = calc(rf)
                    pairs.append((float(np.mean(rf.stress[:3])), float(s[:3].mean())))
            if pairs:
                d, m = (np.array(c) for c in zip(*pairs))
                result["reference_pressure"] = {"n_frames": len(pairs), "rmse_GPa": round(float(np.sqrt(np.mean((m - d) ** 2))), 3),
                                                "bias_GPa": round(float(np.mean(m - d)), 3)}
    finally:
        w.chimes_close_instance(ptr)
    out = Path(getattr(args, "output_dir", None) or ".")
    fs.ensure_dir(out)
    if getattr(args, "plot", True):
        from ..io import plots

        pts = result["eos"]["points"]
        fig, ax = plots.figure((6.5, 4.5))
        if fig is not None:
            v = [p["V_per_atom"] for p in pts]
            ax.plot(v, [p["E_per_atom"] for p in pts], "o", label="model E(V)")
            vv = np.linspace(min(v), max(v), 100)
            x = (v0 / vv) ** (2 / 3)
            ax.plot(vv, e0 + 9 * v0 * b0 / 16 * ((x - 1) ** 3 * b0p + (x - 1) ** 2 * (6 - 4 * x)), "-", lw=1,
                    label=f"Birch-Murnaghan: V0 {v0:.2f} Å³, B0 {b0 * GPA:.0f} GPa")
            ax.set_xlabel("volume per atom (Å³)")
            ax.set_ylabel("energy per atom (kcal/mol)")
            ax.set_title(f"Equation of state: {result['composition']}")
            ax.legend(fontsize=8)
            result["plot"] = plots.save(fig, out / "eos.png")
    atomic.write_json(out / "eos_check.json", result, indent=1)
    return result
