"""Shared machinery for the hyperparameter phase: distance/RDF analysis of a
training set (any cell shape), the single-fit runner, and model selection.

ChIMES guidance this encodes (codes/chimes_lsq-LLfork/doc/source/
lsq_input_file.rst, quick_start.rst; docs/concepts/cutoffs_and_lambdas.md):
  - S_MINIM: just below the smallest sampled distance of the pair
    (documented delta 0.002-0.02 A). Below it the model has no data.
  - MORSE_LAMBDA: the pair's first RDF peak.
  - S_MAXIM: 2-body ~ second shell (typically 5-8 A); 3-body the first
    non-bonded shell (first RDF minimum); 4-body between the first and second
    minimum. Outer cutoffs set cost as much as accuracy, so they are searched,
    not fixed.
  - Orders: start near 12/7/3; choose by holdout error.

Distances come from ASE neighbour lists, which enumerate every periodic image
within the radius -- correct for triclinic cells thinner than the cutoff
(most open-database frames), unlike minimum-image.
"""

from __future__ import annotations

import hashlib
import json
import math
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
from ..io import fs

BIN = 0.02
SMOOTH_SIGMA = 0.08


def pair_key(a: str, b: str) -> str:
    return "-".join(sorted((a, b)))


def all_pairs(elements) -> list:
    return [pair_key(a, b) for a, b in combinations_with_replacement(sorted(elements), 2)]


# ------------------------------------------------------------------ analysis

def _widths(cell) -> list:
    cell = np.asarray(cell, dtype=float)
    vol = abs(np.linalg.det(cell))
    return [vol / np.linalg.norm(np.cross(cell[(k + 1) % 3], cell[(k + 2) % 3])) for k in range(3)]


def nlayers_required(s_maxim: float, thinnest_width: float) -> int:
    """chimes_lsq needs s_maxim <= (2*N_LAYERS+1) * width / 2 (IS_RCUT_SAFE)."""
    return max(0, math.ceil((2 * s_maxim / thinnest_width - 1) / 2))


def _smooth(y, sigma_bins):
    half = int(4 * sigma_bins) + 1
    x = np.arange(-half, half + 1)
    k = np.exp(-0.5 * (x / sigma_bins) ** 2)
    return np.convolve(y, k / k.sum(), mode="same")


def _extrema(r, g, rmin):
    """(peaks, minima) positions of a smoothed shape curve beyond rmin, keeping
    only features with prominence >= 5% of the curve maximum."""
    from scipy.signal import find_peaks

    # start just below rmin: in a perfect crystal the first peak sits exactly
    # at the minimum distance (smoothing spreads it by ~3 sigma)
    mask = r > rmin - 3 * SMOOTH_SIGMA
    rr, gg = r[mask], g[mask]
    if gg.size < 5 or gg.max() <= 0:
        return [], []
    prom = 0.05 * gg.max()
    pk, _ = find_peaks(gg, prominence=prom)
    mn, _ = find_peaks(-gg, prominence=prom)
    return [float(rr[i]) for i in pk], [float(rr[i]) for i in mn]


def analyze(frames, elements, r_max: float = 8.0, s_minim_delta: float = 0.02) -> dict:
    """Per-pair distance statistics, RDF features and cutoff/lambda
    suggestions for a list of xyzf Frames."""
    from ..data_sources import convert

    edges = np.arange(0.0, r_max + BIN, BIN)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist = {p: np.zeros(len(centers)) for p in all_pairs(elements)}
    mins = {p: math.inf for p in hist}
    widths = []
    n_atoms_total = 0
    n_energy = 0
    for fr in frames:
        atoms = convert.frame_to_atoms(fr)
        widths.append(min(_widths(atoms.cell[:])))
        n_atoms_total += fr.natoms
        n_energy += fr.energy is not None
        i, j, d = convert.unique_pairs(atoms, r_max)
        sym = np.asarray(fr.symbols)
        for p in hist:
            a, b = p.split("-")
            sel = ((sym[i] == a) & (sym[j] == b)) | ((sym[i] == b) & (sym[j] == a))
            if sel.any():
                ds = d[sel]
                hist[p] += np.histogram(ds, bins=edges)[0]
                mins[p] = min(mins[p], float(ds.min()))

    pairs = {}
    for p, h in hist.items():
        if not h.any():
            pairs[p] = {"sampled": False}
            continue
        shape = _smooth(h / np.maximum(centers, BIN) ** 2, SMOOTH_SIGMA / BIN)
        peaks, minima = _extrema(centers, shape, mins[p])
        first_peak = peaks[0] if peaks else None
        after = [m for m in minima if first_peak is not None and m > first_peak]
        second_after = [m for m in after[1:]] if len(after) > 1 else []
        cum = np.cumsum(h)
        near = int(h[centers < 1.2 * mins[p]].sum())
        pairs[p] = {
            "sampled": True,
            "min_distance": round(mins[p], 4),
            "p01_distance": round(float(centers[np.searchsorted(cum, 0.01 * cum[-1])]), 3),
            "n_distances": int(h.sum()),
            "n_within_1.2x_min": near,
            "rdf_peaks": [round(x, 3) for x in peaks[:4]],
            "rdf_minima": [round(x, 3) for x in after[:3]],
            "suggested": {
                "s_minim": round(mins[p] - s_minim_delta, 3),
                "morse_lambda": round(first_peak if first_peak else 1.1 * mins[p], 3),
                "first_shell_end": round(after[0], 3) if after else None,
                "second_shell_end": round(second_after[0], 3) if second_after else None,
            },
        }

    sampled = {p: v for p, v in pairs.items() if v["sampled"]}
    first_shells = [v["suggested"]["first_shell_end"] for v in sampled.values() if v["suggested"]["first_shell_end"]]
    second_shells = [v["suggested"]["second_shell_end"] for v in sampled.values() if v["suggested"]["second_shell_end"]]
    thinnest = float(min(widths))
    shell1 = max(first_shells) if first_shells else None
    shell2 = max(second_shells) if second_shells else None

    cand_2b = sorted({round(x, 2) for x in [5.0, 6.0, 7.0, 8.0] + ([shell2] if shell2 else []) if x <= r_max})
    # (2-body candidates below the largest many-body candidate are still valid;
    # hyper-search only pairs a many-body cutoff with a 2-body cutoff >= it)
    # Many-body cutoffs span first shell -> second shell. ChIMES' cubic smoothing
    # multiplies one (1 - r/r_c)^3 per cluster distance, so a cutoff at the first
    # shell leaves 3-/4-body terms almost no signal (measured on Cu-Zr:
    # docs/concepts/cutoffs_and_lambdas.md); the search weighs signal against cost.
    far = shell2 if shell2 and shell1 and shell2 > shell1 + 0.5 else (shell1 + 2.0 if shell1 else None)
    cand_3b = sorted({round(min(x, r_max), 2) for x in ([shell1, (shell1 + far) / 2, far] if shell1 else [4.0, 5.0, 6.0])})
    cand_4b = sorted({round(min(x, r_max), 2) for x in ([shell1, (shell1 + far) / 2] if shell1 else [4.0, 5.0])})
    all_cands = sorted(set(cand_2b + cand_3b + cand_4b))

    notes = []
    for p, v in pairs.items():
        if not v["sampled"]:
            notes.append(f"{p} never occurs within {r_max} A: its terms are unconstrained; fix the data before fitting")
        elif v["n_within_1.2x_min"] < 20:
            notes.append(f"{p}: only {v['n_within_1.2x_min']} distances near its minimum; the inner cutoff sits on sparse data")
        elif not v["rdf_peaks"]:
            notes.append(f"{p}: no clear first RDF peak; morse_lambda falls back to 1.1 x min distance")
    if shell1 is None:
        notes.append("no first-shell minimum found for any pair; 3-/4-body cutoff candidates fall back to 4-5 A")

    return {
        "elements": sorted(elements),
        "n_frames": len(frames),
        "n_force_equations": 3 * n_atoms_total,
        "n_energy_equations": n_energy,
        "thinnest_cell_width": round(thinnest, 3),
        "pairs": pairs,
        "global": {"first_shell_end": shell1, "second_shell_end": shell2},
        "candidates": {"s_maxim_2b": cand_2b, "s_maxim_3b": cand_3b, "s_maxim_4b": cand_4b},
        "nlayers_required": {f"{c:.2f}": nlayers_required(c, thinnest) for c in all_cands},
        "notes": notes,
    }


# ------------------------------------------------------------------ one fit

def config_key(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def scaled_lambdas(cfg: dict) -> dict:
    """Morse lambdas after the global scale and any per-pair scales."""
    pairs = cfg.get("lambda_scale_pairs") or {}
    return {p: round(cfg["morse_lambda"][p] * cfg.get("lambda_scale", 1.0) * pairs.get(p, 1.0), 4) for p in cfg["morse_lambda"]}


def cv_assignment(frames, k: int, seed: int = 0) -> list:
    """Fold id per training frame for k-fold cross-validation, or -1 for frames
    that always stay in training. Correlated frames (same group, see
    dataset_select.correlated_groups) share a fold; each pair's closest-contact
    frame is never held out (the inner cutoff is set from it); groups are
    dealt round-robin within each composition class so folds are stratified."""
    import random

    from .dataset_select import _composition_class, closest_contact_frames, correlated_groups

    groups = correlated_groups(frames)
    protected = {groups[i] for i in closest_contact_frames(frames)}
    rng = random.Random(seed)
    fold_of_group = {}
    by_class = {}
    for i, g in enumerate(groups):
        if g not in protected:
            by_class.setdefault(_composition_class(frames[i]), set()).add(g)
    for cls in sorted(by_class):
        gids = sorted(by_class[cls])
        rng.shuffle(gids)
        for n, g in enumerate(gids):
            fold_of_group[g] = n % k
    return [fold_of_group.get(g, -1) for g in groups]


def cross_validate(amat_dir, point_dir, task: dict, cfg: dict, base_weights_path, solve_fn) -> dict:
    """k-fold CV from one design matrix: rows of held-out frames get weight 0
    (equivalent to leaving them out of the fit), each fold is solved, and the
    held-out frames are scored. Returns pooled per-frame rows in training-frame
    order plus pooled metrics; the fold solve directories are removed."""
    import shutil

    from . import evaluate
    from .weights import row_frames

    k = int(task["cv_folds"])
    assign = task["cv_assign"]
    frames = task["train_frames"]
    tags = [ln.split()[0] for ln in (amat_dir / "b-labeled.txt").read_text().splitlines() if ln.strip()]
    natoms = [float(x) for x in (amat_dir / "natoms.txt").read_text().split()]
    rows_frame = np.asarray(row_frames(tags, natoms))
    base_w = np.loadtxt(base_weights_path) if base_weights_path else np.ones(len(tags))
    per_frame = {}
    for fold in range(k):
        held = [i for i, f in enumerate(assign) if f == fold]
        if not held:
            continue
        mask = np.isin(rows_frame, held)
        wpath = point_dir / f"weights.cv{fold}.dat"
        np.savetxt(wpath, np.where(mask, 0.0, base_w), fmt="%.10g")
        d = point_dir / f"cv{fold}"
        try:
            params = solve_fn(str(wpath), str(d))
            ev = evaluate.evaluate_frames([frames[i] for i in held], [params], per_frame=True)["results"][0]
            for j, i in enumerate(held):
                per_frame[i] = (ev["per_frame_force"][j], ev["per_frame_group"][j],
                                ev["per_frame_energy_err_per_atom"][j], ev["per_frame_pressure_err_gpa"][j])
        finally:
            shutil.rmtree(d, ignore_errors=True)
            wpath.unlink(missing_ok=True)
    order = sorted(per_frame)
    rows = [per_frame[i][0] for i in order]
    groups = [per_frame[i][1] for i in order]
    # per-frame relative errors: the pooled RMS is dominated by frames the held-out fit extrapolates on
    frame_rel = [float(np.sqrt(r[0] / r[1])) if r[1] else float("nan") for r in rows]
    fragile = [i for i, e in zip(order, frame_rel) if e > 1.0]
    e_err = [per_frame[i][2] for i in order if per_frame[i][2] is not None]
    p_err = [per_frame[i][3] for i in order if per_frame[i][3] is not None]
    err = sum(r[0] for r in rows)
    ref = sum(r[1] for r in rows)
    return {
        "cv_folds": k, "cv_n_frames": len(rows),
        "cv_relative_force_error": float(np.sqrt(err / ref)) if ref else None,
        "cv_rmse_energy_per_atom": float(np.sqrt(np.mean(np.square(e_err)))) if e_err else None,
        "cv_rmse_pressure_gpa": float(np.sqrt(np.mean(np.square(p_err)))) if p_err else None,
        "cv_per_frame_force": rows, "cv_frame_groups": groups,
        "cv_median_frame_error": float(np.nanmedian(frame_rel)) if frame_rel else None,
        "cv_fragile_frames": fragile,  # training frames mispredicted (relative error > 1) when held out
        "cv_by_composition": evaluate.group_breakdown(rows, groups, {})["by_composition"],
    }


def profiles(all_results: dict, final_cfg: dict, keys=("order_2b", "order_3b", "order_4b", "s_maxim_2b", "s_maxim_3b",
                                                       "s_maxim_4b", "lambda_scale", "alpha", "stress_weight", "fcuttyp",
                                                       "weights_preset", "solver")) -> dict:
    """One-dimensional sensitivity profiles: for each hyperparameter, every
    fitted point that differs from the final configuration in that key only."""
    def norm(c):
        return {k: v for k, v in c.items() if v not in (None, 0, [], {}, 1.0, "CUBIC", "uniform") or k in ("order_2b", "s_maxim_2b")}

    base = norm(final_cfg)
    out = {}
    for key in keys:
        rows = []
        for r in all_results.values():
            if r.get("status") != "done":
                continue
            c = norm(r["cfg"])
            if {k: v for k, v in c.items() if k != key} != {k: v for k, v in base.items() if k != key}:
                continue
            rows.append({"value": r["cfg"].get(key, final_cfg.get(key)), "score": r.get("score"),
                         "holdout_relative_force_error": r.get("holdout_relative_force_error"),
                         "holdout_rmse_energy_per_atom": r.get("holdout_rmse_energy_per_atom"),
                         "n_params": r.get("n_params"), "md_cost": r.get("md_cost"),
                         "chosen": r["cfg"].get(key, final_cfg.get(key)) == final_cfg.get(key)})
        # one row per value (the chosen point and an explicit re-fit at the same value are the same evidence)
        by_value = {}
        for row in rows:
            k = repr(row["value"])
            if k not in by_value or row["chosen"]:
                by_value[k] = row
        rows = list(by_value.values())
        if len(rows) > 1:
            out[key] = sorted(rows, key=lambda x: (str(type(x["value"])), x["value"] if x["value"] is not None else -1))
    return out


def build_fm_args(cfg: dict, train_xyzf: str, n_train: int, masses: dict, thinnest: float, out_dir: str):
    from ._compose import ns

    pair3 = cfg.get("s_maxim_3b_pairs") if cfg.get("order_3b") else None
    cutoff = max([cfg["s_maxim_2b"]] + [c for c in (cfg.get("s_maxim_3b"), cfg.get("s_maxim_4b")) if c]
                 + list((pair3 or {}).values()))
    order = {"2": cfg["order_2b"], "3": cfg.get("order_3b") or 0}
    if cfg.get("order_4b"):
        order["4"] = cfg["order_4b"]
    return ns(
        trjfile=str(Path(train_xyzf).resolve()), nframes=n_train, elements=cfg["elements"], masses=masses,
        charges=None, order=order, cheby_range=[-1, 1],
        pair_cutoffs={p: [cfg["s_minim"][p], cfg["s_maxim_2b"]] for p in cfg["s_minim"]},
        morse_lambda=scaled_lambdas(cfg),
        default_s_minim=1.0, default_s_maxim=cfg["s_maxim_2b"], default_morse_lambda=1.5, s_delta=0.01,
        wraptrj=True, nlayers=max(1, nlayers_required(cutoff, thinnest)), fitcoul=False, fitstrs=str(cfg.get("fitstrs") or "false"),
        fitener="true" if cfg.get("fitener") else "false", fitpovr=False, chbtype="MORSE", fcuttyp=cfg.get("fcuttyp", "CUBIC"),
        exclude_3b=cfg.get("exclude_3b") or None, exclude_4b=cfg.get("exclude_4b") or None,
        special_maxim_3b=cfg.get("s_maxim_3b") if cfg.get("order_3b") else None,
        special_maxim_3b_pairs=pair3,
        special_maxim_4b=cfg.get("s_maxim_4b") if cfg.get("order_4b") else None,
        special_blocks=None, output_dir=out_dir,
    )


def params_nonzero(params_path) -> dict:
    from ..io.params import nonzero_by_body

    return nonzero_by_body(params_path)


def _train_force_error(work_dir: Path, solve_dir: Path | None = None):
    """Training-set force RMSE from chimes_lsq's own force.txt vs b.txt
    (free: no extra evaluation). b lives with the A-matrix, force.txt with the solve."""
    labels = [ln.split()[0] for ln in (work_dir / "b-labeled.txt").read_text().splitlines()]
    b = np.loadtxt(work_dir / "b.txt")
    f = np.loadtxt((solve_dir or work_dir) / "force.txt")
    is_force = np.array([lab != "+1" and "s_" not in lab for lab in labels])  # not energy, not stress rows
    diff = (b - f)[is_force]
    ref = b[is_force]
    rmse = float(np.sqrt(np.mean(diff**2)))
    return rmse, rmse / float(np.sqrt(np.mean(ref**2)))


def body_order_signal(work_dir: Path) -> dict:
    """Per body order: coefficient count and median design-matrix column norm,
    relative to the 2-body median. ChIMES smooths every distance in a cluster
    with (1 - r/r_c)^3, so a 3-/4-body cutoff only slightly past the neighbour
    shell multiplies 3 or 6 small factors and the columns become ~1e-11 of the
    2-body ones: numerically inert, whatever order they are given. Measured on
    real Cu-Zr data, where 4-body terms at a first-shell cutoff changed
    nothing."""
    import re

    log = (work_dir / "fm_setup.log").read_text(errors="replace")

    def count(pattern):
        m = re.search(pattern, log)
        return int(m.group(1)) if m else 0

    n2 = count(r"two-body non-coulomb parameters is:\s*(\d+)")
    n3 = count(r"three-body Chebyshev parameters is:\s*(\d+)")
    n4 = count(r"four-body\s+Chebyshev parameters is:\s*(\d+)")
    out = {"n_2b": n2, "n_3b": n3, "n_4b": n4}
    a_path = work_dir / "A.txt"
    if n2 and a_path.is_file() and a_path.stat().st_size < 400 * 2**20:
        norms = np.linalg.norm(np.loadtxt(a_path, ndmin=2), axis=0)
        m2 = float(np.median(norms[:n2]))
        if n3:
            out["signal_3b"] = float(np.median(norms[n2:n2 + n3])) / m2
        if n4:
            out["signal_4b"] = float(np.median(norms[n2 + n3:n2 + n3 + n4])) / m2
    return out


def bootstrap_se(frame_rows, n_boot: int = 300, seed: int = 0) -> float:
    """Standard error of the relative force error over holdout frames."""
    rows = np.asarray(frame_rows, dtype=float)
    if len(rows) < 3:
        return 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(rows), size=(n_boot, len(rows)))
    s = rows[idx].sum(axis=1)
    rel = np.sqrt(s[:, 0] / s[:, 2]) / np.sqrt(s[:, 1] / s[:, 2])
    return float(rel.std())


def _pair_label_elements(label: str) -> list:
    import re

    return re.findall(r"[A-Z][a-z]?", label)


def cluster_coverage(work_dir: Path) -> dict:
    """Per cluster type, from chimes_lsq's own log: contributing cluster
    instances and minimum pair distances. Keys are element lists as
    fm_setup.in's EXCLUDE blocks spell them ("Cu Cu Zr"). Instance counts
    include periodic images, so tiny replicated cells inflate them; zero
    means the type is absent from the data."""
    import re
    from collections import Counter

    lines = (work_dir / "fm_setup.log").read_text(errors="replace").splitlines()
    out = {"2b": {}, "3b": {}, "4b": {}}
    body_of = {"pair": ("2b", 2, 1), "triplet": ("3b", 3, 2), "quadruplet": ("4b", 4, 3)}
    for i, ln in enumerate(lines):
        m = re.search(r"Total number of configurations contributing to each (pair|triplet|quadruplet) type", ln)
        d = re.search(r"Minimum distances between atoms (triplet|quadruplet) pairs", ln)
        if m or d:
            body, n_atoms, per_atom = body_of[(m or d).group(1)]
            for row in lines[i + 1:]:
                toks = row.split()
                if not toks or not toks[0].isdigit():
                    break
                if body == "2b":
                    labels, rest = [toks[1] + toks[2]], toks[3:]
                else:
                    n_pairs = n_atoms * (n_atoms - 1) // 2
                    labels, rest = toks[1:1 + n_pairs], toks[1 + n_pairs:]
                counts = Counter(e for lab in labels for e in _pair_label_elements(lab))
                elems = sorted(counts.elements()) if body == "2b" else sorted(
                    e for e, c in counts.items() for _ in range(c // per_atom))
                entry = out[body].setdefault(" ".join(elems), {})
                if m:
                    if rest and rest[0] == "excluded":  # chimes_lsq prints this for EXCLUDEd types
                        entry["instances"], entry["excluded"] = 0, True
                    else:
                        entry["instances"] = int(float(rest[0]))
                else:
                    entry["min_distances"] = [round(float(x), 3) for x in rest]
    return out


def group_regressions(r: dict, best: dict, tolerance: float, min_frames: int = 3) -> list:
    """Composition groups where `r` is worse than `best` beyond max(tolerance x
    best's group error, the paired bootstrap SE on that group's frames)."""
    ga, gb = r.get("holdout_frame_groups"), best.get("holdout_frame_groups")
    ra, rb = r.get("holdout_per_frame_force"), best.get("holdout_per_frame_force")
    if not (ga and gb and ra and rb) or ga != gb or len(set(ga)) < 2:
        return []
    out = []
    for g in sorted(set(ga)):
        idx = [i for i, x in enumerate(ga) if x == g]
        if len(idx) < min_frames:
            continue
        a = np.asarray([ra[i] for i in idx], dtype=float)
        b = np.asarray([rb[i] for i in idx], dtype=float)
        ea, eb = np.sqrt(a[:, 0].sum() / a[:, 1].sum()), np.sqrt(b[:, 0].sum() / b[:, 1].sum())
        se = paired_se(a.tolist(), b.tolist())
        if ea - eb > max(tolerance * eb, se):
            out.append({"group": g, "error": round(float(ea), 4), "best": round(float(eb), 4), "margin": round(float(max(tolerance * eb, se)), 4)})
    return out


def paired_se(rows_a, rows_b, n_boot: int = 500, seed: int = 0) -> float:
    """Standard error of (relative force error of a) - (of b) when both were
    scored on the same holdout frames: resample frames jointly. Frame-to-frame
    difficulty cancels, so this is far smaller than either model's own SE."""
    a, b = np.asarray(rows_a, dtype=float), np.asarray(rows_b, dtype=float)
    if len(a) != len(b) or len(a) < 3:
        return float("inf")
    idx = np.random.default_rng(seed).integers(0, len(a), size=(n_boot, len(a)))
    sa, sb = a[idx].sum(axis=1), b[idx].sum(axis=1)
    return float((np.sqrt(sa[:, 0] / sa[:, 1]) - np.sqrt(sb[:, 0] / sb[:, 1])).std())


def _file_sig(path) -> str:
    st = Path(path).stat()
    return f"{st.st_size}:{st.st_mtime_ns}"


def fit_context(task: dict) -> str:
    """Everything besides cfg that changes a point's result: the data files
    (by size + mtime), solver, alpha and masses. A cached point is reused only
    when this matches, so new data or another solver in the same output
    directory refits instead of silently returning old numbers."""
    cfg = task["cfg"]
    ctx = {
        "train": [task["train_xyzf"], _file_sig(task["train_xyzf"])],
        "holdout": [task["holdout_xyzf"], _file_sig(task["holdout_xyzf"])],
        "n_train": task["n_train"],
        "solver": cfg.get("solver", task["algorithm"]),
        "alpha": cfg.get("alpha", task["alpha"]),
        "masses": task.get("masses"),
        "cv": [task.get("cv_folds") or 0, task.get("cv_seed") or 0],
    }
    return hashlib.sha256(json.dumps(ctx, sort_keys=True, default=str).encode()).hexdigest()[:16]


def run_point(task: dict) -> dict:
    """Fit + evaluate one configuration. Cached by config hash in its point
    directory, so re-running a search resumes instead of refitting; the cache
    is used only when `fit_context` (data, solver, alpha) also matches."""
    from ._compose import ns
    from . import amat_build, evaluate, fm_setup_gen, solve

    cfg, point_dir = task["cfg"], Path(task["point_dir"])
    context = fit_context(task)
    cached = point_dir / "result.json"
    from ..io import atomic

    prior = atomic.read_json(cached)  # None when missing or truncated by a killed job: refit
    if prior is not None:
        # failures are retried (they may have been our bug); fits from other data or solvers are redone
        if prior.get("status") in ("done", "timeout") and prior.get("context") == context:
            return prior
    fs.ensure_dir(point_dir)
    result = {"key": config_key(cfg), "cfg": cfg, "point_dir": str(point_dir), "context": context}
    try:
        # The design matrix depends on the basis, not on the solve. With task["amat_dir"], it is built once
        # there and kept, and solve-only variants (solver, alpha, stress weight, weights preset) re-solve
        # from it instead of rebuilding.
        amat_dir = Path(task["amat_dir"]) if task.get("amat_dir") else point_dir
        info_path = amat_dir / "amat_info.json"
        if not (task.get("amat_dir") and (amat_dir / "A.txt").is_file() and atomic.read_json(info_path)):
            fs.ensure_dir(amat_dir)
            fm = fm_setup_gen.run(build_fm_args(cfg, task["train_xyzf"], task["n_train"], task["masses"],
                                                task["thinnest"], str(amat_dir)))
            amat_build.run(ns(fm_setup_in=fm["fm_setup_in"], chimes_lsq_bin=None, machine=None, queue="batch",
                              walltime_hours=1.0, nodes=1, ntasks_per_node=None, dry_run=False,
                              timeout_s=task.get("timeout_s"), output_dir=str(amat_dir)))
            atomic.write_json(info_path, {"fm_setup_in": fm["fm_setup_in"], "nlayers": fm["params"]["nlayers"]})
        info = atomic.read_json(info_path) or {}
        weights_path = None
        if (cfg.get("weights_preset") and cfg["weights_preset"] != "uniform") or cfg.get("stress_weight") is not None:
            from . import weights as weights_stage

            over = {"stress": ["A", [float(cfg["stress_weight"])]]} if cfg.get("stress_weight") is not None else None
            weights_path = weights_stage.build(amat_dir, preset=cfg.get("weights_preset") or "uniform", overrides=over,
                                               out_path=point_dir / "weights.dat")["weights"]
        def solve_with(wpath, out_dir):
            return solve.run(ns(A=str(amat_dir / "A.txt"), b=str(amat_dir / "b.txt"), header=str(amat_dir / "params.header"),
                                map=str(amat_dir / "ff_groups.map"), dim=str(amat_dir / "dim.txt"),
                                algorithm=cfg.get("solver", task["algorithm"]), alpha=cfg.get("alpha", task["alpha"]), eps=1e-5,
                                weights=wpath, folds=4, normalize=False, split_files=False, machine=None, queue="batch",
                                walltime_hours=1.0, nodes=1, ntasks_per_node=None, poll_interval_s=60, dry_run=False,
                                output_dir=str(out_dir)))

        sv = solve_with(weights_path, point_dir)
        cv = None
        if task.get("cv_folds") and task.get("cv_assign"):
            cv = cross_validate(amat_dir, point_dir, task, cfg, weights_path, lambda w, d: solve_with(w, d)["params"])
        n_params, n_rows = (int(x) for x in (amat_dir / "dim.txt").read_text().split()[:2])
        tr_rmse, tr_rel = _train_force_error(amat_dir, point_dir)
        signal = body_order_signal(amat_dir)
        coverage = cluster_coverage(amat_dir)
        ev = evaluate.run(ns(params=[sv["params"]], holdout_xyzf=task["holdout_xyzf"], max_frames=None, per_frame=True))
        h = ev["results"][0]
        result.update({
            "status": "done",
            "params": sv["params"],
            "solver": cfg.get("solver", task["algorithm"]),
            "solver_alpha": sv.get("cv_alpha", cfg.get("alpha", task["alpha"])),
            "fm_setup_in": info.get("fm_setup_in"),
            "nlayers": info.get("nlayers"),
            "n_params": n_params,
            "n_equations": n_rows,
            "train_rmse_force": tr_rmse,
            "train_relative_force_error": tr_rel,
            "holdout_rmse_force": h["rmse_force_kcal_mol_ang"],
            "holdout_relative_force_error": h["relative_force_error"],
            "holdout_rmse_energy_per_atom": h["rmse_energy_kcal_mol_per_atom"],
            "holdout_rmse_pressure_gpa": h.get("rmse_pressure_gpa"),
            "holdout_frames_below_inner_cutoff": h.get("n_frames_below_inner_cutoff", 0),
            "holdout_rmse_stress_gpa": h.get("rmse_stress_gpa"),
            "holdout_relative_force_se": bootstrap_se(h["per_frame_force"]),
            "holdout_per_frame_force": h["per_frame_force"],
            "holdout_frame_groups": h.get("per_frame_group"),
            "holdout_by_composition": h.get("by_composition"),
            "signal": signal,
            "nonzero": params_nonzero(sv["params"]),
            "cluster_coverage": coverage,
        })
        if cv:
            # Selection runs on the CV estimate (k times the frames of a holdout); the external holdout stays reported.
            result.update({
                "ext_holdout_relative_force_error": result["holdout_relative_force_error"],
                "ext_holdout_rmse_energy_per_atom": result["holdout_rmse_energy_per_atom"],
                "ext_holdout_by_composition": result["holdout_by_composition"],
                "holdout_relative_force_error": cv["cv_relative_force_error"],
                "holdout_rmse_energy_per_atom": cv["cv_rmse_energy_per_atom"],
                "holdout_rmse_pressure_gpa": cv["cv_rmse_pressure_gpa"],
                "holdout_relative_force_se": bootstrap_se(cv["cv_per_frame_force"]),
                "holdout_per_frame_force": cv["cv_per_frame_force"],
                "holdout_frame_groups": cv["cv_frame_groups"],
                "holdout_by_composition": cv["cv_by_composition"],
                "cv_folds": cv["cv_folds"], "cv_n_frames": cv["cv_n_frames"],
                "cv_median_frame_error": cv["cv_median_frame_error"], "cv_fragile_frames": cv["cv_fragile_frames"],
            })
        if not task.get("keep_amat"):
            (amat_dir / "A.txt").unlink(missing_ok=True)  # the design matrix is the big file; params/force/b stay
    except TimeoutError as exc:
        result.update({"status": "timeout", "error": str(exc)[-600:]})
    except Exception as exc:  # noqa: BLE001 - one bad point must not end the search
        result.update({"status": "failed", "error": str(exc)[-600:]})
    atomic.write_json(cached, result)
    return result


# ------------------------------------------------------------------ selection

def score(r: dict, objective: str, energy_weight: float) -> float:
    s = r["holdout_relative_force_error"]
    if objective == "force+energy" and r.get("holdout_rmse_energy_per_atom") is not None:
        s += energy_weight * r["holdout_rmse_energy_per_atom"]
    return s


DEFAULT_DENSITY = 0.06  # atoms/A^3, dense metal; hyper-search passes the training set's own


def md_cost(r: dict, density: float = DEFAULT_DENSITY) -> float:
    """Relative ChIMES MD cost per atom: for each body order n, the clusters
    per atom within its cutoff, (density * 4/3 pi r^3)^(n-1), times that body
    order's coefficient count (each cluster evaluates its type's
    polynomial). A shorter cutoff with many more coefficients can cost more
    than a longer one with few: 3-body order 4 at 7.0 A is ~half the cost of
    order 6 at 6.33 A."""
    import math

    c = r["cfg"]
    sig = r.get("signal") or {}
    nz = r.get("nonzero")  # live coefficients: LASSO zeroes many, and deploy removes them before MD
    n2 = (nz or {}).get("2b") or sig.get("n_2b") or r["n_params"]
    n3 = (nz or {}).get("3b") if nz else (sig.get("n_3b") or 0)
    n4 = (nz or {}).get("4b") if nz else (sig.get("n_4b") or 0)

    def neigh(rc):
        return density * 4.0 / 3.0 * math.pi * rc**3

    cost = neigh(c["s_maxim_2b"]) * n2
    if c.get("order_3b") and n3:
        r3 = c.get("s_maxim_3b") or c["s_maxim_2b"]
        if c.get("s_maxim_3b_pairs"):  # effective cutoff: cube-mean of the per-pair cutoffs
            vals = list(c["s_maxim_3b_pairs"].values())
            r3 = (sum(v**3 for v in vals) / len(vals)) ** (1 / 3)
        cost += neigh(r3) ** 2 * n3
    if c.get("order_4b") and n4:
        cost += neigh(c.get("s_maxim_4b") or c.get("s_maxim_3b") or c["s_maxim_2b"]) ** 3 * n4
    return cost


def cost_key(r: dict, density: float = DEFAULT_DENSITY):
    return (md_cost(r, density), r["n_params"])


def select(results: list, *, objective: str, energy_weight: float, tolerance: float, max_param_ratio: float,
           density: float = DEFAULT_DENSITY, prefer: str = "cheaper") -> dict:
    """Best score, then the cheapest point (cost_key) statistically tied with
    it: score - best <= max(tolerance x best, SE of the paired difference).
    With prefer="richer", the tied point with the most coefficients is taken
    instead: before active learning the literature errs toward complexity,
    because a sparse initial set makes cross-validation favor bases that are
    too small (Lindsey et al., npj Comput. Mater. 11, 26, 2025).
    The paired SE resamples holdout frames jointly for both models, so shared
    frame difficulty cancels; without per-frame data it falls back to the
    best point's own SE. Points over max_param_ratio coefficients per
    equation are reported but never chosen."""
    done = [r for r in results if r.get("status") == "done" and r.get("holdout_relative_force_error") is not None]
    for r in done:
        r["score"] = score(r, objective, energy_weight)
        r["params_per_equation"] = round(r["n_params"] / r["n_equations"], 3)
    ok = [r for r in done if r["params_per_equation"] <= max_param_ratio]
    if not ok:
        return {"chosen": None, "best": None, "reason": "no completed point within the parameter budget"}
    best = min(ok, key=lambda r: r["score"])

    def margin(r):
        if r is best:
            return 0.0
        if r.get("holdout_per_frame_force") and best.get("holdout_per_frame_force"):
            se = paired_se(r["holdout_per_frame_force"], best["holdout_per_frame_force"])
        else:
            se = best.get("holdout_relative_force_se") or 0.0
        return max(tolerance * best["score"], se)

    for r in ok:
        r["tie_margin"] = round(margin(r), 5)
        r["group_regressions"] = group_regressions(r, best, tolerance) if r is not best else []
    # a candidate is tied only if it is tied overall AND no composition group got worse beyond that group's
    # own paired noise: a pooled score hid a 60% loss on pure Cu (0.09 -> 0.145) on Cu-Zr
    within = [r for r in ok if r["score"] - best["score"] <= r["tie_margin"] and not r["group_regressions"]]
    for r in ok:
        r["md_cost"] = round(md_cost(r, density), 1)
    if prefer == "richer":
        chosen = min(within, key=lambda r: (-r["n_params"], r["score"]))
        reason = ("lowest score and richest" if chosen is best else
                  f"richest point statistically tied with the best ({chosen['n_params']} vs {best['n_params']} coefficients; "
                  f"difference {chosen['score'] - best['score']:.4f} <= margin {chosen['tie_margin']:.4f}; prefer=richer)")
    else:
        chosen = min(within, key=lambda r: cost_key(r, density))
        reason = ("lowest score and cheapest" if chosen is best else
                  f"cheapest point statistically tied with the best (difference {chosen['score'] - best['score']:.4f} <= "
                  f"margin {chosen['tie_margin']:.4f}; estimated MD cost {chosen['md_cost']:.3g} vs {best['md_cost']:.3g})")
    return {"chosen": chosen, "best": best, "reason": reason, "n_excluded_over_budget": len(done) - len(ok)}
