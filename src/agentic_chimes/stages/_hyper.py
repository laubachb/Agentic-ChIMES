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


def build_fm_args(cfg: dict, train_xyzf: str, n_train: int, masses: dict, thinnest: float, out_dir: str):
    from ._compose import ns

    cutoff = max([cfg["s_maxim_2b"]] + [c for c in (cfg.get("s_maxim_3b"), cfg.get("s_maxim_4b")) if c])
    order = {"2": cfg["order_2b"], "3": cfg.get("order_3b") or 0}
    if cfg.get("order_4b"):
        order["4"] = cfg["order_4b"]
    return ns(
        trjfile=str(Path(train_xyzf).resolve()), nframes=n_train, elements=cfg["elements"], masses=masses,
        charges=None, order=order, cheby_range=[-1, 1],
        pair_cutoffs={p: [cfg["s_minim"][p], cfg["s_maxim_2b"]] for p in cfg["s_minim"]},
        morse_lambda={p: round(cfg["morse_lambda"][p] * cfg.get("lambda_scale", 1.0), 4) for p in cfg["morse_lambda"]},
        default_s_minim=1.0, default_s_maxim=cfg["s_maxim_2b"], default_morse_lambda=1.5, s_delta=0.01,
        wraptrj=True, nlayers=max(1, nlayers_required(cutoff, thinnest)), fitcoul=False, fitstrs="false",
        fitener="true" if cfg.get("fitener") else "false", fitpovr=False, chbtype="MORSE", fcuttyp=cfg.get("fcuttyp", "CUBIC"),
        exclude_3b=cfg.get("exclude_3b") or None, exclude_4b=cfg.get("exclude_4b") or None,
        special_maxim_3b=cfg.get("s_maxim_3b") if cfg.get("order_3b") else None,
        special_maxim_4b=cfg.get("s_maxim_4b") if cfg.get("order_4b") else None,
        special_blocks=None, output_dir=out_dir,
    )


def _train_force_error(work_dir: Path):
    """Training-set force RMSE from chimes_lsq's own force.txt vs b.txt
    (free: no extra evaluation)."""
    labels = [ln.split()[0] for ln in (work_dir / "b-labeled.txt").read_text().splitlines()]
    b = np.loadtxt(work_dir / "b.txt")
    f = np.loadtxt(work_dir / "force.txt")
    is_force = np.array([lab != "+1" for lab in labels])
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


def run_point(task: dict) -> dict:
    """Fit + evaluate one configuration. Cached by config hash in its point
    directory, so re-running a search resumes instead of refitting."""
    from ._compose import ns
    from . import evaluate, fm_setup_gen, model_build

    cfg, point_dir = task["cfg"], Path(task["point_dir"])
    cached = point_dir / "result.json"
    if cached.is_file():
        prior = json.loads(cached.read_text())
        if prior.get("status") in ("done", "timeout"):  # failures are retried: they may have been our bug
            return prior
    point_dir.mkdir(parents=True, exist_ok=True)
    result = {"key": config_key(cfg), "cfg": cfg, "point_dir": str(point_dir)}
    try:
        fm = fm_setup_gen.run(build_fm_args(cfg, task["train_xyzf"], task["n_train"], task["masses"],
                                            task["thinnest"], str(point_dir)))
        mb = model_build.run(ns(fm_setup_in=fm["fm_setup_in"], chimes_lsq_bin=None,
                                algorithm=cfg.get("solver", task["algorithm"]), alpha=cfg.get("alpha", task["alpha"]), eps=1e-5, weights=None, folds=4, normalize=False, machine=None,
                                queue="batch", walltime_hours=1.0, nodes=1, ntasks_per_node=None, poll_interval_s=60,
                                timeout_s=task.get("timeout_s"), output_dir=str(point_dir)))
        work = Path(mb["work_dir"])
        n_params, n_rows = (int(x) for x in (work / "dim.txt").read_text().split()[:2])
        tr_rmse, tr_rel = _train_force_error(work)
        signal = body_order_signal(work)
        coverage = cluster_coverage(work)
        ev = evaluate.run(ns(params=[mb["params"]], holdout_xyzf=task["holdout_xyzf"], max_frames=None, per_frame=True))
        h = ev["results"][0]
        result.update({
            "status": "done",
            "params": mb["params"],
            "solver": cfg.get("solver", task["algorithm"]),
            "solver_alpha": mb["solve"].get("cv_alpha", cfg.get("alpha", task["alpha"])),
            "fm_setup_in": fm["fm_setup_in"],
            "nlayers": fm["params"]["nlayers"],
            "n_params": n_params,
            "n_equations": n_rows,
            "train_rmse_force": tr_rmse,
            "train_relative_force_error": tr_rel,
            "holdout_rmse_force": h["rmse_force_kcal_mol_ang"],
            "holdout_relative_force_error": h["relative_force_error"],
            "holdout_rmse_energy_per_atom": h["rmse_energy_kcal_mol_per_atom"],
            "holdout_relative_force_se": bootstrap_se(h["per_frame_force"]),
            "holdout_per_frame_force": h["per_frame_force"],
            "signal": signal,
            "cluster_coverage": coverage,
        })
        for bulky in ("A.txt",):
            (work / bulky).unlink(missing_ok=True)  # the design matrix is the big file; params/force/b stay
    except TimeoutError as exc:
        result.update({"status": "timeout", "error": str(exc)[-600:]})
    except Exception as exc:  # noqa: BLE001 - one bad point must not end the search
        result.update({"status": "failed", "error": str(exc)[-600:]})
    cached.write_text(json.dumps(result, indent=1))
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
    n2 = sig.get("n_2b") or r["n_params"]
    n3 = sig.get("n_3b") or 0
    n4 = sig.get("n_4b") or 0

    def neigh(rc):
        return density * 4.0 / 3.0 * math.pi * rc**3

    cost = neigh(c["s_maxim_2b"]) * n2
    if c.get("order_3b") and n3:
        cost += neigh(c.get("s_maxim_3b") or c["s_maxim_2b"]) ** 2 * n3
    if c.get("order_4b") and n4:
        cost += neigh(c.get("s_maxim_4b") or c.get("s_maxim_3b") or c["s_maxim_2b"]) ** 3 * n4
    return cost


def cost_key(r: dict, density: float = DEFAULT_DENSITY):
    return (md_cost(r, density), r["n_params"])


def select(results: list, *, objective: str, energy_weight: float, tolerance: float, max_param_ratio: float,
           density: float = DEFAULT_DENSITY) -> dict:
    """Best score, then the cheapest point (cost_key) statistically tied with
    it: score - best <= max(tolerance x best, SE of the paired difference).
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
    within = [r for r in ok if r["score"] - best["score"] <= r["tie_margin"]]
    for r in ok:
        r["md_cost"] = round(md_cost(r, density), 1)
    chosen = min(within, key=lambda r: cost_key(r, density))
    reason = ("lowest score and cheapest" if chosen is best else
              f"cheapest point statistically tied with the best (difference {chosen['score'] - best['score']:.4f} <= "
              f"margin {chosen['tie_margin']:.4f}; estimated MD cost {chosen['md_cost']:.3g} vs {best['md_cost']:.3g})")
    return {"chosen": chosen, "best": best, "reason": reason, "n_excluded_over_budget": len(done) - len(ok)}
