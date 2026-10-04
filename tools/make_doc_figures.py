#!/usr/bin/env python3
"""Draw the figures in docs/assets/figures.

Concept figures are computed from the formulas on the page that shows them.
Cu-Zr figures read docs/assets/figures/data/cuzr.json, small extracts of the
development study's own artifacts (learning_curve.json, quests.json/.npz,
fingerprint.json, batch.json).

    python3 tools/make_doc_figures.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets" / "figures"

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]          # fixed categorical order: blue, orange, aqua, yellow

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "legend.frameon": False, "axes.axisbelow": True,
    "xtick.major.size": 0, "ytick.major.size": 0,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(OUT / name)


# ---------------------------------------------------------------- the ChIMES basis

R_IN, R_OUT, LAM = 2.0, 7.0, 2.5      # an illustrative metal pair: inner cutoff, outer cutoff, Morse lambda (A)


def morse_s(r):
    x, x_in, x_out = np.exp(-r / LAM), math.exp(-R_IN / LAM), math.exp(-R_OUT / LAM)
    return (x - 0.5 * (x_in + x_out)) / (0.5 * (x_in - x_out))


def cubic(r, r_out=R_OUT):
    return np.where(r < r_out, (1 - r / r_out) ** 3, 0.0)


def tersoff(r, f_o, r_out=R_OUT):
    d_t = r_out * (1 - f_o)
    mid = 0.5 + 0.5 * np.sin(np.pi * (r - d_t) / (r_out - d_t) + np.pi / 2)
    return np.where(r < d_t, 1.0, np.where(r > r_out, 0.0, mid))


def fig_basis():
    r = np.linspace(R_IN, R_OUT, 400)
    s = morse_s(r)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.3))
    ax[0].plot(r, s, color=SERIES[0])
    ax[0].set(title="Distance transform", xlabel="pair distance r (Å)", ylabel="transformed distance s")
    ax[0].annotate("inner cutoff: s = +1", (R_IN, 1), (R_IN + 0.45, 0.93), color=INK2, fontsize=9)
    ax[0].annotate("outer cutoff: s = −1", (R_OUT, -1), (R_OUT, -0.2), color=INK2, fontsize=9, ha="right")
    for n in range(1, 5):
        ax[1].plot(r, np.cos(n * np.arccos(np.clip(s, -1, 1))), color=SERIES[n - 1], label=f"T{n}")
    ax[1].set(title="Chebyshev polynomials of s(r)", xlabel="pair distance r (Å)", ylabel="Tₙ(s)")
    ax[1].legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.24), handlelength=1.4, columnspacing=1.2)
    for k, (lab, f) in enumerate((("cubic", cubic(r)), ("Tersoff, f_O = 0.5", tersoff(r, 0.5)), ("Tersoff, f_O = 0.75", tersoff(r, 0.75)))):
        ax[2].plot(r, f, color=SERIES[k], label=lab)
    ax[2].set(title="Smoothing functions", xlabel="pair distance r (Å)", ylabel="f_s(r)")
    ax[2].legend(ncol=1, loc="upper center", bbox_to_anchor=(0.5, -0.24), handlelength=1.4)
    fig.tight_layout()
    save(fig, "chimes_basis.png")


def fig_manybody_smoothing():
    r = np.linspace(R_IN, R_OUT, 400)
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.3), sharey=True)
    for a, (name, f) in zip(ax, (("Cubic", cubic(r)), ("Tersoff, f_O = 0.5", tersoff(r, 0.5)))):
        for k, (lab, power) in enumerate((("2-body: 1 factor", 1), ("3-body: 3 factors", 3), ("4-body: 6 factors", 6))):
            a.plot(r, f ** power, color=SERIES[k], label=lab)
        a.set(title=name, xlabel="every edge of the cluster at r (Å)")
    ax[0].set_ylabel("product of smoothing factors")
    ax[1].legend(loc="upper right")
    fig.tight_layout()
    save(fig, "smoothing_manybody.png")


# ---------------------------------------------------------------- QUESTS, illustrated in one dimension

def fig_quests_concept():
    rng = np.random.default_rng(3)
    x_ref = np.concatenate([rng.normal(0.30, 0.035, 60), rng.normal(0.62, 0.02, 25)])
    h = 0.015
    y = np.linspace(0.1, 0.95, 600)
    ksum = np.exp(-0.5 * ((y[:, None] - x_ref[None, :]) / h) ** 2).sum(1)
    fig, ax = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True, gridspec_kw={"height_ratios": [1, 1.25]})
    ax[0].fill_between(y, ksum / len(x_ref), color=SERIES[0], alpha=0.10, linewidth=0)
    ax[0].plot(y, ksum / len(x_ref), color=SERIES[0])
    ax[0].plot(x_ref, np.full_like(x_ref, -0.012), "|", color=INK2, markersize=7, markeredgewidth=0.8)
    ax[0].set(title="Reference environments (ticks) and their kernel density", ylabel="(1/N) Σⱼ K(x, xⱼ)")
    dh = -np.log(np.maximum(ksum, 1e-300))
    ax[1].plot(y, dh, color=SERIES[1])
    ax[1].axhline(0, color=INK2, linewidth=1)
    ax[1].annotate("δH > 0: not covered by the reference", (0.495, 17.5), color=INK2, fontsize=9, ha="center")
    ax[1].annotate("δH < 0: covered", (0.30, -6.3), color=INK2, fontsize=9, ha="center")
    ax[1].set(title="Differential entropy of a new environment y", xlabel="descriptor value (one dimension, illustrative)",
              ylabel="δH(y) = −ln Σⱼ K(y, xⱼ)")
    ax[1].set_ylim(-8, 25)
    fig.tight_layout()
    save(fig, "quests_concept.png")


# ---------------------------------------------------------------- Cu-Zr results

def _cuzr():
    return json.loads((OUT / "data" / "cuzr.json").read_text())


def fig_cuzr_quests():
    d = _cuzr()["quests"]
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.4))
    n = [p["n_frames"] for p in d["entropy_curve"]]
    hh = [p["entropy"] for p in d["entropy_curve"]]
    ax[0].plot(n, hh, color=SERIES[0], marker="o", markersize=6.5, markeredgecolor=SURFACE, markeredgewidth=2)
    ax[0].annotate(f"{hh[-1]:.2f} nats", (n[-1], hh[-1]), (n[-1] - 7, hh[-1]), ha="right", va="center", color=INK, fontsize=9)
    ax[0].set(title="Entropy of the training set is still rising", xlabel="training frames", ylabel="information entropy H (nats)")
    e = np.array(d["bin_edges"])
    c = 0.5 * (e[:-1] + e[1:])
    ax[1].step(c, d["reference_dH_density"], where="mid", color=SERIES[0], label="training set (self)")
    ax[1].step(c, d["candidate_dH_density"], where="mid", color=SERIES[1], label="MD harvest vs training set")
    ax[1].axvline(d["threshold"], color=INK2, linewidth=1)
    ax[1].annotate("novelty threshold", (d["threshold"], ax[1].get_ylim()[1] * 0.9), (d["threshold"] + 1.2, ax[1].get_ylim()[1] * 0.9),
                   color=INK2, fontsize=9)
    ax[1].set(title="Most MD environments are new", xlabel="δH per atomic environment (nats)", ylabel="density")
    ax[1].legend(loc="upper right", bbox_to_anchor=(1.0, 0.82))
    fig.tight_layout()
    save(fig, "cuzr_quests.png")


def fig_cuzr_learning_curve():
    c = _cuzr()["learning_curve"]
    n = np.array([p["n_frames"] for p in c])
    f = np.array([p["relative_force_error"] for p in c])
    sp = np.array([p.get("relative_force_error_spread") or 0.0 for p in c])
    en = np.array([p["rmse_energy_per_atom"] for p in c])
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.3))
    ax[0].fill_between(n, f - sp, f + sp, color=SERIES[0], alpha=0.10, linewidth=0)
    ax[0].plot(n, f, color=SERIES[0], marker="o", markersize=6.5, markeredgecolor=SURFACE, markeredgewidth=2)
    ax[0].annotate(f"{f[-1]:.2f}", (n[-1], f[-1]), (n[-1], f[-1] + 0.018), ha="center", color=INK, fontsize=9)
    ax[0].set(title="Forces: plateau from 63 frames", xlabel="training frames", ylabel="holdout relative force error")
    ax[1].plot(n, en, color=SERIES[1], marker="o", markersize=6.5, markeredgecolor=SURFACE, markeredgewidth=2)
    ax[1].annotate(f"{en[-1]:.2f}", (n[-1], en[-1]), (n[-1], en[-1] + 0.09), ha="center", color=INK, fontsize=9)
    ax[1].set(title="Energies: still improving", xlabel="training frames", ylabel="holdout energy RMSE (kcal/mol/atom)")
    for a in ax:
        a.set_xscale("log")
        a.set_xticks(n)
        a.set_xticklabels([str(int(v)) for v in n])
        a.minorticks_off()
    fig.tight_layout()
    save(fig, "cuzr_learning_curve.png")


def fig_cuzr_structure_weight():
    d = _cuzr()["structure_weight"]
    keys = sorted((k for k in d if k != "type-agnostic"), key=float)
    labels = [f"α = {k}" for k in keys] + ["type-agnostic"]
    vals = [d[k] for k in keys] + [d["type-agnostic"]]
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    bars = ax.bar(range(len(vals)), vals, width=0.42, color=SERIES[0])
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.0f}" if v >= 10 else f"{v:.1f}", (b.get_x() + b.get_width() / 2, v), (0, 3), textcoords="offset points",
                    ha="center", color=INK, fontsize=9)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels)
    ax.grid(axis="x", visible=False)
    ax.set(title="MD harvest vs training set: separation by structure weight", ylabel="D² / critical value")
    ax.annotate("composition only", (0, 0), (0, -30), textcoords="offset points", ha="center", color=INK2, fontsize=8.5)
    ax.annotate("structure only", (len(keys) - 1, 0), (0, -30), textcoords="offset points", ha="center", color=INK2, fontsize=8.5)
    fig.tight_layout()
    save(fig, "cuzr_structure_weight.png")


def fig_cuzr_al_batch():
    rows = [r for r in _cuzr()["al_batch"] if r["quests_dH_max"] is not None and r["committee_force_spread"] is not None]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for k, (lab, want) in enumerate((("selected for labeling", True), ("not selected", False))):
        pts = [r for r in rows if bool(r["selected"]) is want]
        ax.scatter([r["quests_dH_max"] for r in pts], [r["committee_force_spread"] for r in pts], s=58, color=SERIES[k],
                   edgecolor=SURFACE, linewidth=2, label=f"{lab} ({len(pts)})", zorder=3)
    ax.set(title="One active-learning batch: 46 MD frames",
           xlabel="structural novelty: largest δH in the frame (nats)", ylabel="committee force spread (kcal/mol/Å)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "cuzr_al_batch.png")


if __name__ == "__main__":
    fig_basis()
    fig_manybody_smoothing()
    fig_quests_concept()
    fig_cuzr_quests()
    fig_cuzr_learning_curve()
    fig_cuzr_structure_weight()
    fig_cuzr_al_batch()
