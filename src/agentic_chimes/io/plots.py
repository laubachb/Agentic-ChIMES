"""Shared matplotlib helpers: headless, one style, PNGs next to the JSON.

Plots are optional. Every stage works without matplotlib; `figure()` returns
None when it is missing and callers skip plotting.
"""

from __future__ import annotations

import warnings
from pathlib import Path


def figure(figsize=(6.0, 4.5)):
    try:
        import matplotlib

        matplotlib.use("Agg")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import matplotlib.pyplot as plt
    except ImportError:
        return None, None
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.grid(True, alpha=0.3)
    return fig, ax


def save(fig, path) -> str | None:
    if fig is None:
        return None
    path = Path(path)
    fig.savefig(path, dpi=130)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return str(path)


def parity(pred, ref, path, *, labels=None, unit="kcal/mol/Å", title="Forces: model vs reference"):
    """Parity plot of predicted vs reference values, optionally colored by label."""
    import numpy as np

    fig, ax = figure((5.5, 5.5))
    if fig is None:
        return None
    pred, ref = np.asarray(pred, float).ravel(), np.asarray(ref, float).ravel()
    if labels is not None:
        labels = np.asarray(labels)
        for lab in sorted(set(labels.tolist())):
            m = labels == lab
            ax.scatter(ref[m], pred[m], s=6, alpha=0.5, label=f"{lab} (n={int(m.sum())})")
        ax.legend(fontsize=8)
    else:
        ax.scatter(ref, pred, s=6, alpha=0.5)
    lo, hi = float(min(ref.min(), pred.min())), float(max(ref.max(), pred.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel(f"reference ({unit})")
    ax.set_ylabel(f"model ({unit})")
    ax.set_title(title)
    return save(fig, path)


def lines(series, path, *, xlabel="", ylabel="", title="", logx=False, logy=False, marker="o"):
    """series: {label: (x, y)} -> one PNG."""
    fig, ax = figure()
    if fig is None:
        return None
    for lab, (x, y) in series.items():
        ax.plot(x, y, marker=marker, ms=4, lw=1.2, label=lab)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if len(series) > 1:
        ax.legend(fontsize=8)
    return save(fig, path)


def bars(values: dict, path, *, ylabel="", title=""):
    fig, ax = figure((max(4.0, 0.8 * len(values) + 2), 4.0))
    if fig is None:
        return None
    keys = list(values)
    ax.bar(range(len(keys)), [values[k] for k in keys])
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return save(fig, path)


def histograms(samples: dict, path, *, xlabel="", title="", bins=40, logx=False):
    """samples: {label: 1-D values} -> overlaid density histograms."""
    import numpy as np

    fig, ax = figure()
    if fig is None:
        return None
    allv = np.concatenate([np.asarray(v, float).ravel() for v in samples.values() if len(v)])
    if logx:
        allv = allv[allv > 0]
        edges = np.logspace(np.log10(allv.min()), np.log10(allv.max()), bins) if len(allv) else bins
        ax.set_xscale("log")
    else:
        edges = np.linspace(allv.min(), allv.max(), bins) if len(allv) else bins
    for lab, v in samples.items():
        v = np.asarray(v, float).ravel()
        ax.hist(v[v > 0] if logx else v, bins=edges, alpha=0.5, density=True, label=f"{lab} (n={len(v)})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend(fontsize=8)
    return save(fig, path)
