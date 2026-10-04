"""Per-row fitting weights for a built A-matrix (`weights.dat` for
`solve --weights`), with the published ChIMES weighting schemes as presets.

Rows are classified from amat-build's own `b-labeled.txt` exactly as
al_driver's `gen_ff.py` does: tag `+1` = energy row, a tag containing `s_` =
stress component, otherwise a force component (`G_` prefixes mark gas-phase
clusters). Weight methods are al_driver's (config.py `WEIGHTS_FORCE` etc.),
so a scheme written here means the same thing inside al_driver:

  A  w = a0                          E  w = n_atoms^a0
  B  w = a0 * I^a1  (I = cycle, 0->1) F  w = a0*exp(a1*(X/n_atoms - a2)/a3)
  C  w = a0*exp(a1*|X|/a2)           G  w = a0*exp(a1*(|X| - a2)/a3)
  D  w = a0*exp(a1*(X - a2)/a3)      (X = the row's reference value)

Presets (forces / energies / stresses; see docs/concepts/literature.md):

  uniform           1 / 1 / 1        (what every fit used before this stage)
  al_driver         1 / 0.1 / 250    (al_driver's defaults)
  lindsey2020       1 / 5 / -        (JCP 153, 134117: reactive C/O)
  carbon2_large     1 / 0.1 / 100    (npj Comput. Mater. 11, 26, 2025)
  hierarchical2026  1 / 0.3 / 100    (npj Comput. Mater. 12, 18, 2026)

Active-learning decay (Lindsey 2025, 2026): with `frame_cycles` (the cycle
each training frame was added in) and `decay_cycles` n, every row of a frame
from cycle I is multiplied by n / max(I, 1), so early, possibly unphysical
frames cannot pull the fit away from later ground-truth data.

Weights rescale rows, so they change what "training error" means; holdout
errors from `evaluate` stay comparable across schemes.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

NAME = "weights"
USES_OUTPUT_DIR = False  # writes weights.dat into the amat-build work dir it reads
SUMMARY = "Per-row fitting weights (weights.dat for solve --weights): al_driver methods + published ChIMES presets."

PRESETS = {
    "uniform": {"force": ["A", [1.0]], "energy": ["A", [1.0]], "stress": ["A", [1.0]]},
    "al_driver": {"force": ["A", [1.0]], "energy": ["A", [0.1]], "stress": ["A", [250.0]]},
    "lindsey2020": {"force": ["A", [1.0]], "energy": ["A", [5.0]], "stress": ["A", [1.0]]},
    "carbon2_large": {"force": ["A", [1.0]], "energy": ["A", [0.1]], "stress": ["A", [100.0]]},
    "hierarchical2026": {"force": ["A", [1.0]], "energy": ["A", [0.3]], "stress": ["A", [100.0]]},
}
_KINDS = ("force", "energy", "stress", "force_gas", "energy_gas")
_NPARAM = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 1, "F": 4, "G": 4}

SCHEMA = {
    "type": "object",
    "required": ["work_dir"],
    "properties": {
        "work_dir": {"type": "string", "description": "amat-build work dir holding b-labeled.txt and natoms.txt."},
        "preset": {"type": "string", "enum": sorted(PRESETS), "default": "uniform"},
        "force_method": {"type": ["array", "null"], "description": "Override the preset, al_driver form, e.g. [\"A\", [1.0]]."},
        "energy_method": {"type": ["array", "null"]},
        "stress_method": {"type": ["array", "null"]},
        "force_gas_method": {"type": ["array", "null"], "description": "Gas-cluster (G_) force rows; default = force."},
        "energy_gas_method": {"type": ["array", "null"], "description": "Gas-cluster energy rows; default = energy."},
        "cycle": {"type": "integer", "default": 0, "description": "Current ALC, for method B."},
        "frame_cycles": {"type": ["array", "string", "null"], "description": "Cycle index per training frame (list or JSON file), for decay."},
        "decay_cycles": {"type": ["integer", "null"], "description": "n in the n/I active-learning decay; needs frame_cycles."},
    },
}


def add_arguments(parser) -> None:
    parser.add_argument("--work-dir", dest="work_dir", default=None)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="uniform")
    for k in _KINDS:
        parser.add_argument(f"--{k.replace('_', '-')}-method", dest=f"{k}_method", type=json.loads, default=None,
                            help='al_driver weight method as JSON, e.g. \'["A", [0.3]]\'')
    parser.add_argument("--cycle", type=int, default=0)
    parser.add_argument("--frame-cycles", dest="frame_cycles", default=None, help="JSON file: cycle per training frame")
    parser.add_argument("--decay-cycles", dest="decay_cycles", type=int, default=None)


def method_weight(method, value: float, natoms: float, cycle: int) -> float:
    kind, a = method[0], [float(x) for x in method[1]]
    if kind not in _NPARAM or len(a) != _NPARAM[kind]:
        raise ValueError(f"weight method {method!r}: expected one of {sorted(_NPARAM)} with {_NPARAM.get(kind)} parameters")
    if kind == "A":
        return a[0]
    if kind == "B":
        return a[0] * float(max(cycle, 1)) ** a[1]
    if kind == "C":
        return a[0] * math.exp(a[1] * abs(value) / a[2])
    if kind == "D":
        return a[0] * math.exp(a[1] * (value - a[2]) / a[3])
    if kind == "E":
        return float(natoms) ** a[0]
    if kind == "F":
        return a[0] * math.exp(a[1] * (value / natoms - a[2]) / a[3])
    return a[0] * math.exp(a[1] * (abs(value) - a[2]) / a[3])


def row_kind(tag: str) -> str:
    gas = "G_" in tag
    if "+1" in tag:
        return "energy_gas" if gas else "energy"
    if "s_" in tag:
        return "stress"
    return "force_gas" if gas else "force"


def row_frames(tags: list, natoms: list) -> list:
    """Training-frame index per row: 3N force rows, then that frame's energy/stress rows."""
    frames, i, f = [], 0, 0
    while i < len(tags):
        j = i + 3 * int(natoms[i])
        while j < len(tags) and row_kind(tags[j]) in ("energy", "energy_gas", "stress"):
            j += 1
        frames.extend([f] * (j - i))
        i, f = j, f + 1
    return frames[: len(tags)]


def build(work_dir, *, preset="uniform", overrides=None, cycle=0, frame_cycles=None, decay_cycles=None,
          out_path=None) -> dict:
    work = Path(work_dir)
    rows = [ln.split() for ln in (work / "b-labeled.txt").read_text().splitlines() if ln.strip()]
    natoms = [float(x) for x in (work / "natoms.txt").read_text().split()]
    if len(natoms) != len(rows):
        raise ValueError(f"natoms.txt has {len(natoms)} lines but b-labeled.txt has {len(rows)} rows")
    scheme = {**PRESETS[preset], **{k: v for k, v in (overrides or {}).items() if v is not None}}
    scheme.setdefault("force_gas", scheme["force"])
    scheme.setdefault("energy_gas", scheme["energy"])
    tags = [r[0] for r in rows]
    w = np.array([method_weight(scheme[row_kind(t)], float(v), n, cycle) for (t, v), n in zip(rows, natoms)])

    decay_note = None
    if decay_cycles:
        if frame_cycles is None:
            raise ValueError("decay_cycles needs frame_cycles (the cycle each training frame was added in)")
        if isinstance(frame_cycles, str):
            frame_cycles = json.loads(Path(frame_cycles).read_text())
        fidx = row_frames(tags, natoms)
        n_frames = fidx[-1] + 1 if fidx else 0
        if len(frame_cycles) != n_frames:
            raise ValueError(f"frame_cycles has {len(frame_cycles)} entries but the A-matrix has {n_frames} frames")
        w *= np.array([decay_cycles / max(int(frame_cycles[f]), 1) for f in fidx])
        decay_note = f"rows of frames from cycle I scaled by {decay_cycles}/max(I,1)"

    out = Path(out_path) if out_path else work / "weights.dat"
    np.savetxt(out, w, fmt="%.10g")
    kinds = [row_kind(t) for t in tags]
    summary = {}
    for k in sorted(set(kinds)):
        wk = w[[i for i, x in enumerate(kinds) if x == k]]
        summary[k] = {"rows": int(len(wk)), "min": float(wk.min()), "max": float(wk.max())}
    return {"weights": str(out), "preset": preset, "scheme": scheme, "rows": len(w), "by_row_type": summary,
            "decay": decay_note}


def run(args) -> dict:
    if not getattr(args, "work_dir", None):
        raise ValueError("weights requires --work-dir (an amat-build work dir)")
    overrides = {k: getattr(args, f"{k}_method", None) for k in _KINDS}
    return build(args.work_dir, preset=getattr(args, "preset", "uniform") or "uniform", overrides=overrides,
                 cycle=getattr(args, "cycle", 0) or 0, frame_cycles=getattr(args, "frame_cycles", None),
                 decay_cycles=getattr(args, "decay_cycles", None))
