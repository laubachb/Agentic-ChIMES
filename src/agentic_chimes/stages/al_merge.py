"""Merge an active-learning round's labeled frames into the training set.

What `cat a.xyzf b.xyzf` would get wrong, and this stage checks:

- **One level of theory.** The new frames' provenance (from `qe-relabel
  --collect`) must match the base data's, by the same rules as `data-curate`.
  Active learning must reuse the base set's QE settings.
- **Holdout unchanged.** Only the training file grows. The manifest's
  holdout stays fixed, so errors stay comparable across rounds.
- **File format.** Frames are rewritten together: once any frame is
  triclinic, every frame is written NON_ORTHO (chimes_lsq segfaults on a
  mixed file).
- **Fit flags.** New frames must carry what the base fit uses: energies if
  `fitener`; stresses if `fitstrs`, since chimes_lsq fits stresses for all
  frames or none.
- **Cycle bookkeeping.** `frame_cycles.json` records the cycle each training
  frame joined in, for `weights --frame-cycles ... --decay-cycles n` (the
  n/I decay of Lindsey 2025, 2026).

Writes `train.xyzf`, `frame_cycles.json` and an updated `data_manifest.json`
that later stages (`hyper-search`, `fm-setup-gen`, `evaluate`) read like the
original.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..io import xyzf as xyzf_io
from . import data_curate
from ..io import atomic
from ..io import fs

NAME = "al-merge"
SUMMARY = "Merge labeled active-learning frames into the training set (theory check, fixed holdout, cycle bookkeeping)."
SCHEMA = {
    "type": "object",
    "required": ["data_manifest", "new_xyzf", "cycle"],
    "properties": {
        "data_manifest": {"type": "string", "description": "Current data_manifest.json (the base one, or the previous round's)."},
        "new_xyzf": {"type": "array", "items": {"type": "string"}, "description": "Labeled frames of this round (provenance.json beside each)."},
        "cycle": {"type": "integer", "description": "Active-learning cycle these frames belong to (base data = 0)."},
        "allow_mixed_theory": {"type": "boolean", "default": False},
    },
}


def add_arguments(parser) -> None:
    parser.add_argument("--data-manifest", dest="data_manifest", default=None)
    parser.add_argument("--new-xyzf", dest="new_xyzf", action="append", default=None)
    parser.add_argument("--cycle", type=int, default=None)
    parser.add_argument("--allow-mixed-theory", dest="allow_mixed_theory", action="store_true")


def run(args) -> dict:
    if not getattr(args, "data_manifest", None) or not getattr(args, "new_xyzf", None) or getattr(args, "cycle", None) is None:
        raise ValueError("al-merge needs --data-manifest, --new-xyzf (repeatable) and --cycle")
    if args.cycle < 1:
        raise ValueError("cycle must be >= 1 (the base data is cycle 0)")
    manifest = json.loads(Path(args.data_manifest).read_text())
    base = xyzf_io.read_xyzf(manifest["train_xyzf"])
    cycles_path = Path(manifest.get("frame_cycles") or "")
    cycles = json.loads(cycles_path.read_text()) if manifest.get("frame_cycles") and cycles_path.is_file() else [0] * len(base)
    if len(cycles) != len(base):
        raise ValueError(f"frame_cycles has {len(cycles)} entries but the training set has {len(base)} frames")

    provs = [p for p in manifest.get("sources", []) if isinstance(p, dict)]
    new_frames, new_provs = [], []
    for path in args.new_xyzf:
        prov_path = Path(path).parent / "provenance.json"
        if not prov_path.is_file():
            raise ValueError(f"{path}: no provenance.json beside it, so its level of theory cannot be checked")
        new_provs.append(json.loads(prov_path.read_text()))
        new_frames += xyzf_io.read_xyzf(path)
    data_curate.check_theory(provs + new_provs, bool(getattr(args, "allow_mixed_theory", False)))

    hints = manifest.get("fit_hints") or {}
    elements = set(manifest.get("elements") or [])
    problems = []
    if elements and any(set(f.symbols) - elements for f in new_frames):
        problems.append(f"new frames contain elements outside {sorted(elements)}")
    if hints.get("fitener") and any(f.energy is None for f in new_frames):
        problems.append("the fit uses energies but some new frames have none")
    if hints.get("fitstrs") and any(f.stress is None for f in new_frames):
        problems.append("the fit uses stresses but some new frames have none (label with the same QE settings; "
                        "stresses are computed by default)")
    if problems:
        raise ValueError("; ".join(problems))

    out = Path(getattr(args, "output_dir", None) or ".").resolve()
    fs.ensure_dir(out)
    train = out / "train.xyzf"
    merged = base + new_frames
    xyzf_io.write_xyzf(merged, train)
    cycles = cycles + [int(args.cycle)] * len(new_frames)
    atomic.write_json((out / "frame_cycles.json"), cycles)
    new_manifest = {**manifest, "train_xyzf": str(train), "n_train": len(merged), "frame_cycles": str(out / "frame_cycles.json"),
                    "sources": provs + new_provs,
                    "al_rounds": (manifest.get("al_rounds") or []) + [{"cycle": int(args.cycle), "n_added": len(new_frames),
                                                                      "from": [str(Path(p).resolve()) for p in args.new_xyzf]}]}
    atomic.write_json((out / "data_manifest.json"), new_manifest, indent=1)
    return {"data_manifest": str(out / "data_manifest.json"), "train_xyzf": str(train), "n_train": len(merged),
            "n_added": len(new_frames), "holdout_xyzf": manifest.get("holdout_xyzf"),
            "frame_cycles": str(out / "frame_cycles.json"), "cycle": int(args.cycle)}
