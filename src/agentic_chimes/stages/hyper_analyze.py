"""Analyze a training set for ChIMES hyperparameters: per element pair, the
minimum sampled distance (-> inner cutoff), the first RDF peak (-> Morse
lambda) and the RDF shell minima (-> outer-cutoff candidates for the
2-, 3- and 4-body terms), plus the N_LAYERS each candidate cutoff needs and
the equation count that bounds how large a basis the data can support.

Works on any cell shape (ASE neighbour lists over all periodic images).
Writes `hyper_analysis.json`, which `hyper-search` reads. See
stages/_hyper.py for the ChIMES guidance each number implements.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..io import xyzf as xyzf_io
from . import _hyper
from ..io import atomic
from ..io import fs

NAME = "hyper-analyze"
SUMMARY = "Per-pair distance/RDF analysis -> inner cutoffs, Morse lambdas, outer-cutoff candidates, N_LAYERS, data size."
SCHEMA = {
    "type": "object",
    "properties": {
        "data_manifest": {"type": ["string", "null"], "description": "data_manifest.json from data-curate (supplies train_xyzf and elements)."},
        "train_xyzf": {"type": ["string", "null"]},
        "elements": {"type": ["array", "null"], "items": {"type": "string"}},
        "r_max": {"type": "number", "default": 8.0, "description": "Analysis radius (A); also the largest 2-body cutoff proposed."},
        "s_minim_delta": {"type": "number", "default": 0.02, "description": "Inner cutoff = min distance - delta (documented 0.002-0.02 A)."},
    },
}


def add_arguments(parser) -> None:
    parser.add_argument("--data-manifest", dest="data_manifest", default=None)
    parser.add_argument("--train-xyzf", dest="train_xyzf", default=None)
    parser.add_argument("--elements", type=lambda s: [x for x in s.split(",") if x], default=None)
    parser.add_argument("--r-max", dest="r_max", type=float, default=8.0)
    parser.add_argument("--s-minim-delta", dest="s_minim_delta", type=float, default=0.02)


def resolve_inputs(args):
    manifest = {}
    if getattr(args, "data_manifest", None):
        manifest = json.loads(Path(args.data_manifest).read_text())
    train = getattr(args, "train_xyzf", None) or manifest.get("train_xyzf") or manifest.get("curated_xyzf")
    elements = getattr(args, "elements", None) or manifest.get("elements")
    if not train or not elements:
        raise ValueError("give --data-manifest, or --train-xyzf and --elements")
    return train, sorted(elements), manifest


def run(args) -> dict:
    train, elements, manifest = resolve_inputs(args)
    frames = xyzf_io.read_xyzf(train)
    analysis = _hyper.analyze(frames, elements, r_max=getattr(args, "r_max", 8.0) or 8.0,
                              s_minim_delta=getattr(args, "s_minim_delta", 0.02) or 0.02)
    analysis["train_xyzf"] = str(Path(train).resolve())
    analysis["data_manifest"] = getattr(args, "data_manifest", None)
    if manifest.get("level_of_theory"):
        analysis["level_of_theory"] = manifest["level_of_theory"]

    out = Path(getattr(args, "output_dir", None) or ".")
    fs.ensure_dir(out)
    path = out / "hyper_analysis.json"
    atomic.write_json(path, analysis, indent=1)
    return {"hyper_analysis": str(path), **analysis}
