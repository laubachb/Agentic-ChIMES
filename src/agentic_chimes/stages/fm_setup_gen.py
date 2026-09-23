"""Generate a `fm_setup.in` from typed, high-level parameters (elements,
per-pair cutoffs, Chebyshev order, fit flags) instead of hand-authoring the
raw ChIMES grammar. Validated (see tests/unit/test_fm_setup.py) by
round-tripping the two golden fixtures in
codes/chimes_lsq-LLfork/test_suite-lsq/ through agentic_chimes.io.fm_setup.

Nested fields (pair_cutoffs, morse_lambda, masses, charges, exclude lists)
are naturally JSON, not flat CLI flags -- use --json-in for anything beyond
the single-cutoff/single-lambda convenience case these flags cover.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

from ..io import fm_setup

NAME = "fm-setup-gen"
SUMMARY = "Generate fm_setup.in from typed parameters (elements, cutoffs, order, fit flags)."
SCHEMA = {
    "type": "object",
    "required": ["trjfile", "nframes", "elements", "order"],
    "properties": {
        "trjfile": {"type": "string", "description": "Path to the .xyzf training trajectory (written as-is into TRJFILE; use an absolute path)."},
        "nframes": {"type": "integer"},
        "elements": {"type": "array", "items": {"type": "string"}},
        "masses": {"type": "object", "additionalProperties": {"type": "number"}, "description": "element -> amu; default 1.0 if omitted."},
        "charges": {"type": "object", "additionalProperties": {"type": "number"}, "description": "element -> fixed charge; default 0.0."},
        "order": {"type": "object", "description": '{"2": N, "3": N, "4": N (optional)}'},
        "cheby_range": {"type": "array", "items": {"type": "number"}, "default": [-1, 1]},
        "pair_cutoffs": {
            "type": "object",
            "additionalProperties": {"type": "array", "items": {"type": "number"}},
            "description": '"El1-El2" -> [s_minim, s_maxim]; falls back to --default-s-minim/--default-s-maxim for unlisted pairs.',
        },
        "morse_lambda": {"type": "object", "additionalProperties": {"type": "number"}, "description": '"El1-El2" -> morse lambda; falls back to --default-morse-lambda.'},
        "s_delta": {"type": "number", "default": 0.01},
        "wraptrj": {"type": "boolean", "default": True},
        "nlayers": {"type": "integer", "default": 1},
        "fitcoul": {"type": "boolean", "default": False},
        "fitstrs": {"type": "string", "default": "false"},
        "fitener": {"type": "string", "default": "false"},
        "fitpovr": {"type": "boolean", "default": False},
        "chbtype": {"type": "string", "default": "MORSE"},
        "fcuttyp": {"type": "string", "default": "CUBIC"},
        "exclude_3b": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
        "exclude_4b": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
    },
}


def add_arguments(parser) -> None:
    parser.add_argument("--trjfile", default=None)
    parser.add_argument("--nframes", type=int, default=None)
    parser.add_argument("--elements", type=lambda s: s.split(","), default=None, help="Comma-separated, e.g. C,H")
    parser.add_argument("--order", type=json.loads, default=None, help='JSON, e.g. \'{"2":12,"3":5,"4":4}\'')
    parser.add_argument("--default-s-minim", dest="default_s_minim", type=float, default=1.0)
    parser.add_argument("--default-s-maxim", dest="default_s_maxim", type=float, default=6.0)
    parser.add_argument("--default-morse-lambda", dest="default_morse_lambda", type=float, default=1.5)
    parser.add_argument("--pair-cutoffs", dest="pair_cutoffs", type=json.loads, default=None, help='JSON {"El1-El2":[s_minim,s_maxim]}')
    parser.add_argument("--morse-lambda", dest="morse_lambda", type=json.loads, default=None, help='JSON {"El1-El2": lambda}')
    parser.add_argument("--masses", type=json.loads, default=None, help='JSON {element: amu}')
    parser.add_argument("--charges", type=json.loads, default=None, help='JSON {element: charge}')
    parser.add_argument("--fitener", default="false")
    parser.add_argument("--fitstrs", default="false")
    parser.add_argument("--fitcoul", action="store_true")
    parser.add_argument("--fitpovr", action="store_true")
    parser.add_argument("--wraptrj", type=lambda s: s.lower() != "false", default=True)
    parser.add_argument("--nlayers", type=int, default=1)
    parser.add_argument("--chbtype", default="MORSE")
    parser.add_argument("--fcuttyp", default="CUBIC")
    parser.add_argument("--s-delta", dest="s_delta", type=float, default=0.01)


def _pair_key(a: str, b: str) -> str:
    return f"{a}-{b}"


def _lookup_pair(d: dict, a: str, b: str, default):
    if not d:
        return default
    return d.get(_pair_key(a, b), d.get(_pair_key(b, a), default))


def _build_params(args_dict: dict) -> dict:
    elements = args_dict["elements"]
    if isinstance(elements, str):
        elements = elements.split(",")
    masses = args_dict.get("masses") or {}
    charges = args_dict.get("charges") or {}
    pair_cutoffs = args_dict.get("pair_cutoffs") or {}
    morse_lambda = args_dict.get("morse_lambda") or {}
    order = args_dict["order"]

    atom_types = [
        {"idx": i + 1, "symbol": el, "charge": float(charges.get(el, 0.0)), "mass": float(masses.get(el, 1.0))}
        for i, el in enumerate(elements)
    ]

    pairs = []
    for idx, (a, b) in enumerate(itertools.combinations_with_replacement(elements, 2), start=1):
        s_minim, s_maxim = _lookup_pair(
            pair_cutoffs, a, b, [args_dict.get("default_s_minim", 1.0), args_dict.get("default_s_maxim", 6.0)]
        )
        lam = _lookup_pair(morse_lambda, a, b, args_dict.get("default_morse_lambda", 1.5))
        pairs.append(
            {
                "idx": idx,
                "type1": a,
                "type2": b,
                "s_minim": float(s_minim),
                "s_maxim": float(s_maxim),
                "s_delta": float(args_dict.get("s_delta", 0.01)),
                "morse_lambda": float(lam),
            }
        )

    params = {
        "trjfile": args_dict["trjfile"],
        "wraptrj": bool(args_dict.get("wraptrj", True)),
        "nframes": int(args_dict["nframes"]),
        "nlayers": int(args_dict.get("nlayers", 1)),
        "fitcoul": bool(args_dict.get("fitcoul", False)),
        "fitstrs": str(args_dict.get("fitstrs", "false")),
        "fitener": str(args_dict.get("fitener", "false")),
        "fitpovr": bool(args_dict.get("fitpovr", False)),
        "pairtyp": "CHEBYSHEV",
        "order2": int(order["2"]),
        "order3": int(order["3"]),
        "chbtype": args_dict.get("chbtype", "MORSE"),
        "natmtyp": len(atom_types),
        "atom_types": atom_types,
        "pairs": pairs,
        "fcuttyp": args_dict.get("fcuttyp", "CUBIC"),
        "special_blocks": [],
    }
    if order.get("4") is not None:
        params["order4"] = int(order["4"])
        cheby_range = args_dict.get("cheby_range", [-1, 1])
        params["cheby_min"], params["cheby_max"] = float(cheby_range[0]), float(cheby_range[1])
    if args_dict.get("exclude_3b"):
        params["exclude_3b"] = args_dict["exclude_3b"]
    if args_dict.get("exclude_4b"):
        params["exclude_4b"] = args_dict["exclude_4b"]

    return params


def run(args) -> dict:
    args_dict = vars(args)
    if not args_dict.get("elements") or not args_dict.get("order") or not args_dict.get("trjfile") or not args_dict.get("nframes"):
        raise ValueError("fm-setup-gen requires at least trjfile, nframes, elements, and order (via flags or --json-in)")

    params = _build_params(args_dict)
    text = fm_setup.render(params)

    out_dir = Path(args_dict.get("output_dir") or ".")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fm_setup.in"
    out_path.write_text(text)

    return {"fm_setup_in": str(out_path), "params": params}
