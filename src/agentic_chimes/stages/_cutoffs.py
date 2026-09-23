"""Data-driven ChIMES cutoff/Morse-lambda determination, implementing the
documented guidance in codes/chimes_lsq-LLfork/doc/source/{lsq_input_file,
quick_start}.rst (see docs/concepts/cutoffs_and_lambdas.md for full
citations):

  - S_MINIM: minimum observed pair distance minus a small delta
    (documented range 0.002-0.02 Angstrom).
  - MORSE_LAMBDA: location of the first RDF peak.
  - S_MAXIM (2-body): "usually ~8 Angstrom" (2nd non-bonded solvation
    shell) -- taken from the 2nd RDF minimum when a clear one is found,
    else the documented ~8 Angstrom default.
  - S_MAXIM (3-body): 1st non-bonded solvation shell -- the 1st RDF
    minimum after the bonding peak.
  - S_MAXIM (4-body): documented as "between the first or second RDF
    minimum" -- defaults to the (shorter) 1st minimum, matching 4-body's
    higher computational cost; override via `s_maxim_4b_use_second`.

Every S_MAXIM is capped by the same hard safety bound chimes_lsq itself
enforces in codes/chimes_lsq-LLfork/src/ClassDefs.C (BOXDIM.IS_RCUT_SAFE):
outer cutoff must not exceed half the (layered) box length, or the
fm_setup.in it's used in will error. `capped`/`cap_reason` in the output
report whenever that bound overrode the data-driven value.
"""

from __future__ import annotations

from ..io import rdf as rdf_io

DOCUMENTED_S_MINIM_DELTA_RANGE = (0.002, 0.02)
DEFAULT_S_MINIM_DELTA = 0.02  # the specific example value quick_start.rst gives
DOCUMENTED_S_MAXIM_2B_DEFAULT = 8.0  # Angstrom, "usually set to about 8 A"


def derive_pair_params(
    frames: list,
    elements: list,
    *,
    s_minim_delta: float = DEFAULT_S_MINIM_DELTA,
    s_maxim_2b_default: float = DOCUMENTED_S_MAXIM_2B_DEFAULT,
    nlayers: int = 1,
    s_maxim_4b_use_second: bool = False,
) -> dict:
    mins = rdf_io.pair_min_distance(frames, elements)
    rdfs = rdf_io.pair_rdf(frames, elements)
    box_safety_bound = rdf_io.min_box_dimension(frames) * nlayers / 2.0

    pairs = {}
    for key, min_dist in mins.items():
        pair_str_key = f"{key[0]}-{key[1]}"  # matches fm_setup_gen's "El1-El2" pair_cutoffs convention; JSON needs string keys anyway
        r = rdfs.get(key)
        morse_lambda = r.first_peak() if r else None
        s_maxim_3b_raw = r.first_minimum_after_peak() if r else None
        s_maxim_2b_raw = (r.second_minimum() if r else None) or s_maxim_2b_default
        s_maxim_4b_raw = (r.second_minimum() if (s_maxim_4b_use_second and r) else s_maxim_3b_raw) or s_maxim_3b_raw

        if morse_lambda is None:
            morse_lambda = min_dist  # fall back to the bond-length proxy we do have

        entry = {
            "s_minim": round(min_dist - s_minim_delta, 6),
            "morse_lambda": round(morse_lambda, 6),
        }

        for label, raw in (("s_maxim_2b", s_maxim_2b_raw), ("s_maxim_3b", s_maxim_3b_raw), ("s_maxim_4b", s_maxim_4b_raw)):
            if raw is None:
                raw = min(s_maxim_2b_default, box_safety_bound)
            capped = raw > box_safety_bound
            value = min(raw, box_safety_bound)
            entry[label] = round(value, 6)
            entry[f"{label}_capped"] = capped
            entry[f"{label}_cap_reason"] = (
                f"data-driven value {raw:.4f} exceeded the box-safety bound "
                f"{box_safety_bound:.4f} (min box dim {rdf_io.min_box_dimension(frames):.4f} "
                f"* nlayers {nlayers} / 2); capped."
                if capped
                else None
            )

        pairs[pair_str_key] = entry

    return {
        "pairs": pairs,
        "box_safety_bound": round(box_safety_bound, 6),
        "s_minim_delta": s_minim_delta,
        "nlayers": nlayers,
    }
