"""Diversity-based active-learning batch selection, wrapping al_driver's
existing Metropolis-MC energy-histogram selector
(codes/al_driver-LLfork/src/gen_selections.py:gen_subset). This is coverage
selection, not uncertainty/query-by-committee -- al_driver's own
committeeALmode.rst doc stub confirms that mode was never finished upstream.
The concrete future plug-in point for uncertainty-based selection is
evaluate's `committee_spread` output (multiple params.txt models diffed on
the same frames), deliberately not wired up here.

The selection *signal* is each candidate frame's ChIMES-predicted energy,
normalized per atom -- confirmed against al_driver's own convention
(`codes/al_driver-LLfork/utilities/new-get_dumb_ener_subjob.sh`'s
`paste xyzlist.dat xyzlist.energies | awk '{print $NF/$1}'`, i.e. raw
energy / natoms), computed here in-process via the same ctypes evaluator
`evaluate.py` uses (no external MD run needed). `gen_subset` itself is
imported and called directly out of al_driver's vendored source -- the
actual Metropolis-MC diversity algorithm is al_driver's own, not
reimplemented here.

Deliberately standalone, not a full ALC-X/CENTRAL_REPO integration:
al_driver's `populate_repo`/`cleanup_repo` assume a `../ALC-<n>/` and
`../CENTRAL_REPO/` on-disk layout tied to a live `main.py` run (see
al_run.py's own scope note for the same reasoning re: config.py). Here,
`central_repo` is a plain energies file you can chain across repeated
al-select calls yourself (this stage's own `central_repo_out`), not a
directory convention.

Note: `gen_subset` calls Python's bare `exit()` on a few internal error
paths (e.g. a degenerate histogram) rather than raising a catchable
exception -- this stage pre-checks the common cases (n_select vs. pool
size) to avoid them, but an unusual candidate pool could still hit one of
those paths and terminate the process directly rather than surfacing a
clean CLI error.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from .. import config
from ..io import xyzf as xyzf_io

NAME = "al-select"
SUMMARY = "Diversity (energy-histogram) active-learning batch selection via al_driver's gen_subset."
SCHEMA = {
    "type": "object",
    "required": ["candidate_frames", "params", "n_select"],
    "properties": {
        "candidate_frames": {"type": "string", "description": "Unlabeled/candidate .xyzf pool (positions/box; forces/energy fields ignored)."},
        "params": {"type": "string", "description": "params.txt used to predict each candidate's energy -- the selection signal."},
        "n_select": {"type": "integer", "description": "Number of frames to select (gen_subset's nsel)."},
        "central_repo": {"type": ["string", "null"], "description": "Optional plain energies_normed file from a prior al-select's central_repo_out; biases selection to cover gaps in what's already been picked."},
        "histogram_bins": {"type": "integer", "default": 20, "description": "gen_subset's nbins."},
        "nsweep": {"type": ["integer", "null"], "description": "MC sweep cycles per candidate (multiplied internally by pool size, matching gen_subset's own convention); default mirrors al_driver's documented MEM_CYCL default of histogram_bins/10."},
        "energy_cutoff": {"type": "number", "default": 1.0e10, "description": "Candidates with |energy_normed| >= this are excluded (gen_subset's ecut)."},
        "seed": {"type": "integer", "default": 1},
    },
}


def add_arguments(parser) -> None:
    parser.add_argument("--candidate-frames", dest="candidate_frames", default=None)
    parser.add_argument("--params", default=None)
    parser.add_argument("--n-select", dest="n_select", type=int, default=None)
    parser.add_argument("--central-repo", dest="central_repo", default=None)
    parser.add_argument("--histogram-bins", dest="histogram_bins", type=int, default=20)
    parser.add_argument("--nsweep", type=int, default=None)
    parser.add_argument("--energy-cutoff", dest="energy_cutoff", type=float, default=1.0e10)
    parser.add_argument("--seed", type=int, default=1)


def _load_gen_selections():
    src_dir = str(config.AL_DRIVER_SRC)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    try:
        import gen_selections
    except ImportError as exc:
        raise ImportError(
            "al-select requires matplotlib + cycler (al_driver's gen_selections.py imports them "
            "unconditionally for its diagnostic plots): install with `pip install "
            "agentic-chimes[al-select]`."
        ) from exc
    return gen_selections


def _predicted_energies_normed(frames, params_path) -> list:
    from . import evaluate as evaluate_stage  # reuse the ctypes wrapper loader/cell-vector helper

    wrapper = evaluate_stage._load_wrapper()
    ptr = wrapper.chimes_open_instance()
    wrapper.set_chimes_instance(ptr, small=False)
    wrapper.init_chimes_instance(ptr, params_path, 0)

    energies = []
    try:
        for frame in frames:
            cell_a, cell_b, cell_c = evaluate_stage._cell_vectors(frame)
            xcrd = [p[0] for p in frame.positions]
            ycrd = [p[1] for p in frame.positions]
            zcrd = [p[2] for p in frame.positions]
            fx0 = [0.0] * frame.natoms
            fy0 = [0.0] * frame.natoms
            fz0 = [0.0] * frame.natoms
            stress0 = [0.0] * 9
            _fx, _fy, _fz, _stress, energy = wrapper.calculate_chimes_instance(
                ptr,
                frame.natoms,
                xcrd,
                ycrd,
                zcrd,
                frame.symbols,
                cell_a,
                cell_b,
                cell_c,
                0.0,
                fx0,
                fy0,
                fz0,
                stress0,
            )
            energies.append(energy / frame.natoms)
    finally:
        wrapper.chimes_close_instance(ptr)
    return energies


def run(args) -> dict:
    candidate_path = getattr(args, "candidate_frames", None)
    params_path = getattr(args, "params", None)
    n_select = getattr(args, "n_select", None)
    if not candidate_path:
        raise ValueError("al-select requires --candidate-frames")
    if not params_path:
        raise ValueError("al-select requires --params")
    if not n_select:
        raise ValueError("al-select requires --n-select")

    frames = xyzf_io.read_xyzf(candidate_path)
    if n_select > len(frames):
        raise ValueError(f"n_select={n_select} exceeds candidate pool size {len(frames)}")

    gen_selections = _load_gen_selections()
    energies = _predicted_energies_normed(frames, params_path)

    out_root = Path(getattr(args, "output_dir", None) or ".").resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    energies_path = out_root / "candidate.energies_normed"
    energies_path.write_text("\n".join(str(e) for e in energies) + "\n")

    histogram_bins = getattr(args, "histogram_bins", None) or 20
    nsweep = getattr(args, "nsweep", None)
    if nsweep is None:
        nsweep = max(1, histogram_bins // 10)
    seed = getattr(args, "seed", None) if getattr(args, "seed", None) is not None else 1
    energy_cutoff = getattr(args, "energy_cutoff", None) or 1.0e10

    central_repo = getattr(args, "central_repo", None)
    kwargs = dict(
        energies=str(energies_path),
        nsel=n_select,
        nsweep=nsweep,
        nbins=histogram_bins,
        ecut=energy_cutoff,
        seed=seed,
    )
    if central_repo:
        kwargs["repo"] = str(Path(central_repo).resolve())

    cwd = Path.cwd()
    os.chdir(out_root)
    try:
        gen_selections.gen_subset(**kwargs)
    finally:
        os.chdir(cwd)

    selection_path = out_root / "all.selection.dat"
    if not selection_path.is_file():
        raise RuntimeError(f"gen_subset did not produce {selection_path}")
    selected_indices = sorted(int(x) for x in selection_path.read_text().split())

    selected_frames = [frames[i] for i in selected_indices]
    selected_xyzf = out_root / "selected.xyzf"
    xyzf_io.write_xyzf(selected_frames, selected_xyzf)

    central_repo_out = out_root / "central_repo_energies_normed.txt"
    prior_lines = Path(central_repo).read_text().split() if central_repo else []
    combined = list(prior_lines) + [str(energies[i]) for i in selected_indices]
    central_repo_out.write_text("\n".join(combined) + "\n")

    return {
        "n_candidates": len(frames),
        "n_selected": len(selected_indices),
        "selected_indices": selected_indices,
        "selected_xyzf": str(selected_xyzf),
        "central_repo_out": str(central_repo_out),
        "energy_histogram_pdf": str(out_root / "energy_hist.pdf"),
        "residuals_pdf": str(out_root / "residuals.pdf"),
    }
