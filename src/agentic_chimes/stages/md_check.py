"""Short MD validation of one or more candidate models: the check the ChIMES
literature applies before trusting a model chosen on holdout error.

For water, holdout cross-validation kept preferring larger bases that were
over-structured, or unstable, in MD; the final model was picked by comparing
MD with DFT (Lindsey et al., JCTC 15, 436, 2019). This stage runs short NVT
MD of every candidate `params.txt` at every temperature on the same starting
structure, and reports per run:

  stable            no runaway: LAMMPS finished, no lost atoms, finite
                    energies, no potential-energy jump above
                    `max_pe_jump_per_atom` between dumps (second half), and the second-half
                    mean temperature not above target x (1 + tolerance)
  equilibrated      second-half mean temperature within the tolerance (runs
                    too short to thermalize are stable but not equilibrated)
  close contacts    fraction of dumped frames whose closest pair (per element
                    pair) comes within `close_contact_margin` of the model's
                    inner cutoff; `below_inner_cutoff` counts frames inside
                    it (the model is extrapolating there)
  rdf               partial RDFs over the second half; with `reference_xyzf`
                    (e.g. DFT-MD frames at the same conditions), the mean
                    absolute g(r) difference per pair, `rdf_distance`

and harvests frames for active learning the way the published parallel-AL
recipe does (Lindsey et al. 2025, 2026): up to `harvest_close` close-contact
frames plus up to `harvest_other` others per run, written to one
`harvest.xyzf` (structures only, to label with `qe-relabel`).

Runs locally (`workers` in parallel, single-rank LAMMPS each) or, with
`--machine`, as one Slurm job that runs this same stage on a compute node.
"""

from __future__ import annotations

import json
import math
import random
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ..io import lammps_data
from ..io import xyzf as xyzf_io

NAME = "md-check"
SUMMARY = "Short NVT MD of candidate models: stability, close contacts, RDFs vs a reference, frames harvested for AL."
SUPPORTS_DRY_RUN = True
SCHEMA = {
    "type": "object",
    "required": ["params", "elements", "masses"],
    "properties": {
        "params": {"type": "array", "items": {"type": "string"}, "description": "Candidate params.txt files (same elements/masses)."},
        "structure_xyzf": {"type": ["string", "null"], "description": "Starting structure(s); frame_index picks one."},
        "frame_index": {"type": "integer", "default": 0},
        "prototype": {"type": ["object", "null"], "description": "ase.build.bulk kwargs instead of a file (see benchmark)."},
        "elements": {"type": "array", "items": {"type": "string"}},
        "masses": {"type": "object"},
        "temperatures": {"type": "array", "items": {"type": "number"}, "default": [300.0]},
        "nsteps": {"type": "integer", "default": 2000},
        "timestep": {"type": "number", "default": 1.0, "description": "fs"},
        "dump_every": {"type": "integer", "default": 20},
        "min_atoms": {"type": "integer", "default": 200, "description": "Replicate the structure to at least this many atoms (and >= 2x the outer cutoff)."},
        "temperature_tolerance": {"type": "number", "default": 0.2, "description": "Runaway if the second-half mean temperature exceeds target x (1 + this); equilibrated if within this fraction."},
        "max_pe_jump_per_atom": {"type": "number", "default": 1.0, "description": "kcal/mol/atom change in potential energy between consecutive dumps that counts as a blow-up."},
        "close_contact_margin": {"type": "number", "default": 0.1, "description": "Å above the inner cutoff that counts as a close contact."},
        "reference_xyzf": {"type": ["string", "null"], "description": "Reference frames (e.g. DFT-MD) for the RDF comparison."},
        "rdf_rmax": {"type": "number", "default": 6.0},
        "harvest_close": {"type": "integer", "default": 20},
        "harvest_other": {"type": "integer", "default": 20},
        "workers": {"type": "integer", "default": 1},
        "run_timeout_s": {"type": "integer", "default": 1800},
        "seed": {"type": "integer", "default": 7},
        "machine": {"type": ["string", "null"]},
        "queue": {"type": "string", "default": "debug"},
        "walltime_hours": {"type": "number", "default": 1.0},
    },
}


def _floats(s):
    return [float(x) for x in s.split(",") if x]


def add_arguments(parser) -> None:
    parser.add_argument("--params", action="append", default=None, help="Repeatable: one per candidate model.")
    parser.add_argument("--structure-xyzf", dest="structure_xyzf", default=None)
    parser.add_argument("--frame-index", dest="frame_index", type=int, default=0)
    parser.add_argument("--prototype", type=json.loads, default=None)
    parser.add_argument("--elements", type=lambda s: [x for x in s.split(",") if x], default=None)
    parser.add_argument("--masses", type=json.loads, default=None)
    parser.add_argument("--temperatures", type=_floats, default=None)
    parser.add_argument("--nsteps", type=int, default=2000)
    parser.add_argument("--timestep", type=float, default=1.0)
    parser.add_argument("--dump-every", dest="dump_every", type=int, default=20)
    parser.add_argument("--min-atoms", dest="min_atoms", type=int, default=200)
    parser.add_argument("--temperature-tolerance", dest="temperature_tolerance", type=float, default=0.2)
    parser.add_argument("--max-pe-jump-per-atom", dest="max_pe_jump_per_atom", type=float, default=1.0)
    parser.add_argument("--close-contact-margin", dest="close_contact_margin", type=float, default=0.1)
    parser.add_argument("--reference-xyzf", dest="reference_xyzf", default=None)
    parser.add_argument("--rdf-rmax", dest="rdf_rmax", type=float, default=6.0)
    parser.add_argument("--harvest-close", dest="harvest_close", type=int, default=20)
    parser.add_argument("--harvest-other", dest="harvest_other", type=int, default=20)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--run-timeout-s", dest="run_timeout_s", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--machine", default=None)
    parser.add_argument("--queue", default="debug")
    parser.add_argument("--walltime-hours", dest="walltime_hours", type=float, default=1.0)


# ------------------------------------------------------------------ model facts

def pair_inner_cutoffs(params_path) -> dict:
    """{"A-B": s_minim} from a params.txt pair block (both orders as keys)."""
    lines = Path(params_path).read_text().splitlines()
    start = next(i for i, ln in enumerate(lines) if "PAIRIDX" in ln)
    out = {}
    for ln in lines[start + 1:]:
        toks = ln.split()
        if len(toks) < 5 or not toks[0].isdigit():
            break
        out[f"{toks[1]}-{toks[2]}"] = out[f"{toks[2]}-{toks[1]}"] = float(toks[3])
    return out


# ------------------------------------------------------------------ structures

def _start_frame(args, params_path):
    from . import benchmark
    from .lammps_run import replicate_for_cutoff

    if getattr(args, "prototype", None):
        from ..data_sources import convert

        at = benchmark._base_atoms(args)
        frame = convert.to_frame(at.get_chemical_symbols(), at.cell.array, at.positions)
    elif getattr(args, "structure_xyzf", None):
        frame = xyzf_io.read_xyzf(args.structure_xyzf)[getattr(args, "frame_index", 0) or 0]
    else:
        raise ValueError("md-check needs --structure-xyzf or --prototype")
    frame, _ = replicate_for_cutoff(frame, params_path)
    min_atoms = getattr(args, "min_atoms", 200) or 0
    reps = [1, 1, 1]
    cell = np.asarray(frame.box if frame.non_ortho else np.diag(frame.box), dtype=float)
    while frame.natoms * reps[0] * reps[1] * reps[2] < min_atoms:
        widths = [w * r for w, r in zip(_widths(cell), reps)]
        reps[int(np.argmin(widths))] += 1  # grow the thinnest direction
    if reps != [1, 1, 1]:
        pos = np.asarray(frame.positions, dtype=float)
        shifts = [i * cell[0] + j * cell[1] + k * cell[2]
                  for i in range(reps[0]) for j in range(reps[1]) for k in range(reps[2])]
        big_cell = cell * np.asarray(reps)[:, None]
        n = frame.natoms * len(shifts)
        frame = xyzf_io.Frame(symbols=list(frame.symbols) * len(shifts),
                              positions=np.concatenate([pos + sh for sh in shifts]).tolist(), forces=[[0.0] * 3] * n,
                              box=big_cell.tolist() if frame.non_ortho else [float(x) for x in np.diag(big_cell)],
                              non_ortho=frame.non_ortho)
    return frame


# ------------------------------------------------------------------ analysis

def _to_atoms(symbols, cell, positions):
    from ase import Atoms

    return Atoms(symbols=symbols, cell=cell, positions=positions, pbc=True)


def parse_dump(path, elements):
    """[(symbols, cell 3x3, positions)] from a LAMMPS custom dump (ortho or triclinic)."""
    frames = []
    text = Path(path).read_text().split("ITEM: TIMESTEP")[1:]
    for block in text:
        lines = block.strip().splitlines()
        n = int(lines[lines.index("ITEM: NUMBER OF ATOMS") + 1])
        bi = next(i for i, ln in enumerate(lines) if ln.startswith("ITEM: BOX BOUNDS"))
        tri = "xy" in lines[bi]
        b = [[float(x) for x in lines[bi + k].split()] for k in (1, 2, 3)]
        if tri:
            (xlo_b, xhi_b, xy), (ylo_b, yhi_b, xz), (zlo, zhi, yz) = b
            xlo = xlo_b - min(0.0, xy, xz, xy + xz)
            xhi = xhi_b - max(0.0, xy, xz, xy + xz)
            ylo, yhi = ylo_b - min(0.0, yz), yhi_b - max(0.0, yz)
            cell = np.array([[xhi - xlo, 0, 0], [xy, yhi - ylo, 0], [xz, yz, zhi - zlo]])
        else:
            cell = np.diag([b[0][1] - b[0][0], b[1][1] - b[1][0], b[2][1] - b[2][0]])
        ai = next(i for i, ln in enumerate(lines) if ln.startswith("ITEM: ATOMS"))
        cols = lines[ai].split()[2:]
        it, ix, iy, iz = cols.index("type"), cols.index("x"), cols.index("y"), cols.index("z")
        rows = [ln.split() for ln in lines[ai + 1: ai + 1 + n]]
        syms = [elements[int(r[it]) - 1] for r in rows]
        pos = np.array([[float(r[ix]), float(r[iy]), float(r[iz])] for r in rows])
        frames.append((syms, cell, pos))
    return frames


def closest_pairs(atoms, rmax: float = 4.0) -> dict:
    """Smallest distance per element pair "A-B" (sorted symbols) within rmax."""
    from ase.neighborlist import neighbor_list

    i, j, d = neighbor_list("ijd", atoms, rmax)
    sym = np.array(atoms.get_chemical_symbols())
    out = {}
    for a, b, r in zip(sym[i], sym[j], d):
        key = "-".join(sorted((a, b)))
        if r < out.get(key, 1e9):
            out[key] = float(r)
    return out


def partial_rdf(frames_atoms, rmax: float, bin_width: float = 0.05) -> dict:
    """{"A-B": (r, g)} averaged over frames (ase neighbor lists: exact for any cell)."""
    from ase.neighborlist import neighbor_list

    edges = np.arange(0.0, rmax + bin_width, bin_width)
    centers = 0.5 * (edges[1:] + edges[:-1])
    shell = 4.0 * np.pi * centers ** 2 * bin_width
    hist, norm = {}, {}
    for at in frames_atoms:
        sym = np.array(at.get_chemical_symbols())
        vol = at.get_volume()
        i, j, d = neighbor_list("ijd", at, rmax)
        counts = {e: int((sym == e).sum()) for e in set(sym)}
        for a in counts:
            for b in counts:
                if a > b:
                    continue
                key = f"{a}-{b}"
                m = (sym[i] == a) & (sym[j] == b)
                h, _ = np.histogram(d[m], bins=edges)
                hist[key] = hist.get(key, 0) + h
                norm[key] = norm.get(key, 0.0) + counts[a] * counts[b] / vol
    return {k: (centers, hist[k] / (shell * norm[k])) for k in hist}


def rdf_distance(rdf_a: dict, rdf_b: dict) -> dict:
    """Mean |g_a - g_b| per shared pair (same bins)."""
    out = {}
    for k in sorted(set(rdf_a) & set(rdf_b)):
        ga, gb = rdf_a[k][1], rdf_b[k][1]
        n = min(len(ga), len(gb))
        out[k] = round(float(np.mean(np.abs(ga[:n] - gb[:n]))), 4)
    return out


def _thermo(log_text: str):
    """Rows of (step, temp, pe) from our thermo_style (step temp pe etotal press)."""
    rows = []
    for ln in log_text.splitlines():
        toks = ln.split()
        if len(toks) == 5 and re.match(r"^\d+$", toks[0]):
            try:
                rows.append((int(toks[0]), float(toks[1]), float(toks[2])))
            except ValueError:
                continue
    return rows


# ------------------------------------------------------------------ one run

def run_case(case: dict) -> dict:
    """LAMMPS NVT for one (model, temperature), then analysis. Never raises."""
    from types import SimpleNamespace

    from . import lammps_run

    out = Path(case["dir"])
    res = {"params": case["params"], "temperature": case["temperature"], "dir": str(out)}
    try:
        out.mkdir(parents=True, exist_ok=True)
        xyzf_io.write_xyzf([case["frame"]], out / "start.xyzf")
        text = lammps_run._render_input("md", "structure.data", str(Path(case["params"]).resolve()),
                                        temperature=case["temperature"], nsteps=case["nsteps"],
                                        timestep=case["timestep"], md_seed=case["seed"])
        text = text.replace("thermo 100", f"thermo {case['dump_every']}").replace(
            "dump 1 all custom 100", f"dump 1 all custom {case['dump_every']}")
        lammps_data.write_lammps_data(case["frame"], case["elements"], case["masses"], out / "structure.data")
        (out / "in.lammps").write_text(text)
        import subprocess

        from .. import config
        from ..hpc.local import singleton_env

        lmp = config.resolve_component("lammps_bin")
        try:
            proc = subprocess.run([str(lmp), "-in", "in.lammps", "-log", "log.lammps"], cwd=out, capture_output=True,
                                  text=True, env=singleton_env(), timeout=case["timeout_s"])
            (out / "stdout.log").write_text((proc.stdout or "")[-20000:] + (proc.stderr or "")[-5000:])
            finished = proc.returncode == 0
            err = None if finished else (proc.stdout or proc.stderr or "")[-400:]
        except subprocess.TimeoutExpired:
            finished, err = False, f"timeout after {case['timeout_s']} s"
        log = (out / "log.lammps").read_text() if (out / "log.lammps").is_file() else ""
        thermo = _thermo(log)
        res["steps_completed"] = thermo[-1][0] if thermo else 0
        second = [r for r in thermo if r[0] >= case["nsteps"] / 2]
        t_mean = float(np.mean([r[1] for r in second])) if second else None
        pe_ok = bool(thermo) and all(math.isfinite(r[2]) for r in thermo)
        res["mean_temperature_second_half"] = None if t_mean is None else round(t_mean, 1)
        res["lost_atoms"] = "Lost atoms" in log
        natoms = case["frame"].natoms
        # second half only: the first dumps carry the start-up transient (a crystal started at T converts
        # kinetic into potential energy, ~2.6 kcal/mol/atom in 50 fs for Cu-Zr at 1200 K)
        jumps = [abs(b[2] - a[2]) / natoms for a, b in zip(second, second[1:])]
        res["max_pe_jump_per_atom"] = round(max(jumps), 4) if jumps else None
        # Instability = runaway: overheating, energy jumps, lost atoms, a crash. Running cooler than the
        # target is not instability (a crystal started at T equilibrates near T/2 first); see "equilibrated".
        hot = t_mean is not None and t_mean > (1 + case["t_tol"]) * case["temperature"]
        jump = bool(jumps) and max(jumps) > case["max_pe_jump"]
        res["equilibrated"] = t_mean is not None and abs(t_mean - case["temperature"]) <= case["t_tol"] * case["temperature"]
        res["stable"] = bool(finished and res["steps_completed"] >= case["nsteps"] and pe_ok and not hot and not jump
                             and not res["lost_atoms"])
        reasons = [msg for cond, msg in ((not finished, "LAMMPS did not finish"), (res["lost_atoms"], "lost atoms"),
                                         (not pe_ok, "non-finite energy"), (hot, "temperature ran away"),
                                         (jump, "potential-energy jump between dumps")) if cond]
        if reasons:
            res["instability"] = reasons
        if err and not res["stable"]:
            res["error"] = err

        frames = parse_dump(out / "dump.out", case["elements"]) if (out / "dump.out").is_file() else []
        inner = case["inner"]
        margin = case["margin"]
        close_idx, below_idx, closest = [], [], {}
        atoms_list = []
        for k, (syms, cell, pos) in enumerate(frames):
            at = _to_atoms(syms, cell, pos)
            atoms_list.append(at)
            cp = closest_pairs(at, rmax=max(inner.values()) + margin + 0.5)
            for key, r in cp.items():
                closest[key] = min(closest.get(key, 1e9), r)
            if any(r < inner.get(key, 0.0) for key, r in cp.items()):
                below_idx.append(k)
            if any(r < inner.get(key, 0.0) + margin for key, r in cp.items()):
                close_idx.append(k)
        n = max(len(frames), 1)
        res["n_frames"] = len(frames)
        res["close_contact_fraction"] = round(len(close_idx) / n, 3)
        res["below_inner_cutoff_frames"] = len(below_idx)
        res["closest_distance"] = {k: round(v, 3) for k, v in sorted(closest.items())}
        half = atoms_list[len(atoms_list) // 2:]
        if half and case["rdf_rmax"]:
            rmax = min(case["rdf_rmax"], 0.5 * min(_widths(half[0].cell.array)))
            rdf = partial_rdf(half, rmax)
            np.savez(out / "rdf.npz", **{k: np.vstack(v) for k, v in rdf.items()})
            res["rdf_file"] = str(out / "rdf.npz")
            res["rdf_rmax"] = round(rmax, 3)
            if case.get("reference_rdf") is not None:
                ref = {k: (np.asarray(v[0]), np.asarray(v[1])) for k, v in case["reference_rdf"].items()}
                res["rdf_distance"] = rdf_distance(rdf, ref)
        res["_close_idx"], res["_all_idx"] = close_idx, list(range(len(frames)))
    except Exception as exc:  # noqa: BLE001 - one failed run is a result, not a crash
        res.update({"stable": False, "error": str(exc)[-600:]})
    return res


def _widths(cell):
    cell = np.asarray(cell, dtype=float)
    vol = abs(np.linalg.det(cell))
    return [vol / np.linalg.norm(np.cross(cell[(k + 1) % 3], cell[(k + 2) % 3])) for k in range(3)]


def _reference_rdf(path, rmax):
    frames = xyzf_io.read_xyzf(path)
    atoms = []
    for f in frames:
        cell = np.asarray(f.box if f.non_ortho else np.diag(f.box), dtype=float)
        atoms.append(_to_atoms(f.symbols, cell, f.positions))
    return partial_rdf(atoms, rmax)


# ------------------------------------------------------------------ stage

def _submit(args, out: Path) -> dict:
    from .. import hpc, machines

    payload = {k: v for k, v in vars(args).items()
               if not k.startswith("_") and k not in ("machine", "json_in", "json_out", "describe", "force", "dry_run",
                                                      "output_dir", "stage", "queue", "walltime_hours")}
    profile = machines.load_profile(args.machine)
    n_cases = len(args.params) * len(getattr(args, "temperatures", None) or [300.0])
    cores = min(profile.default_ntasks_per_node or 112, max(1, n_cases))
    payload["workers"] = cores
    inp = out / "md_check_input.json"
    inp.write_text(json.dumps(payload, indent=1, default=str))
    run_dir = out / "run"
    cmd = f"{sys.executable} -m agentic_chimes.cli md-check --json-in {inp} --output-dir {run_dir} --force > md_check.out 2>&1"
    handle = hpc.submit_job(profile, job_name="md-check", commands=[f"cd {out.resolve()}", cmd], work_dir=out, nodes=1,
                            ntasks_per_node=cores, walltime_hours=getattr(args, "walltime_hours", 1.0) or 1.0,
                            queue=getattr(args, "queue", "debug") or "debug", dry_run=bool(getattr(args, "dry_run", False)))
    return {"submitted": not handle.dry_run, "dry_run": handle.dry_run, "job_id": handle.job_id, "job_file": str(handle.job_file),
            "n_runs": n_cases, "cores_requested": cores, "results_when_done": str(run_dir / "md_check.json")}


def run(args) -> dict:
    params = [str(Path(p).resolve()) for p in (getattr(args, "params", None) or [])]
    if not params:
        raise ValueError("md-check needs at least one --params")
    if not getattr(args, "elements", None) or not getattr(args, "masses", None):
        raise ValueError("md-check needs --elements and --masses (the order and masses the models were fitted with)")
    args.params = params
    out = Path(getattr(args, "output_dir", None) or ".").resolve()
    out.mkdir(parents=True, exist_ok=True)
    if getattr(args, "machine", None):
        return _submit(args, out)

    temps = getattr(args, "temperatures", None) or [300.0]
    rmax = getattr(args, "rdf_rmax", 6.0) or 0.0
    ref_rdf = None
    if getattr(args, "reference_xyzf", None) and rmax:
        ref_rdf = {k: (v[0].tolist(), v[1].tolist()) for k, v in _reference_rdf(args.reference_xyzf, rmax).items()}
    from .evaluate import max_outer_cutoff

    frame = _start_frame(args, max(params, key=max_outer_cutoff))
    cases = []
    for m, p in enumerate(params):
        for t in temps:
            cases.append({
                "params": p, "temperature": float(t), "dir": str(out / f"model{m}_T{int(t)}"), "frame": frame,
                "elements": args.elements, "masses": args.masses, "nsteps": args.nsteps, "timestep": args.timestep,
                "dump_every": getattr(args, "dump_every", 20) or 20, "seed": getattr(args, "seed", 7) or 7,
                "timeout_s": getattr(args, "run_timeout_s", 1800) or 1800,
                "t_tol": getattr(args, "temperature_tolerance", 0.2), "inner": pair_inner_cutoffs(p),
                "max_pe_jump": getattr(args, "max_pe_jump_per_atom", 1.0) or 1.0,
                "margin": getattr(args, "close_contact_margin", 0.1), "rdf_rmax": rmax, "reference_rdf": ref_rdf,
            })
    workers = max(1, getattr(args, "workers", 1) or 1)
    if workers > 1 and len(cases) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(cases))) as pool:
            results = list(pool.map(run_case, cases))
    else:
        results = [run_case(c) for c in cases]

    # harvest frames for active learning
    rng = random.Random(getattr(args, "seed", 7))
    harvest, n_close, n_other = [], 0, 0
    for r in results:
        dump = Path(r["dir"]) / "dump.out"
        if not dump.is_file():
            continue
        frames = parse_dump(dump, args.elements)
        close = list(r.get("_close_idx", []))
        others = [i for i in r.get("_all_idx", []) if i not in set(close)]
        rng.shuffle(close)
        rng.shuffle(others)
        pick = close[: getattr(args, "harvest_close", 20)] + others[: getattr(args, "harvest_other", 20)]
        n_close += min(len(close), getattr(args, "harvest_close", 20))
        n_other += min(len(others), getattr(args, "harvest_other", 20))
        for i in sorted(pick):
            syms, cell, pos = frames[i]
            harvest.append(xyzf_io.Frame(symbols=syms, positions=pos.tolist(), forces=[[0.0] * 3] * len(syms),
                                         box=cell.tolist(), non_ortho=True))
    harvest_path = None
    if harvest:
        harvest_path = out / "harvest.xyzf"
        xyzf_io.write_xyzf(harvest, harvest_path)
    for r in results:
        r.pop("_close_idx", None)
        r.pop("_all_idx", None)

    per_model = []
    for p in params:
        rs = [r for r in results if r["params"] == p]
        dist = [np.mean(list(r["rdf_distance"].values())) for r in rs if r.get("rdf_distance")]
        per_model.append({
            "params": p,
            "stable_at_all_temperatures": all(r.get("stable") for r in rs),
            "unstable_temperatures": [r["temperature"] for r in rs if not r.get("stable")],
            "not_equilibrated_temperatures": [r["temperature"] for r in rs if r.get("stable") and not r.get("equilibrated")],
            "max_close_contact_fraction": max((r.get("close_contact_fraction", 0.0) for r in rs), default=None),
            "below_inner_cutoff_frames": sum(r.get("below_inner_cutoff_frames", 0) for r in rs),
            "mean_rdf_distance": round(float(np.mean(dist)), 4) if dist else None,
        })
    notes = []
    if not any(m["stable_at_all_temperatures"] for m in per_model):
        notes.append("no candidate is stable at every temperature: the data (short-range coverage) is the limit; "
                     "label harvest.xyzf and refit (active learning) rather than choosing among these")
    if any(m["below_inner_cutoff_frames"] for m in per_model):
        notes.append("some runs sampled distances inside a model's inner cutoff (the penalty region): those are exactly "
                     "the close-contact frames active learning should label")
    if any(m["not_equilibrated_temperatures"] for m in per_model):
        notes.append("some stable runs had not reached the target temperature: lengthen nsteps before comparing RDFs")
    if ref_rdf is None:
        notes.append("no reference_xyzf: RDFs were saved but not compared; give DFT-MD frames at these conditions to rank "
                     "candidates by structure (Lindsey 2019)")
    report = {"structure_natoms": frame.natoms, "temperatures": temps, "nsteps": args.nsteps, "timestep_fs": args.timestep,
              "models": per_model, "runs": results, "harvest_xyzf": str(harvest_path) if harvest_path else None,
              "n_harvested": len(harvest), "n_harvested_close_contact": n_close, "n_harvested_other": n_other,
              "notes": notes}
    (out / "md_check.json").write_text(json.dumps(report, indent=1, default=str))
    return {**{k: report[k] for k in ("models", "harvest_xyzf", "n_harvested", "notes")},
            "report": str(out / "md_check.json")}
