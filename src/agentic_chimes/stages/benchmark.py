"""Strong and weak scaling of a ChIMES model in LAMMPS, and the cost model
downstream compute requests are sized from.

Submit mode builds one directory per case and submits a single Slurm job
that runs them in order of size:

  strong  a fixed system (~`strong_atoms`) on each rank count
  weak    ~`atoms_per_rank` x ranks atoms on each rank count

Every case runs short NVT MD (`steps`, after `warmup_steps` that are not
timed) on an orthorhombic supercell of the given structure. Collect mode
(`--collect DIR`) parses each LAMMPS log's `Loop time` and derives:

  core_s_per_atom_step = ranks x loop_time / (steps x atoms)   (the cost unit)
  strong efficiency    = (T_ref x P_ref) / (T x P), relative to the fewest ranks
  weak efficiency      = per-atom time at the fewest ranks / per-atom time
  ns/day               = steps x timestep / loop_time

and a recommendation: the largest rank count still >= `min_efficiency`
efficient, and CPU-hour estimates for typical requests. A model's cost per
atom-step depends on its own cutoffs and cluster counts, which is why the
final model is benchmarked, not a generic one.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

from .. import config, hpc, machines
from ..data_sources import convert
from ..io import lammps_data
from ..io import xyzf as xyzf_io
from .lammps_run import _render_input

NAME = "benchmark"
SUMMARY = "Strong/weak scaling of a ChIMES model in LAMMPS (Slurm submit, then --collect) -> cost model for compute requests."
DEFAULT_RANKS = [1, 2, 4, 8, 16, 28, 56, 112]
CUBIC_PROTOTYPES = ("fcc", "bcc", "cesiumchloride", "diamond", "rocksalt", "zincblende", "sc")
SCHEMA = {
    "type": "object",
    "properties": {
        "params": {"type": ["string", "null"], "description": "params.txt of the model to benchmark."},
        "structure_xyzf": {"type": ["string", "null"], "description": "Orthorhombic frame to replicate (see frame_index)."},
        "frame_index": {"type": "integer", "default": 0},
        "prototype": {"type": ["object", "null"], "description": "ase.build.bulk kwargs instead of a file, e.g. {\"name\":\"CuZr\",\"crystalstructure\":\"cesiumchloride\",\"a\":3.26}."},
        "elements": {"type": ["array", "null"], "items": {"type": "string"}},
        "masses": {"type": ["object", "null"]},
        "ranks": {"type": "array", "items": {"type": "integer"}, "default": DEFAULT_RANKS},
        "nodes": {"type": "array", "items": {"type": "integer"}, "default": [1], "description": "Extra multi-node points use nodes x cores-per-node ranks."},
        "modes": {"type": "array", "items": {"type": "string"}, "default": ["strong", "weak"]},
        "strong_atoms": {"type": "integer", "default": 16000},
        "atoms_per_rank": {"type": "integer", "default": 250},
        "steps": {"type": "integer", "default": 100},
        "warmup_steps": {"type": "integer", "default": 10},
        "timestep": {"type": "number", "default": 1.0, "description": "fs"},
        "temperature": {"type": "number", "default": 300.0},
        "case_timeout_s": {"type": "integer", "default": 900},
        "machine": {"type": ["string", "null"]},
        "queue": {"type": "string", "default": "debug"},
        "walltime_hours": {"type": "number", "default": 1.0},
        "collect": {"type": ["string", "null"], "description": "A submitted benchmark directory: parse results."},
        "min_efficiency": {"type": "number", "default": 0.7},
    },
}


def _ints(s):
    return [int(x) for x in s.split(",") if x]


def add_arguments(parser) -> None:
    parser.add_argument("--params", default=None)
    parser.add_argument("--structure-xyzf", dest="structure_xyzf", default=None)
    parser.add_argument("--frame-index", dest="frame_index", type=int, default=0)
    parser.add_argument("--prototype", type=json.loads, default=None)
    parser.add_argument("--elements", type=lambda s: [x for x in s.split(",") if x], default=None)
    parser.add_argument("--masses", type=json.loads, default=None)
    parser.add_argument("--ranks", type=_ints, default=None)
    parser.add_argument("--nodes", type=_ints, default=None)
    parser.add_argument("--modes", type=lambda s: [x for x in s.split(",") if x], default=None)
    parser.add_argument("--strong-atoms", dest="strong_atoms", type=int, default=16000)
    parser.add_argument("--atoms-per-rank", dest="atoms_per_rank", type=int, default=250)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup-steps", dest="warmup_steps", type=int, default=10)
    parser.add_argument("--timestep", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--case-timeout-s", dest="case_timeout_s", type=int, default=900)
    parser.add_argument("--machine", default=None)
    parser.add_argument("--queue", default="debug")
    parser.add_argument("--walltime-hours", dest="walltime_hours", type=float, default=1.0)
    parser.add_argument("--collect", default=None)
    parser.add_argument("--min-efficiency", dest="min_efficiency", type=float, default=0.7)


# ------------------------------------------------------------------ structures

def _base_atoms(args):
    from ase.build import bulk

    if getattr(args, "prototype", None):
        kw = dict(args.prototype)
        name = kw.pop("name")
        if kw.get("crystalstructure") in CUBIC_PROTOTYPES:
            kw.setdefault("cubic", True)
        else:
            kw.setdefault("orthorhombic", True)
        return bulk(name, **kw)
    if getattr(args, "structure_xyzf", None):
        fr = xyzf_io.read_xyzf(args.structure_xyzf)[getattr(args, "frame_index", 0) or 0]
        if fr.non_ortho:
            raise ValueError("benchmark needs an orthorhombic structure (LAMMPS data writing here is orthorhombic-only); "
                             "use --prototype or a data-generate structure")
        return convert.frame_to_atoms(fr)
    raise ValueError("benchmark needs --structure-xyzf or --prototype")


def replicate_to(atoms, target: int):
    """Near-cubic integer repetition with natoms as close to `target` as possible."""
    n0 = len(atoms)
    lengths = np.asarray(atoms.cell.lengths())
    k = max(1.0, (target / n0) ** (1 / 3))
    best = None
    for scale in np.linspace(0.8, 1.3, 26):
        reps = [max(1, int(round(k * scale * lengths.mean() / L))) for L in lengths]
        n = n0 * reps[0] * reps[1] * reps[2]
        if best is None or abs(n - target) < abs(best[1] - target):
            best = (reps, n)
    return atoms.repeat(best[0]), best[0]


def _cases(args, cores_per_node: int):
    ranks = sorted(set(getattr(args, "ranks", None) or DEFAULT_RANKS))
    nodes = sorted(set(getattr(args, "nodes", None) or [1]))
    points = [(r, 1) for r in ranks if r <= cores_per_node] + [(n * cores_per_node, n) for n in nodes if n > 1]
    cases = []
    for mode in getattr(args, "modes", None) or ["strong", "weak"]:
        for r, n in points:
            target = args.strong_atoms if mode == "strong" else args.atoms_per_rank * r
            cases.append({"mode": mode, "ranks": r, "nodes": n, "target_atoms": target})
    return cases


# ------------------------------------------------------------------ submit

def _submit(args) -> dict:
    for req in ("params", "elements", "masses", "machine"):
        if not getattr(args, req, None):
            raise ValueError(f"benchmark requires --{req.replace('_', '-')}")
    profile = machines.load_profile(args.machine)
    cpn = profile.default_ntasks_per_node or 112
    out = Path(getattr(args, "output_dir", None) or ".").resolve()
    out.mkdir(parents=True, exist_ok=True)
    base = _base_atoms(args)
    params = Path(args.params).resolve()
    lmp = config.resolve_component("lammps_bin", required=not getattr(args, "dry_run", False))
    text = _render_input("md", "structure.data", str(params), temperature=args.temperature,
                         nsteps=args.warmup_steps, timestep=args.timestep)
    text = (text.replace("thermo 100", "thermo 50")
            .replace("dump 1 all custom 100 dump.out id type x y z fx fy fz\n", "")
            .replace("dump_modify 1 sort id\n", "")) + f"run {args.steps}\n"
    cases = _cases(args, cpn)
    commands = []
    for c in sorted(cases, key=lambda c: (c["target_atoms"] / c["ranks"], c["ranks"])):
        atoms, reps = replicate_to(base, c["target_atoms"])
        c.update({"atoms": len(atoms), "reps": reps})
        d = out / f"{c['mode']}_r{c['ranks']:04d}"
        d.mkdir(exist_ok=True)
        frame = convert.to_frame(atoms.get_chemical_symbols(), atoms.cell[:], atoms.positions)
        lammps_data.write_lammps_data(frame, args.elements, args.masses, d / "structure.data")
        (d / "in.lammps").write_text(text)
        c["dir"] = str(d)
        commands += [f"cd {d}",
                     f"timeout {args.case_timeout_s} srun -N {c['nodes']} -n {c['ranks']} {lmp} -in in.lammps -log log.lammps "
                     f"> stdout.log 2>&1 || echo \"exit $?\" > FAILED",
                     "cd - > /dev/null"]
    settings = {k: getattr(args, k, None) for k in ("params", "prototype", "structure_xyzf", "frame_index", "elements", "steps",
                                                   "warmup_steps", "timestep", "temperature", "strong_atoms",
                                                   "atoms_per_rank", "min_efficiency")}
    (out / "benchmark_cases.json").write_text(json.dumps({"cases": cases, "settings": settings, "cores_per_node": cpn,
                                                          "machine": args.machine}, indent=1, default=str))
    handle = hpc.submit_job(profile, job_name="chimes-bench", commands=commands, work_dir=out,
                            nodes=max(c["nodes"] for c in cases), ntasks_per_node=cpn,
                            walltime_hours=args.walltime_hours, queue=args.queue, dry_run=bool(getattr(args, "dry_run", False)))
    return {"work_dir": str(out), "n_cases": len(cases), "job_id": handle.job_id, "dry_run": handle.dry_run,
            "job_file": str(handle.job_file),
            "cases": [{k: c[k] for k in ("mode", "ranks", "nodes", "atoms")} for c in cases],
            "next": f"chimes-agent benchmark --collect {out}"}


# ------------------------------------------------------------------ collect

_LOOP = re.compile(r"Loop time of ([\d.eE+-]+) on (\d+) procs for (\d+) steps with (\d+) atoms")


def parse_log(text: str):
    """(loop_time_s, procs, steps, atoms) of the last `run` in a LAMMPS log."""
    hits = _LOOP.findall(text)
    if not hits:
        return None
    t, p, s, n = hits[-1]
    return float(t), int(p), int(s), int(n)


def _thermo_ok(text: str) -> bool:
    return not re.search(r"\bnan\b|ERROR|Lost atoms", text, re.IGNORECASE)


def analyze(cases: list, timestep_fs: float, min_eff: float) -> dict:
    out = {}
    for mode in ("strong", "weak"):
        rows = sorted([c for c in cases if c["mode"] == mode and c.get("loop_time_s")], key=lambda c: c["ranks"])
        if not rows:
            continue
        ref = rows[0]
        for c in rows:
            c["core_s_per_atom_step"] = c["ranks"] * c["loop_time_s"] / (c["steps"] * c["atoms"])
            c["ns_per_day"] = c["steps"] * timestep_fs * 1e-6 / c["loop_time_s"] * 86400
            if mode == "strong":
                c["speedup"] = ref["loop_time_s"] / c["loop_time_s"]
                c["efficiency"] = (ref["loop_time_s"] * ref["ranks"]) / (c["loop_time_s"] * c["ranks"])
            else:
                c["efficiency"] = ref["core_s_per_atom_step"] / c["core_s_per_atom_step"]
        good = [c for c in rows if c["efficiency"] >= min_eff]
        rec = max(good, key=lambda c: c["ranks"]) if good else rows[0]
        out[mode] = {"rows": [{k: (round(v, 9) if isinstance(v, float) else v) for k, v in c.items() if k != "dir"} for c in rows],
                     "recommended_ranks": rec["ranks"], "recommended_efficiency": round(rec["efficiency"], 3),
                     "reference_ranks": ref["ranks"]}
    return out


def cost_model(results: dict) -> dict:
    """Cost per atom-step as a function of ranks used, from weak scaling
    (production runs sit near fixed atoms/rank); strong scaling if that is
    all there is. Per-core cost rises as a node fills (shared memory
    bandwidth, communication), so estimates look up the packing a run
    would actually use instead of one number: on Cu-Zr a full 112-rank
    node cost 1.8x a single rank per atom-step."""
    for mode in ("weak", "strong"):
        if mode in results:
            rows = results[mode]["rows"]
            rec = next(r for r in rows if r["ranks"] == results[mode]["recommended_ranks"])
            return {"core_s_per_atom_step": rec["core_s_per_atom_step"], "from": f"{mode} scaling at {rec['ranks']} ranks",
                    "atoms_per_rank": max(1, round(rec["atoms"] / rec["ranks"])),
                    "by_ranks": {r["ranks"]: r["core_s_per_atom_step"] for r in rows}}
    return {}


def _unit_at(model: dict, ranks: int, cores_per_node: int) -> float:
    table = {int(k): v for k, v in (model.get("by_ranks") or {}).items()}
    if not table:
        return model["core_s_per_atom_step"]
    want = cores_per_node if ranks >= cores_per_node else ranks
    below = [r for r in table if r <= want]
    return table[max(below)] if below else table[min(table)]


def estimate(model: dict, atoms: int, ns: float, timestep_fs: float, cores_per_node: int = 112) -> dict:
    """CPU-hours for `atoms` atoms over `ns` ns, at the benchmarked atoms/rank."""
    steps = ns * 1e6 / timestep_fs
    ranks = max(1, round(atoms / model["atoms_per_rank"]))
    nodes = math.ceil(ranks / cores_per_node)
    if nodes > 1:
        ranks = nodes * cores_per_node  # whole nodes are what gets allocated
    unit = _unit_at(model, ranks, cores_per_node)
    core_h = unit * atoms * steps / 3600
    return {"atoms": atoms, "ns": ns, "timestep_fs": timestep_fs, "cpu_hours": round(core_h, 1),
            "core_s_per_atom_step_used": unit, "suggested_ranks": ranks, "suggested_nodes": nodes,
            "wall_hours_at_suggested": round(core_h / ranks, 2)}


def _collect(args) -> dict:
    d = Path(args.collect).resolve()
    meta = json.loads((d / "benchmark_cases.json").read_text())
    cases, s = meta["cases"], meta["settings"]
    failed = []
    for c in cases:
        cdir = Path(c["dir"])
        log = cdir / "log.lammps"
        text = log.read_text(errors="replace") if log.is_file() else ""
        parsed = parse_log(text)
        if not parsed or (cdir / "FAILED").exists():
            c["status"] = "failed" if (log.is_file() or (cdir / "FAILED").exists()) else "missing"
            failed.append({k: c.get(k) for k in ("mode", "ranks", "atoms", "status")})
            continue
        c["loop_time_s"], c["procs"], c["steps"], c["atoms"] = parsed
        c["status"] = "done" if _thermo_ok(text) else "unstable"
    results = analyze(cases, s["timestep"], s.get("min_efficiency") or 0.7)
    model = cost_model(results)
    cpn = meta.get("cores_per_node", 112)
    estimates = [estimate(model, n, 1.0, s["timestep"], cpn) for n in (1_000, 10_000, 100_000, 1_000_000)] if model else []
    notes = []
    max_nodes = max((c.get("nodes", 1) for c in cases if c.get("loop_time_s")), default=1)
    if any(e["suggested_nodes"] > max_nodes for e in estimates):
        notes.append(f"estimates beyond {max_nodes} node(s) extrapolate the full-node cost; inter-node communication "
                     "was not measured -- rerun with --nodes 2,4 before a large multi-node request")
    report = {"work_dir": str(d), "machine": meta.get("machine"), "settings": s, "results": results, "cost_model": model,
              "estimates_1ns": estimates, "failed": failed, "notes": notes,
              "unstable": [c["ranks"] for c in cases if c.get("status") == "unstable"]}
    (d / "benchmark.json").write_text(json.dumps(report, indent=1, default=str))
    return {"benchmark": str(d / "benchmark.json"), "cost_model": model,
            "strong": {k: results.get("strong", {}).get(k) for k in ("recommended_ranks", "recommended_efficiency")},
            "weak": {k: results.get("weak", {}).get(k) for k in ("recommended_ranks", "recommended_efficiency")},
            "estimates_1ns": estimates, "failed": failed, "unstable": report["unstable"], "notes": notes}


def run(args) -> dict:
    if getattr(args, "collect", None):
        return _collect(args)
    return _submit(args)
