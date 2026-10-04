"""Single-point/MD via the ChIMES-patched `lmp_mpi_chimes` build (`chimes-agent
setup --component lammps`). Runs locally, or with `--machine` as one Slurm
job that re-invokes this stage on the compute node (the inputs are written
and checked here first, so a bad structure or mass fails before anything is
submitted); the job's result lands in `run/lammps_run.json`.

Single-point mode (`run 0`) is the documented independent cross-check
against the ctypes evaluator (stages/evaluate.py): both read the same
params.txt and should agree on forces/energy for a given configuration.
Forces are parsed from a LAMMPS dump file (a stable, simple format);
potential energy is parsed from the thermo log we control the format of
(`thermo_style custom step pe`) -- a best-effort parse, not fatal if it
fails, since forces from the dump are the primary reliable output.

Cells thinner than twice the model's outer cutoff are replicated first
(`replicate`, default true), exactly as `evaluate` does: chimesFF in LAMMPS
gets forces right on such cells but under-counts the energy (by 30-50
kcal/mol on 2-atom MatPES Cu-Zr cells; exact once replicated). Single-point
results are reported per original cell (energy / copies, the first copy's
forces); MD runs on the supercell.

Correctness-critical: `elements` (LAMMPS atom-type order) and `masses`
must match what the params.txt being tested was fit with -- see
io/lammps_data.py's docstring.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

from .. import config
from ..io import lammps_data, xyzf as xyzf_io
from ..io import fs

NAME = "lammps-run"
SUPPORTS_DRY_RUN = True
SUMMARY = "Single-point/MD via lmp_mpi_chimes: local, or one Slurm job with --machine."
SCHEMA = {
    "type": "object",
    "required": ["params", "structure_xyzf"],
    "properties": {
        "params": {"type": "string"},
        "structure_xyzf": {"type": "string", "description": "A .xyzf file; frame_index selects which frame (default 0)."},
        "frame_index": {"type": "integer", "default": 0},
        "elements": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Optional: checked against params.txt (types and order come from the model)."},
        "masses": {"type": ["object", "null"], "additionalProperties": {"type": "number"}, "description": "Optional: checked against params.txt; LAMMPS matches types by mass, so a mismatch is refused."},
        "mode": {"type": "string", "enum": ["single_point", "md"], "default": "single_point"},
        "temperature": {"type": "number", "default": 300.0},
        "nsteps": {"type": "integer", "default": 1000},
        "timestep": {"type": "number", "default": 1.0, "description": "fs (units real)"},
        "md_seed": {"type": "integer", "default": 12345},
        "lammps_bin": {"type": ["string", "null"]},
        "nprocs": {"type": "integer", "default": 1},
        "replicate": {"type": "boolean", "default": True, "description": "Replicate cells thinner than 2x the outer cutoff (LAMMPS energies are wrong on them)."},
        "machine": {"type": ["string", "null"], "description": "Submit as one Slurm job on this machine profile instead of running here."},
        "queue": {"type": "string", "default": "debug"},
        "walltime_hours": {"type": "number", "default": 1.0},
        "nodes": {"type": "integer", "default": 1},
        "ntasks_per_node": {"type": ["integer", "null"], "description": "MPI ranks per node for the job. Default: sized to the system (~250 atoms per rank), at most the profile's node."},
    },
}


def add_arguments(parser) -> None:
    parser.add_argument("--params", default=None)
    parser.add_argument("--structure-xyzf", dest="structure_xyzf", default=None)
    parser.add_argument("--frame-index", dest="frame_index", type=int, default=0)
    parser.add_argument("--elements", type=lambda s: s.split(","), default=None)
    parser.add_argument("--masses", type=json.loads, default=None)
    parser.add_argument("--mode", choices=["single_point", "md"], default="single_point")
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--nsteps", type=int, default=1000)
    parser.add_argument("--timestep", type=float, default=1.0)
    parser.add_argument("--md-seed", dest="md_seed", type=int, default=12345)
    parser.add_argument("--lammps-bin", dest="lammps_bin", default=None)
    parser.add_argument("--nprocs", type=int, default=1)
    parser.add_argument("--no-replicate", dest="replicate", action="store_false", default=True)
    parser.add_argument("--machine", default=None)
    parser.add_argument("--queue", default="debug")
    parser.add_argument("--walltime-hours", dest="walltime_hours", type=float, default=1.0)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--ntasks-per-node", dest="ntasks_per_node", type=int, default=None)


def _render_input(mode: str, data_file: str, params_file: str, *, temperature=300.0, nsteps=1000, timestep=1.0, md_seed=12345) -> str:
    lines = [
        "units real",
        "atom_style atomic",
        "boundary p p p",
        "atom_modify sort 0 0.0",
        f"read_data {data_file}",
        "pair_style chimesFF",
        f"pair_coeff * * {params_file}",
        "neighbor 2.0 bin",
        "neigh_modify delay 0 every 1 check yes",
    ]
    if mode == "single_point":
        lines += [
            "thermo 1",
            "thermo_style custom step pe",
            "dump 1 all custom 1 dump.out id type x y z fx fy fz",
            "dump_modify 1 sort id",
            "run 0",
        ]
    else:
        lines += [
            f"velocity all create {temperature} {md_seed}",
            f"fix 1 all nvt temp {temperature} {temperature} 100.0",
            f"timestep {timestep}",
            "thermo 100",
            "thermo_style custom step temp pe etotal press",
            "dump 1 all custom 100 dump.out id type x y z fx fy fz",
            "dump_modify 1 sort id",
            f"run {nsteps}",
        ]
    return "\n".join(lines) + "\n"


def _parse_dump_forces(dump_path: Path):
    text = dump_path.read_text()
    blocks = text.split("ITEM: TIMESTEP")
    last = blocks[-1]
    lines = last.splitlines()
    idx = next(i for i, ln in enumerate(lines) if ln.startswith("ITEM: ATOMS"))
    columns = lines[idx].split()[2:]  # e.g. ["id","type","x","y","z","fx","fy","fz"]
    fx_i, fy_i, fz_i = columns.index("fx"), columns.index("fy"), columns.index("fz")
    id_i = columns.index("id")
    rows = []
    for ln in lines[idx + 1 :]:
        if not ln.strip():
            continue
        toks = ln.split()
        rows.append((int(toks[id_i]), [float(toks[fx_i]), float(toks[fy_i]), float(toks[fz_i])]))
    rows.sort(key=lambda r: r[0])
    return [f for _, f in rows]


def _parse_log_pe(log_text: str) -> Optional[float]:
    """Our thermo_style is always exactly "step pe" (2 columns), so the
    header is deterministic modulo LAMMPS' own display label for "pe"
    (rendered "PotEng", not "PE") -- match on the first token being "Step"
    rather than hardcoding the second column's label."""
    lines = log_text.splitlines()
    for i, ln in enumerate(lines):
        toks = ln.split()
        if len(toks) == 2 and toks[0] == "Step":
            for data_ln in lines[i + 1 :]:
                data_toks = data_ln.split()
                if len(data_toks) == 2 and re.match(r"^-?\d", data_toks[0]):
                    return float(data_toks[1])
            break
    return None


def replicate_for_cutoff(frame, params_path):
    """(frame, n_copies): the frame replicated until every perpendicular width
    exceeds 2x the model's outer cutoff (unchanged when already wide enough)."""
    import numpy as np

    from . import evaluate

    big_pos, big_sym, big_cell, ncopies = evaluate._supercell(frame, evaluate.max_outer_cutoff(params_path))
    if ncopies == 1:
        return frame, 1
    n = len(big_sym)
    box = big_cell.tolist() if frame.non_ortho else [float(x) for x in np.diag(big_cell)]
    big = xyzf_io.Frame(symbols=big_sym, positions=big_pos.tolist(), forces=[[0.0, 0.0, 0.0]] * n, box=box,
                        non_ortho=frame.non_ortho)
    return big, ncopies


ATOMS_PER_RANK = 250      # below this, more MPI ranks cost more than they return (benchmark, Cu-Zr)


def _parse_thermo_last(log_text: str) -> Optional[dict]:
    """The last row of the last thermo table (MD mode: step temp pe etotal press)."""
    lines = log_text.splitlines()
    header, last = None, None
    for i, ln in enumerate(lines):
        toks = ln.split()
        if toks and toks[0] == "Step" and len(toks) > 2:
            header = toks
            for data_ln in lines[i + 1:]:
                d = data_ln.split()
                if len(d) != len(header) or not re.match(r"^-?\d", d[0]):
                    break
                last = d
    if not header or not last:
        return None
    try:
        return {h: float(v) for h, v in zip(header, last)}
    except ValueError:
        return None


def _submit(args, out: Path, natoms: int) -> dict:
    """One Slurm job that re-runs this stage on the compute node with the same inputs."""
    from .. import hpc, machines
    from ..io import atomic

    profile = machines.load_profile(args.machine)
    nodes = max(1, int(getattr(args, "nodes", 1) or 1))
    per_node = getattr(args, "ntasks_per_node", None)
    if not per_node:
        want = max(1, int(getattr(args, "nprocs", 1) or 1), natoms // ATOMS_PER_RANK)
        per_node = min(profile.default_ntasks_per_node or want, max(1, -(-want // nodes)))
    nprocs = nodes * per_node
    payload = {k: v for k, v in vars(args).items()
               if not k.startswith("_") and k not in ("machine", "json_in", "json_out", "describe", "force", "dry_run",
                                                      "output_dir", "stage", "queue", "walltime_hours", "nodes",
                                                      "ntasks_per_node")}
    for k in ("params", "structure_xyzf", "lammps_bin"):
        if payload.get(k):
            payload[k] = str(Path(payload[k]).resolve())
    payload["nprocs"] = nprocs
    inp = out / "lammps_run_input.json"
    atomic.write_json(inp, payload, indent=1, default=str)
    run_dir = out / "run"
    result_json = run_dir / "lammps_run.json"
    cmd = (f"{sys.executable} -m agentic_chimes.cli lammps-run --json-in {inp} --output-dir {run_dir} --force "
           f"--json-out {result_json} > lammps_run.out 2>&1")
    handle = hpc.submit_job(profile, job_name="lammps-run", commands=[f"cd {out.resolve()}", cmd], work_dir=out, nodes=nodes,
                            ntasks_per_node=per_node, walltime_hours=getattr(args, "walltime_hours", 1.0) or 1.0,
                            queue=getattr(args, "queue", "debug") or "debug", dry_run=bool(getattr(args, "dry_run", False)),
                            expect=[result_json])
    return {"submitted": not handle.dry_run, "dry_run": handle.dry_run, "job_id": handle.job_id, "job_file": str(handle.job_file),
            "natoms_simulated": natoms, "mpi_ranks": nprocs, "nodes": nodes, "results_when_done": str(result_json),
            "check_with": f"chimes-agent job-status --work-dir {out}"}


def run(args) -> dict:
    for req in ("params", "structure_xyzf"):
        if not getattr(args, req, None):
            raise ValueError(f"lammps-run requires --{req.replace('_', '-')} (or {req!r} in --json-in)")

    params_path = Path(args.params).resolve()
    frames = xyzf_io.read_xyzf(args.structure_xyzf)
    frame_index = getattr(args, "frame_index", 0) or 0
    if frame_index >= len(frames):
        raise ValueError(f"frame_index={frame_index} out of range (structure_xyzf has {len(frames)} frames)")
    frame = frames[frame_index]
    original_natoms, ncopies = frame.natoms, 1
    if getattr(args, "replicate", True) is not False:
        frame, ncopies = replicate_for_cutoff(frame, params_path)

    lammps_bin = Path(args.lammps_bin) if getattr(args, "lammps_bin", None) else config.resolve_component("lammps_bin")

    work_dir = Path(getattr(args, "output_dir", None) or ".")
    fs.ensure_dir(work_dir)

    if getattr(args, "machine", None):
        # fail here, not on the node: types, masses and elements are checked before anything is submitted
        from ..io import params as params_io

        params_io.resolve_types(params_path, getattr(args, "elements", None), getattr(args, "masses", None))
        params_io.frame_elements_check(params_path, frame.symbols)
        return _submit(args, work_dir.resolve(), frame.natoms)

    data_path = work_dir / "structure.data"
    from ..io import params as params_io

    elements, masses = params_io.resolve_types(params_path, getattr(args, "elements", None), getattr(args, "masses", None))
    params_io.frame_elements_check(params_path, frame.symbols)
    lammps_data.write_lammps_data(frame, elements, masses, data_path)

    mode = getattr(args, "mode", "single_point") or "single_point"
    in_text = _render_input(
        mode,
        data_path.name,
        str(params_path),
        temperature=getattr(args, "temperature", 300.0),
        nsteps=getattr(args, "nsteps", 1000),
        timestep=getattr(args, "timestep", 1.0),
        md_seed=getattr(args, "md_seed", 12345),
    )
    in_path = work_dir / "in.lammps"
    in_path.write_text(in_text)

    log_path = work_dir / "log.lammps"
    nprocs = getattr(args, "nprocs", 1) or 1
    cmd = [str(lammps_bin), "-in", in_path.name, "-log", log_path.name]
    if nprocs > 1:
        # inside a Slurm allocation the scheduler's launcher places the ranks; mpirun only outside one
        launcher = ["srun", "-n", str(nprocs)] if os.environ.get("SLURM_JOB_ID") else ["mpirun", "-n", str(nprocs)]
        cmd = launcher + cmd

    from ..hpc.local import no_core_dumps, singleton_env

    no_core_dumps()

    proc = subprocess.run(cmd, cwd=str(work_dir), capture_output=True, text=True,
                          env=singleton_env() if nprocs == 1 else None)
    (work_dir / "stdout.log").write_text((proc.stdout or "") + (proc.stderr or ""))

    if proc.returncode != 0:
        raise RuntimeError(
            f"{lammps_bin} exited {proc.returncode}; see {work_dir / 'stdout.log'} and {log_path if log_path.is_file() else '(no log.lammps written)'}"
        )

    result = {"work_dir": str(work_dir), "mode": mode, "in_lammps": str(in_path), "log": str(log_path), "data_file": str(data_path),
              "n_copies": ncopies, "natoms_simulated": frame.natoms}

    dump_path = work_dir / "dump.out"
    if dump_path.is_file():
        result["dump"] = str(dump_path)
        if mode == "single_point":
            forces = _parse_dump_forces(dump_path)
            if frame.non_ortho:  # back from LAMMPS' restricted-triclinic frame
                import numpy as np

                forces = (np.asarray(forces) @ lammps_data.rotation_for(frame)).tolist()
            result["forces_kcal_mol_ang"] = forces[:original_natoms]

    if log_path.is_file():
        log_text = log_path.read_text()
        pe = _parse_log_pe(log_text)
        if pe is not None:
            result["energy_kcal_mol"] = pe / ncopies
        if mode == "md":
            thermo = _parse_thermo_last(log_text)
            if thermo:
                result["thermo_last"] = thermo
        result["mpi_ranks"] = nprocs

    return result
