"""Single-point/MD via the ChIMES-patched `lmp_mpi_chimes` build (`chimes-agent
setup --component lammps`). Local execution only in this phase (matches
amat-build/solve's current scope) -- HPC submission for long MD runs is
future work.

Single-point mode (`run 0`) is the documented independent cross-check
against the ctypes evaluator (stages/evaluate.py): both read the same
params.txt and should agree on forces/energy for a given configuration.
Forces are parsed from a LAMMPS dump file (a stable, simple format);
potential energy is parsed from the thermo log we control the format of
(`thermo_style custom step pe`) -- a best-effort parse, not fatal if it
fails, since forces from the dump are the primary reliable output.

Correctness-critical: `elements` (LAMMPS atom-type order) and `masses`
must match what the params.txt being tested was fit with -- see
io/lammps_data.py's docstring.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Optional

from .. import config
from ..io import lammps_data, xyzf as xyzf_io

NAME = "lammps-run"
SUMMARY = "Single-point/MD via lmp_mpi_chimes (local execution)."
SCHEMA = {
    "type": "object",
    "required": ["params", "structure_xyzf", "elements", "masses"],
    "properties": {
        "params": {"type": "string"},
        "structure_xyzf": {"type": "string", "description": "A .xyzf file; frame_index selects which frame (default 0)."},
        "frame_index": {"type": "integer", "default": 0},
        "elements": {"type": "array", "items": {"type": "string"}, "description": "LAMMPS atom-type order; must match the params.txt's own element order."},
        "masses": {"type": "object", "additionalProperties": {"type": "number"}},
        "mode": {"type": "string", "enum": ["single_point", "md"], "default": "single_point"},
        "temperature": {"type": "number", "default": 300.0},
        "nsteps": {"type": "integer", "default": 1000},
        "timestep": {"type": "number", "default": 1.0, "description": "fs (units real)"},
        "md_seed": {"type": "integer", "default": 12345},
        "lammps_bin": {"type": ["string", "null"]},
        "nprocs": {"type": "integer", "default": 1},
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


def run(args) -> dict:
    for req in ("params", "structure_xyzf", "elements", "masses"):
        if not getattr(args, req, None):
            raise ValueError(f"lammps-run requires --{req.replace('_', '-')} (or {req!r} in --json-in)")

    params_path = Path(args.params).resolve()
    frames = xyzf_io.read_xyzf(args.structure_xyzf)
    frame_index = getattr(args, "frame_index", 0) or 0
    if frame_index >= len(frames):
        raise ValueError(f"frame_index={frame_index} out of range (structure_xyzf has {len(frames)} frames)")
    frame = frames[frame_index]

    lammps_bin = Path(args.lammps_bin) if getattr(args, "lammps_bin", None) else config.resolve_component("lammps_bin")

    work_dir = Path(getattr(args, "output_dir", None) or ".")
    work_dir.mkdir(parents=True, exist_ok=True)

    data_path = work_dir / "structure.data"
    lammps_data.write_lammps_data(frame, args.elements, args.masses, data_path)

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
        cmd = ["mpirun", "-n", str(nprocs)] + cmd

    proc = subprocess.run(cmd, cwd=str(work_dir), capture_output=True, text=True)
    (work_dir / "stdout.log").write_text((proc.stdout or "") + (proc.stderr or ""))

    if proc.returncode != 0:
        raise RuntimeError(
            f"{lammps_bin} exited {proc.returncode}; see {work_dir / 'stdout.log'} and {log_path if log_path.is_file() else '(no log.lammps written)'}"
        )

    result = {"work_dir": str(work_dir), "mode": mode, "in_lammps": str(in_path), "log": str(log_path), "data_file": str(data_path)}

    dump_path = work_dir / "dump.out"
    if dump_path.is_file():
        result["dump"] = str(dump_path)
        if mode == "single_point":
            result["forces_kcal_mol_ang"] = _parse_dump_forces(dump_path)

    if log_path.is_file():
        pe = _parse_log_pe(log_path.read_text())
        if pe is not None:
            result["energy_kcal_mol"] = pe

    return result
