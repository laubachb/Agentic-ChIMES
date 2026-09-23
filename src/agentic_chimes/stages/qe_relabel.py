"""Submit Quantum ESPRESSO single-point (SCF) labeling jobs for unlabeled
configurations, and collect finished output into a labeled `.xyzf` via
converters/qe2xyzf.py.

Two modes, both in this one stage (mirrors qe2xyzf's scf-only scope):
  - submit (default): write one `pw.in` per selected frame under
    `--output-dir/frame_NNNN/`, symlink pseudopotentials in, and submit ONE
    combined Slurm job that runs `pw.x` on each frame directory in sequence
    (not one job per frame -- simpler and more robust to get right first;
    see docs/commands/qe-relabel.md for the tradeoff and how to extend to
    per-frame parallel submission later). Writes a manifest recording which
    original frame index maps to which frame_NNNN dir, read back by collect.
  - collect (--collect DIR): read that manifest, parse every completed
    frame_NNNN/pw.out, merge energy/forces onto the original structure, and
    write one packed `labeled.xyzf` plus a convergence report. A frame
    whose pw.out is missing or unparseable is reported, not silently
    dropped.

`chimes-agent setup --component quantum_espresso` builds `pw.x` itself;
this stage is what actually runs and parses QM jobs with it.
"""

from __future__ import annotations

import json
from pathlib import Path

from .. import config, hpc, machines
from ..converters import qe2xyzf
from ..io import xyzf as xyzf_io

NAME = "qe-relabel"
SUMMARY = "Submit QE single-point (SCF) labeling jobs; --collect converts finished output to .xyzf."
SCHEMA = {
    "type": "object",
    "properties": {
        "structure_xyzf": {"type": "string", "description": "Unlabeled configs (positions/box; forces/energy fields ignored)."},
        "frame_indices": {"type": ["array", "null"], "items": {"type": "integer"}, "description": "Which frames to submit; default all."},
        "elements": {"type": "array", "items": {"type": "string"}},
        "masses": {"type": "object", "additionalProperties": {"type": "number"}},
        "pseudopotentials": {"type": "object", "additionalProperties": {"type": "string"}, "description": "element -> .upf file path"},
        "ecutwfc": {"type": "number", "description": "Ry"},
        "ecutrho": {"type": ["number", "null"], "description": "Ry; default 4*ecutwfc"},
        "kpoints": {"type": "array", "items": {"type": "integer"}, "default": [1, 1, 1]},
        "smearing": {"type": "string", "default": "gaussian"},
        "degauss": {"type": "number", "default": 0.01},
        "conv_thr": {"type": "number", "default": 1.0e-8},
        "machine": {"type": ["string", "null"]},
        "queue": {"type": "string", "default": "batch"},
        "walltime_hours": {"type": "number", "default": 4.0},
        "nodes": {"type": "integer", "default": 1},
        "ntasks_per_node": {"type": ["integer", "null"]},
        "collect": {"type": ["string", "null"], "description": "Path to a previously-submitted work dir; switches to collect mode."},
    },
}


def add_arguments(parser) -> None:
    parser.add_argument("--structure-xyzf", dest="structure_xyzf", default=None)
    parser.add_argument("--frame-indices", dest="frame_indices", type=lambda s: [int(x) for x in s.split(",")], default=None)
    parser.add_argument("--elements", type=lambda s: s.split(","), default=None)
    parser.add_argument("--masses", type=json.loads, default=None)
    parser.add_argument("--pseudopotentials", type=json.loads, default=None)
    parser.add_argument("--ecutwfc", type=float, default=None)
    parser.add_argument("--ecutrho", type=float, default=None)
    parser.add_argument("--kpoints", type=lambda s: [int(x) for x in s.split(",")], default=[1, 1, 1])
    parser.add_argument("--smearing", default="gaussian")
    parser.add_argument("--degauss", type=float, default=0.01)
    parser.add_argument("--conv-thr", dest="conv_thr", type=float, default=1.0e-8)
    parser.add_argument("--machine", default=None)
    parser.add_argument("--queue", default="batch")
    parser.add_argument("--walltime-hours", dest="walltime_hours", type=float, default=4.0)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--ntasks-per-node", dest="ntasks_per_node", type=int, default=None)
    parser.add_argument("--collect", default=None)


_MANIFEST_NAME = "qe_relabel_manifest.json"


def _render_pw_in(frame, elements, masses, pseudopotentials, *, ecutwfc, ecutrho, kpoints, smearing, degauss, conv_thr) -> str:
    if frame.non_ortho:
        raise ValueError("qe-relabel only supports orthorhombic cells for now")
    lx, ly, lz = frame.box

    lines = [
        "&CONTROL",
        "  calculation = 'scf'",
        "  restart_mode = 'from_scratch'",
        "  outdir = './out'",
        "  pseudo_dir = '.'",
        "  prefix = 'qe_relabel'",
        "  tprnfor = .true.",
        "  tstress = .true.",
        "/",
        "&SYSTEM",
        "  ibrav = 0",
        f"  nat = {frame.natoms}",
        f"  ntyp = {len(elements)}",
        f"  ecutwfc = {ecutwfc}",
        f"  ecutrho = {ecutrho if ecutrho else 4 * ecutwfc}",
        "  occupations = 'smearing'",
        f"  smearing = '{smearing}'",
        f"  degauss = {degauss}",
        "/",
        "&ELECTRONS",
        f"  conv_thr = {conv_thr}",
        "/",
        "ATOMIC_SPECIES",
    ]
    for el in elements:
        upf = Path(pseudopotentials[el]).name
        lines.append(f"  {el}  {masses[el]}  {upf}")

    lines.append("ATOMIC_POSITIONS angstrom")
    for sym, pos in zip(frame.symbols, frame.positions):
        lines.append(f"  {sym}  {pos[0]}  {pos[1]}  {pos[2]}")

    lines += [
        "CELL_PARAMETERS angstrom",
        f"  {lx} 0.0 0.0",
        f"  0.0 {ly} 0.0",
        f"  0.0 0.0 {lz}",
        "K_POINTS automatic",
        f"  {kpoints[0]} {kpoints[1]} {kpoints[2]} 0 0 0",
    ]
    return "\n".join(lines) + "\n"


def _submit(args) -> dict:
    for req in ("structure_xyzf", "elements", "masses", "pseudopotentials", "ecutwfc"):
        if not getattr(args, req, None):
            raise ValueError(f"qe-relabel requires --{req.replace('_', '-')} (or {req!r} in --json-in)")
    if not getattr(args, "machine", None):
        raise ValueError("qe-relabel requires --machine")

    frames = xyzf_io.read_xyzf(args.structure_xyzf)
    frame_indices = getattr(args, "frame_indices", None) or list(range(len(frames)))

    work_dir = Path(getattr(args, "output_dir", None) or ".")
    work_dir.mkdir(parents=True, exist_ok=True)

    commands = []
    frame_dirs = []
    for pos, frame_idx in enumerate(frame_indices):
        frame_dir = work_dir / f"frame_{pos:04d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        frame_dirs.append(str(frame_dir))

        pw_in_text = _render_pw_in(
            frames[frame_idx],
            args.elements,
            args.masses,
            args.pseudopotentials,
            ecutwfc=args.ecutwfc,
            ecutrho=getattr(args, "ecutrho", None),
            kpoints=getattr(args, "kpoints", None) or [1, 1, 1],
            smearing=getattr(args, "smearing", "gaussian") or "gaussian",
            degauss=getattr(args, "degauss", 0.01),
            conv_thr=getattr(args, "conv_thr", 1.0e-8),
        )
        (frame_dir / "pw.in").write_text(pw_in_text)

        for el in args.elements:
            src = Path(args.pseudopotentials[el]).resolve()
            dst = frame_dir / src.name
            if not dst.exists():
                dst.symlink_to(src)

        commands.append(f"cd {frame_dir.resolve()}")
        commands.append("pw.x -in pw.in > pw.out 2> pw.err")
        commands.append("cd -")

    manifest = {
        "structure_xyzf": str(Path(args.structure_xyzf).resolve()),
        "frame_indices": frame_indices,
        "frame_dirs": frame_dirs,
    }
    (work_dir / _MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))

    profile = machines.load_profile(args.machine)
    dry_run = bool(getattr(args, "dry_run", False))
    # a dry-run preview must not require the binary to actually be built
    # yet -- only resolve (and require) it for a real submission
    pw_bin = config.resolve_component("qe_pw_bin", required=not dry_run)
    pw_bin_dir = Path(pw_bin).parent if pw_bin else "<qe_pw_bin not yet built; run chimes-agent setup --component quantum_espresso>"
    # profile.modules is already loaded by hpc.submit_job/dry_run's own
    # sbatch template -- only add PATH here, don't duplicate the module load.
    launch_commands = [f"export PATH={pw_bin_dir}:$PATH"] + commands

    handle = hpc.submit_job(
        profile,
        job_name="qe-relabel",
        commands=launch_commands,
        work_dir=work_dir,
        nodes=getattr(args, "nodes", 1) or 1,
        ntasks_per_node=getattr(args, "ntasks_per_node", None),
        walltime_hours=getattr(args, "walltime_hours", 4.0) or 4.0,
        queue=getattr(args, "queue", "batch") or "batch",
        dry_run=dry_run,
    )

    return {
        "work_dir": str(work_dir),
        "n_frames": len(frame_indices),
        "frame_dirs": frame_dirs,
        "manifest": str(work_dir / _MANIFEST_NAME),
        "job_id": handle.job_id,
        "dry_run": handle.dry_run,
        "job_file": str(handle.job_file),
    }


def _collect(args) -> dict:
    work_dir = Path(args.collect)
    manifest_path = work_dir / _MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{manifest_path} not found -- is {work_dir} a qe-relabel work dir from a prior submit?")
    manifest = json.loads(manifest_path.read_text())

    original_frames = xyzf_io.read_xyzf(manifest["structure_xyzf"])

    labeled = []
    report = []
    for orig_idx, frame_dir in zip(manifest["frame_indices"], manifest["frame_dirs"]):
        pw_out = Path(frame_dir) / "pw.out"
        entry = {"frame_index": orig_idx, "frame_dir": frame_dir}
        if not pw_out.is_file():
            entry.update({"status": "missing_output"})
            report.append(entry)
            continue
        try:
            labeled_frame, converged = qe2xyzf.frame_from_qe_output(original_frames[orig_idx], pw_out.read_text())
        except ValueError as exc:
            entry.update({"status": "parse_failed", "error": str(exc)})
            report.append(entry)
            continue

        entry.update({"status": "converged" if converged else "not_converged"})
        report.append(entry)
        if converged:
            labeled.append(labeled_frame)

    out_path = work_dir / "labeled.xyzf"
    xyzf_io.write_xyzf(labeled, out_path)

    n_converged = sum(1 for e in report if e["status"] == "converged")
    return {
        "labeled_xyzf": str(out_path),
        "n_total": len(report),
        "n_converged": n_converged,
        "n_missing_or_failed": len(report) - n_converged,
        "report": report,
    }


def run(args) -> dict:
    if getattr(args, "collect", None):
        return _collect(args)
    return _submit(args)
