"""Check that this installation can run a study, before a study finds out.

Each check reports ok / warn / fail, what it saw, and the command that fixes
it. Local checks take seconds. They include one real LAMMPS single-point and
one chimes_calculator evaluation against the published CHON reference that
the unit tests use, so "installed" also means "gives the right numbers".
With `--machine`, the profile is checked too: account set, scratch root
shared and writable, sbatch and the partitions present.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .. import config

NAME = "doctor"
SUMMARY = "Check the installation (components, numerics, machine profile, credentials) and say how to fix each problem."
USES_OUTPUT_DIR = False
SCHEMA = {
    "type": "object",
    "properties": {
        "machine": {"type": ["string", "null"], "description": "Also check this machine profile (name or YAML path)."},
        "quick": {"type": "boolean", "default": False, "description": "Skip the numerical LAMMPS/calculator checks."},
    },
}

_CHON_ENERGY = -7.8371473
_CHON_FORCE0 = (-19.8826, 20.5959, 12.0526)
_CHON_MASSES = {"C": 12.011, "H": 1.0079, "O": 15.9994, "N": 14.007}


def add_arguments(parser) -> None:
    parser.add_argument("--machine", default=None)
    parser.add_argument("--quick", action="store_true")


def _check(name, status, detail, fix=None):
    out = {"check": name, "status": status, "detail": detail}
    if fix and status != "ok":
        out["fix"] = fix
    return out


def _component(name, fix, *, optional=False):
    path = config.resolve_component(name, required=False)
    if path is None:
        return _check(name, "warn" if optional else "fail", "not installed", fix)
    if not Path(path).exists():
        return _check(name, "fail", f"recorded at {path}, which does not exist", fix)
    return _check(name, "ok", str(path))


def _chon_frame():
    from ..io import xyzf as xyzf_io

    fx = config.CHIMES_CALCULATOR_ROOT / "serial_interface" / "tests"
    lines = (fx / "configurations" / "CHON.testfile_#000.xyz").read_text().splitlines()
    n = int(lines[0].split()[0])
    box9 = [float(x) for x in lines[1].split()]
    syms, pos = [], []
    for row in lines[2: 2 + n]:
        t = row.split()
        syms.append(t[0])
        pos.append([float(t[1]), float(t[2]), float(t[3])])
    frame = xyzf_io.Frame(symbols=syms, positions=pos, forces=[[0.0] * 3] * n, box=[box9[0], box9[4], box9[8]])
    return frame, fx / "force_fields" / "test_params.CHON.txt"


def _numerics() -> list:
    from types import SimpleNamespace

    from ..io import xyzf as xyzf_io

    checks = []
    try:
        frame, params = _chon_frame()
    except OSError as exc:
        return [_check("reference test data", "fail", str(exc), "chimes-agent setup --component codes")]
    if config.resolve_component("chimescalc_lib", required=False):
        try:
            from . import evaluate

            w = evaluate._load_wrapper()
            ptr = w.chimes_open_instance()
            w.set_chimes_instance(ptr, small=False)
            w.init_chimes_instance(ptr, str(params), 0)
            e, f = evaluate.predict(w, ptr, frame, evaluate.max_outer_cutoff(params))
            ok = abs(e - _CHON_ENERGY) < 1e-3 and all(abs(a - b) < 1e-2 for a, b in zip(f[0], _CHON_FORCE0))
            checks.append(_check("chimes_calculator numerics", "ok" if ok else "fail",
                                 f"CHON reference energy {e:.5f} (expected {_CHON_ENERGY})",
                                 "rebuild: chimes-agent setup --component chimes_calculator --machine <m>"))
        except Exception as exc:  # noqa: BLE001
            checks.append(_check("chimes_calculator numerics", "fail", str(exc)[-300:],
                                 "rebuild: chimes-agent setup --component chimes_calculator --machine <m>"))
    if config.resolve_component("lammps_bin", required=False):
        from . import lammps_run

        with tempfile.TemporaryDirectory() as td:
            try:
                xyzf_io.write_xyzf([frame], Path(td) / "s.xyzf")
                r = lammps_run.run(SimpleNamespace(
                    params=str(params), structure_xyzf=str(Path(td) / "s.xyzf"), frame_index=0,
                    elements=["C", "H", "O", "N"], masses=_CHON_MASSES, mode="single_point", temperature=300.0,
                    nsteps=0, timestep=1.0, md_seed=1, lammps_bin=None, nprocs=1, replicate=True,
                    output_dir=str(Path(td) / "run")))
                e = r.get("energy_kcal_mol")
                ok = e is not None and abs(e - _CHON_ENERGY) < 1e-3
                checks.append(_check("LAMMPS (chimesFF) numerics", "ok" if ok else "fail",
                                     f"CHON reference energy {e} (expected {_CHON_ENERGY})",
                                     "rebuild: chimes-agent setup --component lammps --machine <m>"))
            except Exception as exc:  # noqa: BLE001
                checks.append(_check("LAMMPS (chimesFF) numerics", "fail", str(exc)[-300:],
                                     "rebuild: chimes-agent setup --component lammps --machine <m>; if it is an MPI "
                                     "launch error, check the profile's modules"))
    return checks


def lustre_file_quota(path) -> dict | None:
    """{files, soft, hard, fraction} from `lfs quota` for the filesystem holding
    `path`, or None when it is not Lustre / lfs is unavailable. On LLNL lustre2
    the file COUNT, not space, is the limit users hit."""
    import getpass

    if not shutil.which("lfs"):
        return None
    try:
        p = subprocess.run(["lfs", "quota", "-u", getpass.getuser(), str(path)], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = p.stdout.splitlines()
    try:
        k = next(i for i, ln in enumerate(lines) if ln.strip().startswith("Filesystem"))
    except StopIteration:
        return None
    toks = " ".join(lines[k + 1:]).split()
    if toks and not toks[0].rstrip("*").isdigit():
        toks = toks[1:]  # the path column
    try:
        files, soft, hard = (int(t.rstrip("*")) for t in toks[4:7])
    except (ValueError, IndexError):
        return None
    limit = soft or hard
    return {"files": files, "soft": soft, "hard": hard, "fraction": files / limit if limit else None}


def _machine(name) -> list:
    from .. import machines
    from ..hpc import slurm

    checks = []
    try:
        prof = machines.load_profile(name)
    except Exception as exc:  # noqa: BLE001
        return [_check("machine profile", "fail", str(exc), "pass a built-in name or a valid YAML path")]
    checks.append(_check("machine profile", "ok", f"{prof.name} ({prof.source_path})"))
    checks.append(_check("account", "ok" if prof.account else "fail", prof.account or "empty",
                         "export CHIMES_ACCOUNT=<your bank/allocation>"))
    root = (prof.filesystem or {}).get("scratch_root")
    if not root:
        checks.append(_check("scratch_root", "warn", "not set in the profile", "add filesystem.scratch_root"))
    else:
        p = Path(root)
        try:
            slurm.check_shared_dir(p)
            writable = p.is_dir() and os.access(p, os.W_OK)
            checks.append(_check("scratch_root", "ok" if writable else "fail",
                                 f"{p} ({'writable' if writable else 'missing or not writable'})",
                                 f"create it or fix filesystem.scratch_root ({p})"))
            q = lustre_file_quota(p) if writable else None
            if q and q["fraction"] is not None:
                frac = q["fraction"]
                status = "fail" if frac >= 0.98 else "warn" if frac >= 0.85 else "ok"
                checks.append(_check("file-count quota", status,
                                     f"{q['files']:,} of {q['soft']:,} files ({frac:.0%}) on {p}",
                                     "archive many-file directories as tars on bulk storage (e.g. /p/lustre3) and "
                                     "remove caches; stages write packed files, but studies and envs add up"))
        except ValueError as exc:
            checks.append(_check("scratch_root", "fail", str(exc), "point scratch_root at a shared filesystem"))
    submit = "qsub" if prof.job_system == "torque" else "sbatch"
    have = shutil.which(submit)
    checks.append(_check(submit, "ok" if have else "fail", have or "not on PATH",
                         "run on a login node of this cluster, or load its scheduler module"))
    if have and submit == "sbatch" and shutil.which("sinfo"):
        missing = []
        for logical, part in prof.partitions.items():
            r = subprocess.run(["sinfo", "-h", "-p", part, "-o", "%P"], capture_output=True, text=True)
            if not r.stdout.strip():
                missing.append(f"{logical}={part}")
        checks.append(_check("partitions", "fail" if missing else "ok",
                             f"missing: {missing}" if missing else ", ".join(prof.partitions.values()),
                             "fix `partitions` in the profile"))
    return checks


def run(args) -> dict:
    checks = [_check("python", "ok", f"{sys.version.split()[0]} at {sys.executable}")]
    for comp, fix, opt in (
        ("chimes_lsq_bin", "chimes-agent setup --component chimes_lsq --machine <m>", False),
        ("chimescalc_lib", "chimes-agent setup --component chimes_calculator --machine <m>", False),
        ("lammps_bin", "chimes-agent setup --component lammps --machine <m>", False),
        ("dlars_bin", "chimes-agent setup --component chimes_lsq --machine <m> (needed only for dlars/dlasso)", True),
        ("qe_pw_bin", "chimes-agent setup --component quantum_espresso --machine <m> (needed only for QE labeling)", True),
    ):
        checks.append(_component(comp, fix, optional=opt))
    checks.append(_check("al_driver", "ok" if config.AL_DRIVER_SRC.is_dir() else "fail", str(config.AL_DRIVER_SRC),
                         "chimes-agent setup --component codes"))
    for mod, why in (("ase", "structures, RDFs, md-check"), ("pyarrow", "open-data fetch"), ("sklearn", "local solvers"),
                     ("matplotlib", "plots in evaluate/md-check/eos-check/learning-curve (pip install -e '.[plots]')"),
                     ("quests", "QUESTS entropy/novelty/selection (pip install -e '.[quests]')"),
                     ("scipy", "chi-squared critical values for fingerprint tests")):
        try:
            __import__(mod)
            checks.append(_check(f"python: {mod}", "ok", why))
        except ImportError:
            checks.append(_check(f"python: {mod}", "warn", f"missing ({why})", 'pip install -e ".[data]"'))
    checks.append(_check("HF_TOKEN", "ok" if os.environ.get("HF_TOKEN") else "warn",
                         "set" if os.environ.get("HF_TOKEN") else "unset: open-data downloads may hit HTTP 429 on shared networks",
                         "export HF_TOKEN=<Hugging Face read token>"))
    papers = config.REPO_ROOT / "chimes_papers"
    if papers.is_dir():
        indexed = (papers / "text" / "INDEX.txt").is_file()
        checks.append(_check("literature index", "ok" if indexed else "warn",
                             f"{papers}/text" if indexed else "PDFs present but not indexed", "tools/index_papers.sh"))
    if not getattr(args, "quick", False):
        checks += _numerics()
    if getattr(args, "machine", None):
        checks += _machine(args.machine)
    n_fail = sum(c["status"] == "fail" for c in checks)
    n_warn = sum(c["status"] == "warn" for c in checks)
    return {"ok": n_fail == 0, "n_fail": n_fail, "n_warn": n_warn, "checks": checks}
