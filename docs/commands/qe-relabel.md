# `chimes-agent qe-relabel`

**Status: implemented (single-point SCF only).**

Submits Quantum ESPRESSO single-point (`calculation = 'scf'`) labeling
jobs for unlabeled configurations, and (`--collect`) converts finished
output into a labeled `.xyzf`. There was no QE support anywhere in the
vendored forks before this — see `src/agentic_chimes/converters/qe2xyzf.py`
and `src/agentic_chimes/stages/qe_relabel.py`.

`chimes-agent setup --component quantum_espresso` (separate command)
fetches and builds `pw.x` itself; this stage is what actually runs and
parses QM jobs with it.

## Scope: single-point SCF, not relax

Positions don't change in an SCF run, so `qe2xyzf` doesn't re-parse QE's
echoed geometry at all (QE has several different position-echo unit
conventions depending on input options — alat, crystal, angstrom — parsing
them back out is unnecessary complexity for this use case). It merges the
*original* structure (the one `pw.in` was generated from) with QE's parsed
energy/forces. A relax/vc-relax workflow would need different handling and
isn't implemented.

## Usage

```bash
# 1. submit (writes pw.in per frame, symlinks pseudopotentials, submits ONE
#    combined job that runs pw.x on each frame directory in sequence)
chimes-agent qe-relabel \
  --structure-xyzf unlabeled.xyzf \
  --elements C,H --masses '{"C":12.011,"H":1.008}' \
  --pseudopotentials '{"C":"/path/to/C.upf","H":"/path/to/H.upf"}' \
  --ecutwfc 60 --kpoints 2,2,2 \
  --machine dane --queue batch --walltime-hours 4 \
  --output-dir ./qe_run

# preview the job script + pw.in files first, no allocation spent:
chimes-agent qe-relabel ... --output-dir ./qe_run --dry-run

# 2. once the Slurm job finishes, collect + convert
chimes-agent qe-relabel --collect ./qe_run --json-out ./qe_run/collect.json
```

## Why one combined job, not one job per frame

Simpler and more robust to get right first — see
`docs/concepts/stages_and_contracts.md`'s phasing philosophy. The
trade-off is throughput: frames run sequentially within one allocation,
not in parallel across many small jobs (the pattern al_driver's own
per-config QM submission uses). Submitting one job per frame is a natural
future extension (would reuse the same `pw.in` generation and
`converters/qe2xyzf.py` collection logic, changing only the submission
loop in `stages/qe_relabel.py:_submit`) if throughput at scale becomes the
bottleneck.

## Flags (submit mode)

- `--structure-xyzf PATH` (required) — unlabeled configs; `forces`/`energy`
  fields are ignored (ChIMES `.xyzf` requires *some* forces field, so pass
  zeros — see `io/xyzf.py`)
- `--frame-indices 0,1,2` — which frames to submit; default all
- `--elements`, `--masses` (required) — same convention as `fm-setup-gen`
- `--pseudopotentials '{"C":"/path/C.upf",...}'` (required)
- `--ecutwfc RY` (required), `--ecutrho RY` (default `4*ecutwfc`)
- `--kpoints nx,ny,nz` (default `1,1,1` — a Γ-only/small grid; override for
  small unit cells)
- `--smearing`, `--degauss`, `--conv-thr`
- `--machine`, `--queue`, `--walltime-hours`, `--nodes`, `--ntasks-per-node`
- `--dry-run` (generic flag) — render the job + `pw.in` files, don't submit

## Flags (collect mode)

- `--collect DIR` — a work dir from a prior submit (reads its
  `qe_relabel_manifest.json` to know which frame maps to which
  `frame_NNNN/` subdirectory)

## Output (submit)

```json
{
  "work_dir": "./qe_run", "n_frames": 20,
  "frame_dirs": ["./qe_run/frame_0000", "..."],
  "manifest": "./qe_run/qe_relabel_manifest.json",
  "job_id": "1234567", "dry_run": false, "job_file": "./qe_run/run.cmd"
}
```

## Output (collect)

```json
{
  "labeled_xyzf": "./qe_run/labeled.xyzf",
  "n_total": 20, "n_converged": 18, "n_missing_or_failed": 2,
  "report": [
    {"frame_index": 0, "frame_dir": "./qe_run/frame_0000", "status": "converged"},
    {"frame_index": 1, "frame_dir": "./qe_run/frame_0001", "status": "missing_output"}
  ]
}
```

Only `"converged"` frames are written into `labeled_xyzf`; `"status"` is
one of `converged`, `not_converged` (pw.x ran but didn't reach SCF
convergence), `missing_output` (job hasn't finished / never ran), or
`parse_failed` (with an `"error"` message).

## Units

QE reports energy in Rydberg and forces in Ry/bohr; `qe2xyzf` converts to
the same target convention `contrib/vasp2xyzf.py` writes (energy kcal/mol,
forces hartree/bohr) — see `docs/concepts/units_and_conventions.md`.
