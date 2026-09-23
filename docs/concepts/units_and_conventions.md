# Units and conventions

Centralized in `src/agentic_chimes/converters/units.py` so a wrong factor
is a one-line fix, not a hunt through every stage.

## Unit conventions

| Quantity | ChIMES params.txt / A-matrix convention | Notes |
|---|---|---|
| Energy | kcal/mol | `EV_TO_KCAL_PER_MOL = 23.0605` |
| Force | kcal/mol/Å (fit convention) | some upstream tooling (`vasp2xyzf.py`) instead targets hartree/bohr — `qe2xyzf.py` must be explicit about which one it writes |
| Force (hartree/bohr) | — | `HARTREE_PER_BOHR_TO_EV_PER_ANG = 51.4221` (product of Hartree→eV and Bohr→Å) |
| Stress (ctypes return) | ChIMES-internal | `CHIMES_STRESS_TO_GPA = 6.9479` to convert; note the `chimescalc` standalone binary's own printed "Stress tensors (GPa)" output is already converted, so no further scaling is needed there |

DFT codes (VASP, QE) natively report eV and eV/Å; always convert
explicitly at the point where a `.xyzf` file is written, and say which
convention (kcal/mol/Å vs hartree/bohr) that file uses in the stage that
produced it.

## Guardrails baked in as defaults, not opt-in flags

These come directly from hard-won lessons in prior hand-rolled ChIMES
studies with this same toolchain, and are enforced by default rather than
left for a caller to remember:

- **Dane `--ntasks-per-node`** — a bare `sbatch -N 1` on Dane gives 1 CPU +
  ~2.3GB, not a full node. `hpc/slurm.py` and `hpc/dry_run.py` always set
  `ntasks_per_node` from the machine profile's `default_ntasks_per_node`
  unless a caller overrides it, and refuse to render/submit without one.
  See `docs/concepts/machine_profiles.md`.
- **lustre2 file-count quota** — `/p/lustre2` is limited by *file count*
  (~1.05M soft), not space; writing one file per frame/config at scale
  exhausts it long before disk space runs out. Any stage that could
  plausibly write per-frame output at scale (dataset selection, QE
  relabeling output, AL candidate sets) is designed to default to packed
  `.npz`/JSON-array output (`io/packed_frames.py`, planned) rather than
  one-file-per-frame, and to support archiving genuinely bulk raw output
  to `/p/lustre3` (space-rich, not file-count-limited) via a
  `bulk_archive_root` on the machine profile.
- **DLARS cliff finalize margin** — `stages/_cliff_monitor.py` subtracts a
  small margin (default 3 iterations) from the iteration at first
  detected Cholesky failure before finalizing, reproducing the
  hand-validated "re-run DLARS fresh with `--iterations=<N-a-few>`" fix
  rather than finalizing from a restart file that's already past the
  cliff.

## Where a DFT/QM comparison needs a unit fix and where it doesn't

A DFT-vs-ChIMES comparison (e.g. holdout force RMSE against raw QM output)
needs the appropriate `converters.units` conversion applied to the QM side
first. A ChIMES-vs-ChIMES comparison (e.g. `evaluate --params modelA
--params modelB` committee mode, or comparing two fits' predictions against
each other) needs no such factor — both sides are already in the same
`params.txt` convention.
