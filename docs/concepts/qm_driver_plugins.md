# QM-driver plugins

## Status: planned (Phase 5), not yet built

This page documents the *design* for the QM-driver registry and the QE
driver that will use it — `src/agentic_chimes/qm_drivers/` does not exist
yet. `chimes-agent qe-relabel` is currently a stub (see
`stages/qe_relabel.py`) that echoes its parsed input; `chimes-agent setup
--component quantum_espresso` (already implemented) only fetches and builds
`pw.x` — it does not yet run or parse QM jobs. This page exists now so the
design is written down before implementation, and so `qe-relabel`'s stub
schema already matches the shape the real driver will need.

## Why a registry, not al_driver's if/elif chain

al_driver's `qm_driver.py` dispatches to one of `vasp_driver.py`,
`cp2k_driver.py`, `dftbplus_driver.py`, `gauss_driver.py` via a hardcoded
`if bulk_qm_method == "VASP": ... elif == "CP2K": ...` chain, repeated
across five call sites (`cleanup_and_setup`, `setup_qm`, `continue_job`,
`check_convergence`, `post_process`). Adding a sixth QM code today means
editing five places in a vendored file we don't want to modify. Instead:

```python
class QMDriver(Protocol):
    name: str
    def cleanup_and_setup(self, targets, my_alc, *, build_dir=".") -> None: ...
    def setup(self, my_alc, targets, *, build_dir=".") -> JobHandle | list[JobHandle]: ...
    def continue_job(self, targets, *, build_dir=".") -> JobHandle | list[JobHandle]: ...
    def check_convergence(self, my_alc, targets, *, build_dir=".") -> ConvergenceReport: ...
    def post_process(self, targets, *, build_dir=".") -> None: ...
```

`qm_drivers/__init__.py` will hold a `QM_DRIVER_REGISTRY: dict[str,
QMDriver]` and a `register_driver(name, driver)` function. `vasp.py`,
`cp2k.py`, `dftbplus.py`, `gaussian.py` will be thin adapters that import
the corresponding module from `codes/al_driver-LLfork/src/` and expose it
through this protocol — the underlying input-generation/output-parsing
logic is reused as-is, only the calling convention changes from
`*argv, **kwargs` to named, typed parameters.

## The QE driver

`qe.py` will be the first genuinely new driver (no upstream QE support
exists anywhere in `codes/` — confirmed by a full-repo grep during design).
Planned behavior, matching the five-function contract:

- **`setup`**: write `pw.in` from a template (elements, k-points,
  ecutwfc/ecutrho, pseudopotential paths, smearing — all from stage input,
  not hardcoded), submit via `hpc/slurm.py` using the active
  `MachineProfile`'s modules.
- **`continue_job`**: QE supports `restart_mode = 'restart'` reading its
  own `.save` directory; the driver checks for that directory and sets the
  flag, mirroring how `vasp_driver.continue_job` swaps `CONTCAR` → `POSCAR`.
- **`check_convergence`**: parse `pw.x` stdout for `convergence has been
  achieved` (SCF) and `Final energy` / ionic-convergence lines for a
  relax/vc-relax run.
- **`post_process`**: call `converters/qe2xyzf.py` to emit `.xyzf` frames.

## The `.xyzf` output contract

`converters/qe2xyzf.py` (also not yet built) must match the same output
contract as `codes/chimes_lsq-LLfork/contrib/vasp2xyzf.py` — same per-frame
`.xyzf` block layout (see `io/xyzf.py`'s docstring) and the same unit
conventions (`converters/units.py`): energies ×23.0605 eV→kcal/mol, forces
×(1/51.4221) eV/Å→hartree/bohr where that convention is required. Unlike
`vasp2xyzf.py` (Python 2, parses VASP `OUTCAR` text), `qe2xyzf.py` will be
a full Python 3 implementation parsing QE's `pw.x` stdout (`!    total
energy`, `ATOMIC_POSITIONS`, `Forces acting on atoms`, `total   stress`
blocks) or its `data-file-schema.xml`.

## Plugging in a new QM code yourself

Once the registry lands, adding another code (e.g. a different DFT
package) means writing one module implementing the five-function
`QMDriver` protocol and calling `register_driver("MYCODE", MyDriver())` —
no changes to any other file, and no changes to `codes/al_driver-LLfork`.
