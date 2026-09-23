# QM relabeling: QE now, a registry if/when more codes join

## What actually shipped: `qe-relabel`, implemented directly

`chimes-agent qe-relabel` (`src/agentic_chimes/stages/qe_relabel.py` +
`src/agentic_chimes/converters/qe2xyzf.py`) is implemented — submit +
`--collect` both work, validated against a hand-crafted realistic `pw.x`
output fixture with independently hand-computed unit conversions (see
`tests/unit/test_qe2xyzf.py`, `tests/unit/test_qe_relabel.py`). `chimes-agent
setup --component quantum_espresso` fetches and builds `pw.x` itself
(separate command); `qe-relabel` is what actually runs and parses QM jobs
with it.

It was built as a **direct implementation**, not through an abstract
`QMDriver` plugin registry — QE was the only new QM code being added, and
building a registry to hold a single member added indirection without
paying for itself yet. See `docs/commands/qe-relabel.md` for the full
scope (single-point SCF only; one combined Slurm job across frames, not
one job per frame) and rationale for both choices.

## If/when a second QM code is added: the registry design

This is the design for *if* another QM code joins QE later — worth writing
down now, not yet built. al_driver's own `qm_driver.py` dispatches to
`vasp_driver.py`/`cp2k_driver.py`/`dftbplus_driver.py`/`gauss_driver.py`
via a hardcoded `if bulk_qm_method == "VASP": ... elif == "CP2K": ...`
chain repeated across five call sites — fine for al_driver's own vendored
code, but adding a driver here without touching that file needs a
different shape:

```python
class QMDriver(Protocol):
    name: str
    def cleanup_and_setup(self, targets, my_alc, *, build_dir=".") -> None: ...
    def setup(self, my_alc, targets, *, build_dir=".") -> JobHandle | list[JobHandle]: ...
    def continue_job(self, targets, *, build_dir=".") -> JobHandle | list[JobHandle]: ...
    def check_convergence(self, my_alc, targets, *, build_dir=".") -> ConvergenceReport: ...
    def post_process(self, targets, *, build_dir=".") -> None: ...
```

A `qm_drivers/__init__.py` registry (`QM_DRIVER_REGISTRY: dict[str,
QMDriver]` + `register_driver(name, driver)`) would let a `qe.py`
(wrapping the logic that's currently direct in `stages/qe_relabel.py`) sit
alongside thin adapters over al_driver's existing VASP/CP2K/DFTB+/Gaussian
drivers — imported from `codes/al_driver-LLfork/src/` and exposed through
this protocol, reusing their input-generation/output-parsing logic as-is
and only changing the calling convention from `*argv, **kwargs` to named,
typed parameters. `src/agentic_chimes/qm_drivers/__init__.py` already
exists as a placeholder noting this.

Plugging in a new QM code at that point would mean writing one module
implementing the five-function protocol and calling
`register_driver("MYCODE", MyDriver())` — no changes to any other file,
and no changes to `codes/al_driver-LLfork`.

## Uncertainty-based selection: still a future plug-in point, not wired up

Unrelated to the registry question: al_driver's actual implemented AL
selection is diversity-based (`gen_selections.py`'s Metropolis-MC
energy-histogram flattening, used by `al-run` when you launch the full
driver), not uncertainty/query-by-committee —
`doc/source/files_to_finish/committeeALmode.rst` is an empty stub
confirming that mode was never finished upstream, and this repo doesn't
attempt to finish it either. The concrete plug-in point for
uncertainty-based selection remains `chimes-agent evaluate`'s
`committee_spread` output (multiple `params.txt` models diffed on the same
unlabeled candidate frames) — see `docs/commands/evaluate.md`.
