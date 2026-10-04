# `chimes-agent setup`

**Status: implemented.**

Builds/fetches the binaries and libraries every other stage depends on:
first clones the three vendored forks into `codes/` (gitignored, never
pushed to this repo — see
[docs/concepts/vendored_forks.md](../concepts/vendored_forks.md)), then
builds `chimes_lsq` + `dlars` (from `codes/chimes_lsq-LLfork`), the
serial/ctypes evaluator (`codes/chimes_calculator-LLfork`), the
ChIMES-patched LAMMPS build, and Quantum ESPRESSO's `pw.x` (fetched fresh —
not vendored anywhere). See `src/agentic_chimes/setup/`.

## Usage

```bash
chimes-agent setup --machine dane --component all
chimes-agent setup --machine dane --component chimes_calculator
chimes-agent setup --machine dane --component quantum_espresso --qe-version qe-7.3.1
chimes-agent setup --component codes   # just clone codes/, no build, no --machine needed
chimes-agent setup --status
chimes-agent setup --init-profile ./my_cluster.yaml --scratch /scratch/$USER --modules gcc/12,openmpi/4   # write a machine profile
chimes-agent setup --component codes --ref chimes_lsq=<commit>   # one fork at another commit, this once
```

## Flags

- `--component {codes,chimes_lsq,chimes_calculator,lammps,quantum_espresso,all}`
  (repeatable; default `all`) — `codes` clones the vendored forks without
  building anything; every other component (and `all`) clones them
  automatically as a first step regardless, so you rarely need `codes`
  explicitly except to pre-fetch source without a machine profile
- `--machine {dane,stampede3,<path-to-profile.yaml>}` (required unless
  `--status`, or `--component codes` alone)
- `--qe-version TAG` — override the pinned QE release tag
  (`build_quantum_espresso.DEFAULT_QE_VERSION`; check
  https://gitlab.com/QEF/q-e/-/tags before relying on the built-in default)
- `--status` — report `deps/installed.json` contents and exit, no build
- `--ref REPO=COMMIT` (repeatable) — check one vendored fork out at a
  commit, branch or tag other than its pin, for this call (`REPO` is
  `al_driver`, `chimes_lsq` or `chimes_calculator`; the `-LLfork` suffix is
  optional). The fork is fetched and checked out even if already cloned; the
  result names the pin it replaced and reminds you to rebuild what depends
  on it. The pin in `setup/clone_codes.py` is unchanged, so the next plain
  `setup --component codes --force` returns to it. See
  [vendored forks](../concepts/vendored_forks.md).
- `--init-profile PATH` — write a machine profile for this cluster and
  exit; nothing is built. Partitions, cores per node and your Slurm accounts
  are read from `sinfo` and `sacctmgr`; give `--scratch DIR` (shared job
  directory), `--modules a,b`, and optionally `--account`,
  `--debug-partition`, `--batch-partition`, `--cores-per-node`,
  `--hosttype`. Whatever could not be determined is listed in `todo` and as
  `# TODO` lines in the file. The account is written as
  `${CHIMES_ACCOUNT:-<yours>}` so the file can be shared. On Dane this
  reproduces the built-in profile from the scheduler alone. `--force`
  overwrites an existing file. See
  [machine profiles](../concepts/machine_profiles.md).
- `--force` (generic flag) — rebuild even if already recorded for this
  machine

## What each component does

| Component | Runs | Produces |
|---|---|---|
| `codes` | `git clone` (pinned commit) for each of the three vendored forks into `codes/` | `codes/al_driver-LLfork`, `codes/chimes_lsq-LLfork`, `codes/chimes_calculator-LLfork` |
| `chimes_lsq` | `codes/chimes_lsq-LLfork/install.sh` (fed `n` at its interactive prompt — always safe/idempotent) | `chimes_lsq_bin`, `dlars_bin` |
| `chimes_calculator` | `codes/chimes_calculator-LLfork/install.sh` (non-interactive) | `chimescalc_lib` (`libchimescalc_dl.so`) |
| `lammps` | `codes/chimes_calculator-LLfork/etc/lmp/install.sh` (shallow-clones a pinned LAMMPS tag, patches in the ChIMES pair style + a modified `pair.{h,cpp}`, builds `lmp_mpi_chimes`) | `lammps_bin` (also symlinked at `deps/lammps-chimes/`) |
| `quantum_espresso` | `git clone` a pinned QE release tag into `deps/quantum-espresso`, then `./configure && make pw` | `qe_pw_bin` (`pw.x`) |

Every component is machine-profile aware: it sources
`codes/*/modfiles/<hosttype>.mod` (or the QE build's own module list) via
the profile's `hosttype`/`modules` before building.

## Idempotency

`codes` cloning reports `"status": "already_present"` on any call after
the first, unless `--force` (which does `git fetch` + `git checkout <ref>`
again — picks up a moved pin, does not force-discard local modifications).
Re-running `setup` for a build component already recorded in
`deps/installed.json` **for the same machine** skips the build and reports
`"status": "skipped_already_installed"`. `chimes_lsq`'s own `install.sh`
does a full rebuild every time it runs (`rm -rf build` unconditionally), so
this skip is the only incremental-build story for that component — there's
no partial-rebuild mode to preserve.

## Output

Every call's `results` includes a `"codes"` entry (the clone step always
runs first), plus one entry per requested build component:

```json
{
  "machine": "dane",
  "results": {
    "codes": {
      "al_driver-LLfork": {"path": "/path/to/codes/al_driver-LLfork", "ref": "92db0aa...", "status": "already_present"},
      "chimes_lsq-LLfork": {"path": "/path/to/codes/chimes_lsq-LLfork", "ref": "75b51f8...", "status": "already_present"},
      "chimes_calculator-LLfork": {"path": "/path/to/codes/chimes_calculator-LLfork", "ref": "f4ddddd...", "status": "already_present"}
    },
    "chimes_calculator": {
      "status": "built",
      "machine": "dane",
      "chimescalc_lib": "/path/to/codes/chimes_calculator-LLfork/build/libchimescalc_dl.so"
    }
  }
}
```

(`chimes-agent setup --component codes` alone returns just the `"codes"`
key — no `"machine"` needed, no build components run.)

`"status"` per component is one of `built`, `skipped_already_installed`, or
`failed` (with an `"error"` message; a failed component doesn't stop other
requested components from being attempted).
