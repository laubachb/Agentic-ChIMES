# The vendored forks (`codes/`)

`codes/` holds three of the lab's own GitHub forks — the ground-truth
ChIMES codebases this whole repo builds on top of:

| Fork | Purpose |
|---|---|
| `al_driver-LLfork` | active-learning orchestration (reference for the planned QM-driver contract; `helpers.py`'s Slurm primitives reused by `hpc/slurm.py`) |
| `chimes_lsq-LLfork` | design-matrix generation (`chimes_lsq`) + DLARS/LASSO solver |
| `chimes_calculator-LLfork` | force-field evaluator (ctypes + LAMMPS pair style) |

## `codes/` is gitignored — never pushed to this repo

These are the lab's own forks, each with their own history and GitHub
remote (`LindseyLab-umich/*`). Agentic-ChIMES does not embed, vendor-commit,
or redistribute their contents: `codes/` is listed in `.gitignore`
alongside `deps/`, and a fresh checkout of this repo has no `codes/`
directory at all until `chimes-agent setup` clones it. This is deliberate,
not an oversight — it keeps this repo's own history free of another
project's source and commit history, and means updating a fork here is a
`git` operation against *that* fork's own repo, not a merge inside this one.

## How cloning works

`src/agentic_chimes/setup/clone_codes.py` holds a `REPOS` dict — one entry
per fork, each with a `url` (SSH, matching how these forks are actually
checked out elsewhere) and a pinned `ref` (a full commit SHA, not a branch
name):

```python
REPOS = {
    "al_driver-LLfork": {"url": "git@github.com:LindseyLab-umich/al_driver-LLfork.git", "ref": "92db0aa..."},
    "chimes_lsq-LLfork": {"url": "git@github.com:LindseyLab-umich/chimes_lsq-LLfork.git", "ref": "75b51f8..."},
    "chimes_calculator-LLfork": {"url": "git@github.com:LindseyLab-umich/chimes_calculator-LLfork.git", "ref": "f4ddddd..."},
}
```

`chimes-agent setup` always calls `clone_codes.ensure_all(...)` as its
first action, regardless of which `--component` was requested — every
build component depends on `codes/` being populated, and re-checking an
already-cloned repo is cheap. `ensure_repo(name, ...)`:

- if `codes/<name>` doesn't exist: `git clone` it, then `git checkout
  <pinned ref>`.
- if it already exists and `--force` wasn't passed: does nothing
  (`"status": "already_present"`).
- if it exists and `--force` **was** passed: `git fetch origin` then
  `git checkout <ref>` again (picks up a moved pin; does not
  force-discard local modifications — a dirty vendored checkout fails the
  checkout with a normal git error rather than being silently reset).

Clone (and fetch/checkout) is a plain `git clone` (not `--depth 1`): since
the pinned ref is an arbitrary historical commit, a shallow clone might not
contain it.

## Why pinned commits, not "always latest main"

Reproducibility: every build component (`chimes_lsq`, `chimes_calculator`,
`lammps`) was developed and validated against these exact commits. Pulling
a moving `main` branch on every `chimes-agent setup` would mean a build
that worked yesterday could silently break today from an upstream change
this repo hasn't accounted for. This mirrors the same pattern already used
elsewhere in the toolchain: `chimes_calculator`'s LAMMPS integration pins a
specific LAMMPS tag (`stable_29Aug2024_update1`), and
`setup/build_quantum_espresso.py` pins a specific QE release tag.

## Bumping a pin

```bash
# edit setup/clone_codes.py: change REPOS["chimes_lsq-LLfork"]["ref"] to the new commit
chimes-agent setup --component codes --force
# re-run whatever build components depend on it, e.g.:
chimes-agent setup --component chimes_lsq --machine dane --force
```

Or, for a one-off without editing the file, `ensure_repo`/`ensure_all`
accept a `ref` override programmatically (not yet exposed as a `--ref` CLI
flag on `chimes-agent setup` — pass it via the Python API,
`agentic_chimes.setup.clone_codes.ensure_repo(name, codes_dir=..., ref=...)`,
or edit `REPOS` directly).

## Standalone clone, no build

```bash
chimes-agent setup --component codes    # no --machine needed -- plain git clone, no compiler
```

Useful for pre-fetching the source (e.g. to read/grep it, or to prepare a
machine before deciding which components to build) without triggering any
compilation.
