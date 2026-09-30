# `chimes-agent doctor`

**Status: implemented.**

Checks that an installation can run a study, and prints the fix command for
anything that is wrong. Run it after `chimes-agent setup`, on a new machine,
or when a stage fails in a way that looks environmental.

```bash
chimes-agent doctor                 # local checks (about 10 s)
chimes-agent doctor --machine dane  # plus the machine profile
chimes-agent doctor --quick         # skip the numerical checks
```

## What it checks

| check | fails when |
|---|---|
| components | `chimes_lsq`, `chimescalc`, LAMMPS missing (DLARS and QE `pw.x` are warnings: needed only for those features) |
| `al_driver` | the vendored fork is not cloned |
| Python packages | `ase`, `pyarrow`, `scikit-learn` missing (warnings) |
| `HF_TOKEN` | unset (warning: open-data downloads may hit HTTP 429) |
| literature index | `chimes_papers/` present but `tools/index_papers.sh` not run (warning) |
| numerics | chimes_calculator or LAMMPS does not reproduce the published CHON reference energy/forces (the same reference the tests use) |
| machine profile | profile does not load; empty account (`CHIMES_ACCOUNT`); `scratch_root` missing, not writable or node-local; no `sbatch`; a partition that `sinfo` does not know |

## Output

```json
{"ok": true, "n_fail": 0, "n_warn": 2,
 "checks": [{"check": "LAMMPS (chimesFF) numerics", "status": "ok",
             "detail": "CHON reference energy -7.8371473 (expected -7.8371473)"},
            {"check": "HF_TOKEN", "status": "warn", "detail": "unset: ...",
             "fix": "export HF_TOKEN=<Hugging Face read token>"}]}
```
