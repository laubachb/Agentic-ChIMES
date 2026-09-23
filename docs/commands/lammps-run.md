# `chimes-agent lammps-run`

**Status: implemented (local execution only; HPC submission for long MD
runs is future work).**

Single-point or MD evaluation via the ChIMES-patched `lmp_mpi_chimes`
build (`chimes-agent setup --component lammps`). See
`src/agentic_chimes/stages/lammps_run.py` and
`src/agentic_chimes/io/lammps_data.py`.

Single-point mode (`run 0`) is the documented independent cross-check
against `evaluate`'s ctypes path — both read the same `params.txt` and
should agree on forces/energy for a given configuration. Validated in
`tests/unit/test_lammps_run.py` against the same published reference
`evaluate`/`test_evaluate.py` is validated against: LAMMPS, the ctypes
wrapper, and the standalone `chimescalc` binary all agree to ~1e-3.

## Usage

```bash
chimes-agent lammps-run \
  --params ./model/params.txt \
  --structure-xyzf ./structures.xyzf --frame-index 0 \
  --elements C,H,O,N --masses '{"C":12.011,"H":1.0079,"O":15.9994,"N":14.007}' \
  --mode single_point \
  --output-dir ./lmp_run

# short MD
chimes-agent lammps-run ... --mode md --temperature 300 --nsteps 5000 --timestep 0.5 \
  --output-dir ./lmp_md_run
```

## Correctness-critical: `elements`/`masses` must match the params.txt

LAMMPS atom-type IDs are assigned in the order of `--elements`, and
`pair_coeff * * params.txt` maps those IDs to params.txt's *internal* type
indices positionally, not by element symbol — `elements` here must be the
same ordering `fm-setup-gen` used to build that params.txt. Masses must
match the params.txt's declared masses to within ChIMES' ~0.001 amu
tolerance. Neither is cross-checked automatically (no params.txt parser
exists yet) — get this wrong and LAMMPS will silently evaluate the wrong
pair/cluster types.

## Flags

- `--params PATH` (required)
- `--structure-xyzf PATH` (required) — a `.xyzf` file; `--frame-index`
  selects which frame (default 0)
- `--elements`, `--masses` (required)
- `--mode {single_point,md}` (default `single_point`)
- `--temperature`, `--nsteps`, `--timestep` (fs), `--md-seed` (mode=md only)
- `--nprocs N` — runs via `mpirun -n N` if >1, direct exec otherwise
- `--lammps-bin PATH` — override the resolved binary

## Output (single_point)

```json
{
  "work_dir": "./lmp_run", "mode": "single_point",
  "in_lammps": "./lmp_run/in.lammps", "log": "./lmp_run/log.lammps",
  "data_file": "./lmp_run/structure.data", "dump": "./lmp_run/dump.out",
  "forces_kcal_mol_ang": [[-19.8826, 20.5959, 12.0526], ...],
  "energy_kcal_mol": -7.8371473
}
```

Forces are parsed from the LAMMPS dump file (a stable, simple format,
sorted by atom id); energy is parsed from the thermo log (best-effort —
we control the exact `thermo_style custom step pe` format we ask LAMMPS
to print, so this is reliable for single-point, but isn't attempted for
`md` mode's multi-column thermo table).

## Known limitation: orthorhombic cells only

`io/lammps_data.py`'s data-file writer doesn't support triclinic
(non-orthorhombic) boxes — raises a clear error rather than silently
writing a wrong/ignored box.
