# `chimes-agent qe-converge`

**Status: implemented** (submission mirrors `qe-relabel`; `--collect` tested on synthetic output).

Convergence tests for the Quantum ESPRESSO settings a study will label
with. Labels are only as good as the settings behind them, and active
learning must reuse the base set's settings, so they are decided once, up
front, on one representative frame (the most compressed or densest frame is
the hardest test).

```bash
chimes-agent qe-converge --structure-xyzf 01_data/curate/train.xyzf --frame-index 12 \
  --elements Cu,Zr --masses '{"Cu":63.546,"Zr":91.224}' --pseudopotentials '{"Cu":"/pp/Cu.upf","Zr":"/pp/Zr.upf"}' \
  --ecutwfc-values 30,40,50,60,80 --kspacing-values 0.5,0.35,0.25,0.18,0.12 \
  --machine dane --queue debug --walltime-hours 2 --output-dir 01_data/qe_converge --dry-run
chimes-agent qe-converge --collect 01_data/qe_converge
```

Two one-dimensional scans run in one job: the `ecutwfc` ladder at the
middle `kspacing`, and the `kspacing` ladder at the largest `ecutwfc`.
`--collect` compares each setting with the finest in its scan on the
quantities ChIMES fits (energy per atom, largest force-component change,
pressure) and recommends the cheapest setting within the thresholds
(defaults 1 meV/atom, 10 meV/Å, 0.1 GPa). Use the recommendation as the
`qe-relabel` settings for the whole study.
