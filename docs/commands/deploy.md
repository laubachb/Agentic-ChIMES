# `chimes-agent deploy`

**Status: implemented.** Packages a study's final model for use.

```bash
chimes-agent deploy --study /p/lustre2/$USER/studies/cuzr [--model-name cuzr-v1]
```

The deployed `params.txt` is MD-ready, unlike the raw fit:

- **Repulsive penalty written explicitly**: `--penalty-dist` (0.02 Å) and
  `--penalty-scaling` (1e5 kcal/mol/Å³). chimes_lsq's documentation says to
  add these before MD; without them chimesFF falls back to 1e4 and 0.01 Å.
  On Cu-Zr at 1200 K the explicit default cut frames sampling distances
  inside the inner cutoff from 20 to 2 (of 81).
- **Zeroed coefficients removed** (`post_proc_chimes_lsq.py`; `--no-reduce`
  to keep them). Predictions are identical (checked to 0.0), and a Cu-Zr
  4-body model with 421 of 726 coefficients nonzero evaluated 21 % faster.

Writes `<study>/06_deploy/` (and registers it):

| file | content |
|---|---|
| `params.txt`, `fm_setup.in` | the model and the basis that produced it |
| `in.lammps.example` | complete NVT input (`pair_style chimesFF`) |
| `MODEL_CARD.md` | what the model is, accuracy, where it is valid, how to run it, cost to run, development cost, data provenance and licenses, caveats |
| `model_facts.json` | the facts the card was written from |

The card's "where it is valid" section lists each pair's minimum sampled
distance: ChIMES applies only a penalty below the inner cutoff, so
configurations closer than that are outside the model. Missing inputs (no
benchmark, no usage report) appear as *gaps*, never as invented values.
