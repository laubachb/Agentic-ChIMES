# `chimes-agent deploy`

**Status: implemented.** Packages a study's final model for use.

```bash
chimes-agent deploy --study /p/lustre2/$USER/studies/cuzr [--model-name cuzr-v1]
```

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
