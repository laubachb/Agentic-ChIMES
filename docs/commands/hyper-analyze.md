# `chimes-agent hyper-analyze`

**Status: implemented** (part of the hyperparameter phase, which is still in
development; see [`hyper-search`](hyper-search.md)).

Reads a training set and, per element pair, reports what the ChIMES
hyperparameters should be anchored to. No fitting; a few seconds for a few
hundred frames.

## Usage

```bash
chimes-agent hyper-analyze --data-manifest study/01_data/curate/data_manifest.json \
  --output-dir study/02_fit/analysis
# or
chimes-agent hyper-analyze --train-xyzf train.xyzf --elements Cu,Zr --output-dir ...
```

## What it computes

For every element pair, from ASE neighbour lists over all periodic images
(any cell shape, including cells thinner than the cutoff):

| Output | Meaning | Used for |
|---|---|---|
| `min_distance`, `p01_distance` | smallest / 1st-percentile sampled distance | `suggested.s_minim` = min − `s_minim_delta` (default 0.02 Å, documented range 0.002-0.02) |
| `n_within_1.2x_min` | distances near the minimum | whether the inner region is sampled at all |
| `rdf_peaks`, `rdf_minima` | shell positions from a smoothed r⁻²-weighted histogram | `suggested.morse_lambda` = first peak; shell ends for outer cutoffs |

Globally:

- `candidates.s_maxim_2b` = {5, 6, 7, 8 Å, the largest second-shell end},
  up to `--r-max` (default 8 Å);
- `candidates.s_maxim_3b` = first-shell end, midway, and the second-shell
  end (or first + 2 Å when no second shell is resolved);
- `candidates.s_maxim_4b` = first-shell end and midway to the second shell;
- `nlayers_required` per candidate cutoff, from the thinnest cell
  (`s_maxim ≤ (2·N_LAYERS+1)·width/2`);
- `n_force_equations`, `n_energy_equations`: the rows the fit will have,
  which bound how many coefficients the data can support.

The many-body candidates extend past the documented "first shell" guidance,
because at the first shell ChIMES' cubic smoothing leaves those terms
almost no signal (see [Cutoffs and lambdas](../concepts/cutoffs_and_lambdas.md#many-body-cutoffs-longer-than-the-documented-shells-suggest)).
The search weighs the extra signal against the extra cost.

Distances include periodic images of the same atom, so 1-atom cells (common
in open databases) are analyzed correctly.

On curated MatPES Cu-Zr the first peaks come out at Cu-Cu 2.55 Å, Zr-Zr
3.21 Å and Cu-Zr 2.79 Å (fcc Cu nearest neighbour 2.556 Å, hcp Zr
3.18-3.23 Å).

## Output

`hyper_analysis.json` (read by `hyper-search`), and the same content on
stdout. `notes` flag unsampled pairs, sparse short-range data and missing
RDF features.
