# `chimes-agent al-select`

**Status: implemented.** Requires the `al-select` extra (`pip install
agentic-chimes[al-select]`) for `matplotlib`/`cycler` — al_driver's own
`gen_selections.py` imports them unconditionally for its diagnostic plots.
If you want active learning today without that extra, see
[`al-run`](al-run.md), which launches al_driver's *full* loop (including
its own diversity selection internally via `gen_selections.py`) rather
than exposing selection as a standalone step — `al-select` here is for
calling that same selection logic on its own, outside a full driver cycle.

## Intent

Diversity-based active-learning batch selection, wrapping al_driver's
existing Metropolis-MC energy-histogram selector
(`codes/al_driver-LLfork/src/gen_selections.py:gen_subset`) — the actual
selection algorithm is al_driver's own, imported and called in-process,
not reimplemented. **This is coverage/diversity selection, not
uncertainty/query-by-committee** — al_driver's own
`doc/source/files_to_finish/committeeALmode.rst` is an empty stub
confirming committee-based AL was never finished upstream, and this repo
does not attempt to finish it here either.

The concrete future plug-in point for uncertainty-based selection is
`chimes-agent evaluate`'s `committee_spread` output (multiple `params.txt`
models diffed on the same unlabeled candidate frames, picking the
highest-disagreement ones) — see `docs/commands/evaluate.md`. Deliberately
not wired up in this stage.

## How selection works

1. Each candidate frame's ChIMES-predicted energy is computed in-process
   via the same ctypes evaluator `evaluate` uses, then normalized per atom
   (`energy / natoms`) — matching al_driver's own on-disk convention
   confirmed against `codes/al_driver-LLfork/utilities/new-get_dumb_ener_subjob.sh`'s
   `paste xyzlist.dat xyzlist.energies | awk '{print $NF/$1}'`.
2. Those per-atom energies are written to `all.energies_normed`-equivalent
   input and handed to al_driver's own `gen_subset`, which runs a
   Metropolis-MC sweep that biases toward flattening the selected subset's
   energy histogram (i.e. picks a set that covers the observed energy
   range evenly, not just the most common region) — always keeping the
   observed min/max-energy frames.
3. The winning indices come back from `gen_subset`'s own
   `all.selection.dat` output and are mapped back onto the original
   candidate frames.

## Usage

```bash
chimes-agent al-select \
  --candidate-frames unlabeled_pool.xyzf \
  --params ./current_model/params.txt \
  --n-select 50 --histogram-bins 20 \
  --output-dir ./al_select_run
```

Chaining across cycles (biasing a later selection against what an earlier
one already picked):

```bash
chimes-agent al-select \
  --candidate-frames next_pool.xyzf --params ./current_model/params.txt \
  --n-select 50 --central-repo ./al_select_run/central_repo_energies_normed.txt \
  --output-dir ./al_select_run_2
```

## Flags

- `--candidate-frames PATH` (required) — unlabeled/candidate `.xyzf` pool;
  forces/energy fields, if present, are ignored (only positions/box feed
  the energy prediction).
- `--params PATH` (required) — a `params.txt`; its predicted energy per
  frame is the selection signal.
- `--n-select N` (required) — number of frames to select; must not exceed
  the candidate pool size.
- `--central-repo PATH` (optional) — a plain energies file from a prior
  `al-select` call's `central_repo_out`, biasing the histogram toward
  covering gaps in what's already been picked across cycles.
- `--histogram-bins N` (default 20) — `gen_subset`'s `nbins`.
- `--nsweep N` (optional) — MC sweep cycles per candidate (multiplied
  internally by the pool size, matching `gen_subset`'s own convention);
  defaults to `max(1, histogram_bins // 10)`, mirroring al_driver's own
  documented `MEM_CYCL` default.
- `--energy-cutoff VALUE` (default `1e10`) — candidates with
  `abs(energy_normed) >= this` are excluded (`gen_subset`'s `ecut`).
- `--seed N` (default 1).

## Output

```json
{
  "n_candidates": 200,
  "n_selected": 50,
  "selected_indices": [2, 4, 10, "..."],
  "selected_xyzf": "./al_select_run/selected.xyzf",
  "central_repo_out": "./al_select_run/central_repo_energies_normed.txt",
  "energy_histogram_pdf": "./al_select_run/energy_hist.pdf",
  "residuals_pdf": "./al_select_run/residuals.pdf"
}
```

`selected_xyzf` is ready to hand to [`qe-relabel`](qe-relabel.md) for
real labeling. `central_repo_out` is a plain energies-per-line file (the
selected frames' energies, appended onto whatever `--central-repo` was
given) — pass it as the next cycle's `--central-repo` to keep selection
diverse across repeated calls, without needing al_driver's own
`ALC-<n>`/`CENTRAL_REPO` directory layout.

## Known limitations

- **Standalone, not a full `ALC-X`/`CENTRAL_REPO` integration.**
  al_driver's own `populate_repo`/`cleanup_repo` assume a `../ALC-<n>/`
  and `../CENTRAL_REPO/` on-disk layout tied to a live `main.py` run (the
  same reasoning `al-run` already documents for not generating
  `config.py`). `--central-repo`/`central_repo_out` here are a simpler,
  standalone chaining mechanism instead.
- **`gen_subset` can call Python's bare `exit()`** on a few internal error
  paths (e.g. a degenerate histogram with too few candidates per bin)
  rather than raising a catchable exception — this stage pre-checks the
  common case (`n_select` vs. pool size) to avoid the most likely trigger,
  but an unusual candidate pool could still hit one of those paths and
  terminate the process directly instead of returning a clean CLI error.
