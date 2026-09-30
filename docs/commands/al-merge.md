# `chimes-agent al-merge`

**Status: implemented.**

Merges an active-learning round's labeled frames into the training set. It
checks what `cat a.xyzf b.xyzf` would get wrong:

- **One level of theory.** The new frames' `provenance.json` (from
  `qe-relabel --collect`) must match the base data, by the same rules as
  `data-curate`. Active learning must reuse the base set's QE settings.
- **Fixed holdout.** Only the training file grows, so errors stay comparable
  across rounds.
- **File format.** Frames are rewritten together, so a trajectory mixing
  orthorhombic and triclinic cells stays readable by chimes_lsq.
- **Fit flags.** New frames must carry energies if the fit uses them, and
  stresses if it fits stresses.
- **Cycles.** `frame_cycles.json` records each training frame's cycle, for
  `weights --frame-cycles ... --decay-cycles n` (the n/I decay).

```bash
chimes-agent al-merge --data-manifest 01_data/curate/data_manifest.json \
  --new-xyzf 03_al/round1/qe/labeled.xyzf --cycle 1 --output-dir 03_al/round1/merge
# then refit at the chosen hyperparameters on the merged data:
chimes-agent fm-setup-gen --hyper-choice 02_fit/search/best/hyper_choice.json \
  --trjfile 03_al/round1/merge/train.xyzf --nframes <n_train> --output-dir 03_al/round1/fit
chimes-agent amat-build --fm-setup-in 03_al/round1/fit/fm_setup.in --output-dir 03_al/round1/fit
chimes-agent weights --work-dir 03_al/round1/fit --preset hierarchical2026 \
  --frame-cycles 03_al/round1/merge/frame_cycles.json --decay-cycles 8
chimes-agent solve ... --weights 03_al/round1/fit/weights.dat
```

The output `data_manifest.json` reads like the original: `train_xyzf` points
to the merged set, `holdout_xyzf` is unchanged, and `al_rounds` lists what
each round added. Pass it to the next round.
