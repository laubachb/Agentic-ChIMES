# ChIMES literature and how the toolkit uses it

The toolkit's defaults, agent playbooks and diagnostics follow the published
ChIMES methodology. This page lists the papers, what the toolkit takes from
each, and where it departs from them (usually a known gap, listed in the
[assessment](../development/assessment.md)).

## Papers

| Paper | DOI | Main use here |
|---|---|---|
| Lindsey, Fried, Goldman, *J. Chem. Theory Comput.* **13**, 6222 (2017): ChIMES for molten carbon | [10.1021/acs.jctc.7b00867](https://doi.org/10.1021/acs.jctc.7b00867) | functional form, inner-cutoff penalty, choosing parameters by convergence, reduced RMSE |
| Lindsey, Fried, Goldman, *J. Chem. Theory Comput.* **15**, 436 (2019): water | [10.1021/acs.jctc.8b00831](https://doi.org/10.1021/acs.jctc.8b00831) | Morse transform, λ at the first RDF peak, per-pair 3-body cutoffs, holdout CV plus MD selection, stress weighting |
| Lindsey, Fried, Goldman, Bastea, *J. Chem. Phys.* **153**, 134117 (2020): active learning for reactive systems | [10.1063/5.0021965](https://doi.org/10.1063/5.0021965) | al_driver's cluster + Monte Carlo selection, LASSO/LARS, Tersoff smoothing, 4-body terms |
| Pham, Lindsey, Fried, Goldman, *J. Chem. Phys.* **153**, 224102 (2020): HN₃ detonation | [10.1063/5.0029011](https://doi.org/10.1063/5.0029011) | a reactive application where 4-body terms were needed |
| Goldman et al., *J. Chem. Phys.* **158**, 144112 (2023): DFTB + ChIMES | [10.1063/5.0141616](https://doi.org/10.1063/5.0141616) | ChIMES as a Δ-learning correction to DFTB |
| Lindsey et al., *npj Comput. Mater.* **11**, 26 (2025): ChIMES Carbon 2.0 | [10.1038/s41524-024-01497-y](https://doi.org/10.1038/s41524-024-01497-y) | current hyperparameter heuristics, weights, parallel and multi-fidelity active learning, validation |
| Lindsey et al., *npj Comput. Mater.* **12**, 18 (2026): hierarchical transfer learning | [10.1038/s41524-025-01863-4](https://doi.org/10.1038/s41524-025-01863-4) | multi-element models built from reusable element blocks, weight decay across active-learning cycles |
| Laubach, Lordi, Lindsey, *J. Chem. Inf. Model.* **66**, 182 (2026): cluster-graph fingerprinting | [10.1021/acs.jcim.5c02179](https://doi.org/10.1021/acs.jcim.5c02179) | dataset coverage, novelty detection, a criterion for when to stop active learning |
| Schwalbe-Koda, Hamel, Sadigh, Zhou, Lordi, *Nat. Commun.* **16** (2025): information entropy of atomistic datasets (QUESTS) | [10.1038/s41467-025-59232-0](https://doi.org/10.1038/s41467-025-59232-0) | model-free dataset entropy and saturation, novelty (δH) of environments, entropy-maximizing selection |

The equations and figures behind these uses are on the Methods pages:
[the ChIMES model](chimes_model.md), [data selection and coverage](data_selection.md),
[active learning](active_learning.md), [how a model is chosen](model_selection.md)
and [the workflow in pictures](workflow.md).

## What the toolkit adopts

- **Inner cutoff** just below the closest sampled contact per pair, not
  searched (`hyper-analyze`, `auto-build`). This follows Lindsey 2017 and
  2025.
- **Morse λ** at the first RDF peak per pair, with a small ±10 % check in the
  `lambda` stage. Lindsey 2017 found fits insensitive to λ over a broad
  range, and Lindsey 2019 established the peak rule.
- **Choosing the smallest converged model.** `hyper-search` takes the
  cheapest model that is statistically tied with the best, which
  operationalizes the "smallest parameters giving a converged result" rule
  (Lindsey 2017).
- **LASSO/LARS** as the default solver (`lassolars`, DLARS for large
  problems), per Lindsey 2020 and 2025.
- **al_driver's selector** in `al-select` and `al-run` (Lindsey 2020).
- **One level of theory per fit**, and active-learning labels with the same
  settings as the base set.
- **Reduced RMSE.** `evaluate` reports `reduced_force_rmse` = RMSE ÷ mean
  |F_ref|, the convention every paper above uses. Published values: molten
  carbon 0.44 (2017) → 0.28 (Carbon 2.0); water 0.24-0.31. The toolkit's
  `relative_force_error` (÷ RMS force) is smaller for the same model (0.31 vs
  0.46 on the Cu-Zr example). Only compare like with like.

## Where the toolkit departs (and what to do)

| Topic | Literature | Toolkit | What to do |
|---|---|---|---|
| Smoothing | TERSOFF (f_O 0.5-0.75) for models with more than 3-body terms, because the cubic form shrinks many-body terms | CUBIC by default; `--smoothing 'TERSOFF 0.5'` available | see the Cu-Zr comparison in [hyper-search](../commands/hyper-search.md) |
| 3-body cutoffs | may differ per pair (water: each pair at its own third shell) | `hyper-search` `3b_pairs` stage; `fm-setup-gen special_maxim_3b_pairs` | verified consistent across chimes_lsq, chimes_calculator and LAMMPS |
| Model selection | holdout CV, then MD against DFT for the top candidates | holdout search + [md-check](../commands/md-check.md) | run `md-check` on the tied finalists with a DFT reference |
| Complexity before active learning | err toward richer models, prune after the data is final | `hyper-search --prefer richer` | use it for ALC-0 models |
| Fitting weights | forces 1, energies 0.3-5, stresses 100; AL frames decayed as n/I | [weights](../commands/weights.md) presets + al_driver methods + decay | pick a preset; default is still uniform |
| Stresses | needed for density and pressure; weights 100-250 | carried from open data and QE (sign verified per dataset); `hyper-search` fits them and measures the stress weight | published weights assume large cells: on 2-atom cells 100 wrecked the fit, 3 was best |
| Validation | RDF, EOS, diffusion, spectra, speciation vs DFT | `md-check`: stability, close contacts, RDF vs reference | EOS, diffusion by hand |
| Multi-element fitting | hierarchical element blocks (al_driver 2.0) | [hierarch](../commands/hierarch.md) (subtract, combine) | tied with all-at-once on Cu-Zr |
| Coverage / stopping AL | cluster-graph fingerprints | [fingerprint](../commands/fingerprint.md) (native, matches the shipped tool); QUESTS entropy alongside | type-agnostic by default; `--structure-weight` adds element awareness |

## For agents

The same content, organized by decision and with the quantitative detail,
is the `chimes-literature` skill (`.claude/skills/chimes-literature/SKILL.md`).
If you keep PDFs in `chimes_papers/` (gitignored), run
`tools/index_papers.sh` to build a searchable text copy the agents can
`grep` for detail. Add your own papers to that folder and re-run the script.
