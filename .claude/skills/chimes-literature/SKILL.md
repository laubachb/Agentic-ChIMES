---
name: chimes-literature
description: Published ChIMES methodology distilled for decisions - inner/outer cutoffs, Morse lambda, smoothing function (cubic vs Tersoff), polynomial orders and model selection, solver/regularization, fitting weights (force/energy/stress), training-data design, active learning, validation criteria, fingerprinting, multi-element (hierarchical) fitting, and published accuracy/cost benchmarks. Use whenever choosing or justifying a ChIMES hyperparameter, weight, solver or AL setting, when comparing a model's accuracy to published models, or when writing up methods. Every agent consults it before making a technical recommendation.
---

# ChIMES literature (for agents)

Eight papers by the ChIMES developers, distilled into guidance. Keys like
**[W19]** refer to the table below. When you rely on a point, cite it
(author, year, DOI) in what you write. For detail, search the local text,
which is gitignored and may be absent on another machine:

```bash
tools/index_papers.sh                                   # (re)build chimes_papers/text/ from chimes_papers/*.pdf
grep -n -i "stress\|weight" chimes_papers/text/2019_JCTC_Lindsey_ChIMES_water.txt
```

If `chimes_papers/` does not exist, use this summary and the DOIs, and say
you did not consult the full text. Paraphrase; do not paste passages.
Users can add their own PDFs to `chimes_papers/` and re-run the script
(`text/INDEX.txt` lists everything).

| Key | Paper | DOI | Local text | Use it for |
|---|---|---|---|---|
| C17 | Lindsey, Fried, Goldman, *JCTC* 13, 6222 (2017): ChIMES for molten carbon | 10.1021/acs.jctc.7b00867 | `2017_JCTC_Lindsey_ChIMES_molten_carbon.txt` | functional form, penalty, SVD fitting, choosing parameters by convergence |
| W19 | Lindsey, Fried, Goldman, *JCTC* 15, 436 (2019): water at ambient conditions | 10.1021/acs.jctc.8b00831 | `2019_JCTC_Lindsey_ChIMES_water.txt` | protocol for user parameters, holdout CV of orders, stress weighting, transforms |
| AL20 | Lindsey, Fried, Goldman, Bastea, *JCP* 153, 134117 (2020): active learning for reactive systems | 10.1063/5.0021965 | `2020_JCP_Lindsey_active_learning_reactive.txt` | al_driver's cluster + Monte Carlo selection, memory modes, LASSO/LARS, 4-body, Tersoff smoothing |
| HN20 | Pham, Lindsey, Fried, Goldman, *JCP* 153, 224102 (2020): HN3 detonation | 10.1063/5.0029011 | `2020_JCP_Pham_HN3_detonation.txt` | reactive application; 4-body necessity; LASSO-selected orders |
| DF23 | Goldman et al., *JCP* 158, 144112 (2023): DFTB + ChIMES | 10.1063/5.0141616 | `2023_JCP_Goldman_DFTB_ChIMES.txt` | ChIMES as a Δ-learning correction to DFTB |
| C24 | Lindsey et al., *npj Comput. Mater.* 11, 26 (2025): ChIMES Carbon 2.0 | 10.1038/s41524-024-01497-y | `2024_npjCM_Lindsey_ChIMES_carbon_2.0.txt` | modern hyperparameter heuristics, weights, parallel/multi-fidelity AL, validation |
| HT25 | Lindsey et al., *npj Comput. Mater.* 12, 18 (2026): hierarchical transfer learning | 10.1038/s41524-025-01863-4 | `2025_npjCM_Lindsey_hierarchical_transfer_learning.txt` | multi-element models from reusable element blocks; standard weights and AL weight decay |
| FP26 | Laubach, Lordi, Lindsey, *JCIM* 66, 182 (2026): cluster-graph fingerprinting | 10.1021/acs.jcim.5c02179 | `2026_JCIM_Laubach_cluster_graph_fingerprinting.txt` | dataset coverage, novelty detection, AL stopping |

## Inner cutoff and penalty

- r_c,in is not a tunable: it sits at, or ~0.02 Å below, the smallest pair
  distance sampled in the training data, per pair **[C17, C24]**.
- Penalty below r_c,in: prefactor 10⁵ kcal/mol/Å³, onset d_p 0.01-0.02 Å.
  Keep it as small as possible without contaminating the conserved quantity
  **[C17, W19, AL20, C24]**. Penalties are 2-body only.
- Close contacts are what break stability. Active learning deliberately
  harvests frames with some r in (r_c,in, r_c,in + d_p] **[C24, HT25]**.

## Distance transform and Morse λ

- The Morse-style transform beats "direct" and "inverse" **[W19]**.
- λ = first RDF peak for each pair, i.e. a characteristic bond length
  **[W19, C24]**. Fit quality is insensitive to λ within ~0.75-1.5 Å for
  carbon **[C17]**. Published values are 1.09-1.40 Å for C/N **[HT25]** and
  1.25-1.40 Å for carbon **[C24]**.

## Outer cutoffs

- Order them r_out,2B ≥ r_out,3B ≥ r_out,4B **[C24]**.
- The 2-body cutoff should reach the third non-bonded solvation shell
  **[C24]**. Water used 9.86 Å (3rd-4th O-O shell) **[W19]**. Carbon moved
  from 3.15 Å (2017) to 5 Å for 3-body and 4.5 Å for 4-body in 2024
  **[C24]**.
- 3-body cutoffs can differ by pair within a triplet (water {O-O 8, O-H 6,
  H-H 5} Å, each at its own third shell). This performed like a uniform 8 Å
  but costs less **[W19]**. `fm_setup.in` supports it via
  `SPECIAL 3B S_MAXIM: SPECIFIC`; the toolkit's search uses one global
  3-body cutoff (a known gap).
- Cutoffs above half the training cell hit periodic-image issues. The
  published fix is to add DFT frames of larger cells (7 extra 768-atom frames
  allowed 9.86 Å for water) **[W19]**. `N_LAYERS` is the alternative
  chimes_lsq offers.
- Short cutoffs miss long-range physics such as graphite interlayer
  dispersion and pressure; add explicit dispersion or longer cutoffs
  **[C17, C24]**.

## Smoothing function (important for 3- and 4-body terms)

- The cubic smoothing multiplies one factor per cluster distance, which "can
  otherwise severely reduce" n-body contributions for n > 2 **[AL20]**.
- The Tersoff-style form leaves interactions unmodified below
  r_out(1 − f_O). It is the recommended choice for models with more than
  3-body terms (chimes_lsq docs: `# FCUTTYP # TERSOFF <f_O>`, f_O typically
  0.25-0.5). Published values: f_O = 0.5 **[AL20, HN20]**, 0.75 **[C24]**.
- `hyper-search` defaults to CUBIC (`--smoothing 'TERSOFF 0.5'` switches).
  On Cu-Zr, CUBIC left many-body columns 1e-3 (3-body) to 1e-7 (4-body) of
  the 2-body scale, which is the effect **[AL20]** describes.

## Polynomial orders and model selection

- Choose the smallest order (and cutoff, λ) that gives a converged result
  **[C17]**. This matches the toolkit's "cheapest statistically tied model"
  rule.
- Toolkit: `md-check` runs this MD comparison. `hyper-search --prefer richer`
  implements "err toward complexity" (next points).
- Holdout cross-validation over 2-body {4..28} × 3-body {0..14}: the error
  kept falling to the largest orders, but MD showed overfitting
  (over-/under-structured RDFs, singularities at 3-body 10) and underfitting
  (dissociation at 3-body 2). The final pick (12/4) was made by MD
  comparison to DFT **[W19]**. **Validate the top holdout candidates in MD
  before choosing.**
- A sparse initial set makes holdout CV recommend models that are too
  simple. Err toward higher complexity before active learning, then refit at
  lower complexity once the training set is final **[C24]**. So for an
  ALC-0 model, do not over-apply parsimony.
- 4-body terms become necessary for slow chemistry and structurally complex
  species (torsions): C/O **[AL20]**, HN3 **[HN20]**.
- Published orders:

  | system | 2-body/3-body/4-body | parameters |
  |---|---|---|
  | molten C | 12/4 | 48 |
  | water | 12/4 | |
  | C/O | 12/7/3 | ≤3,978 |
  | HN3 | 12/8 + 4-body | 808 before 4-body |
  | C/N | 25/10/4 | |

  **[C17, W19, AL20, HN20, HT25]**. LASSO leaves fewer parameters than the
  orders imply (~10 % fewer) **[C24]**.

## Solver and regularization

- Early models used SVD with a relative singular-value cutoff ε
  (10⁻⁵ **[C17]**; 6.667×10⁻⁵ **[W19]**).
- Since 2020, regularization "must be used for most problems": LASSO solved
  by LARS, distributed (DLARS) for large problems **[AL20, C24]**.

## Fitting weights (per data type; units are the inverse of the property's)

| study | forces | energies | stresses | note |
|---|---|---|---|---|
| **[AL20]** | 1.0 | 5.0 | – | |
| **[C24]** Large | 1.0 | 0.1 | 100 | |
| **[C24]** Small | Boltzmann-style (∝ exp(−\|x\|/scale)) | | | emphasizes the typical range |
| **[HT25]** | 1.0 | 0.3 | 100 | |
| **[W19]** | 1 | – | 250 Å² | best for density/pressure; w_σ = 1 let the liquid evaporate in NpT; 250-1000 all recovered density |

- Weights balance the 3N force rows per frame against one energy and a few
  stresses **[C17, W19]**.
- **Force-only fits do not constrain pressure or density** **[C17, W19]**.
  Include stresses when the application runs at constant pressure. High
  stress weight with 3-body terms can go singular (w > 500) **[C17]**.
- During active learning, decay weights as w = n_cycles / I (I = current
  cycle), so early unphysical frames cannot pull the fit away from the
  ground-truth data **[C24, HT25]**.
- Toolkit: the `weights` stage implements these as presets (`al_driver`,
  `lindsey2020`, `carbon2_large`, `hierarchical2026`) plus al_driver's
  methods A-G and `--decay-cycles`. `model-build` and `hyper-search` take
  `--weights-preset`.

## Training data

- Decorrelated DFT-MD frames. 26 frames × 256 atoms converged the liquid
  carbon fit **[C17]**. C/N used 10 simulations × 20 frames plus Materials
  Project cell relaxations under pressure **[HT25]**.
- Cover every state point the model must describe; 2-body-only models
  transfer poorly **[C17]**.
- Self-consistent augmentation: fit, run MD, label a fraction of the
  "faulty" frames, refit **[W19]**.
- Multi-fidelity: explore with DFTB (orders of magnitude cheaper), then
  relabel the selected set with DFT **[C24]**.

## Active learning (what al_driver implements)

- Each active-learning cycle (ALC) combines iterative refinement with active
  learning. MD frames are decomposed into molecule-like clusters (two-pass
  distance clustering). A Monte Carlo selection flattens the histogram of
  ChIMES-predicted per-atom cluster energies (Shannon-information criterion),
  and the selected clusters are labeled by DFT **[AL20]**. `al-select` wraps
  this selector.
- The method is largely insensitive to memory mode. Partial memory with 40
  bins was best for speciation, converging in about 8 ALCs **[AL20]**.
- Parallel active learning runs MD at several state points. Per state point
  it takes up to 20 frames with close contacts plus up to 20 others, to cover
  timescales beyond the DFT data **[C24, HT25]**.
- Stopping by inspection: stable (conserved quantity), with RDF, equation
  of state and dynamics matching DFT **[C24]**. Quantitatively, stop when
  ChIMES-generated configurations are statistically indistinguishable from
  DFT ones by fingerprint **[FP26]**.

## Validation

- Stability: the conserved quantity holds.
- Structure: RDF against DFT, and the first-peak potential of mean force
  ΔW = −k_B T ln g(r_peak) **[C17]**.
- Dynamics: vibrational power spectrum and diffusion **[C17, W19]**.
  Diffusion from short trajectories is unreliable; compare at equal length
  **[W19]**.
- Thermodynamics: pressure and equation of state, density in NpT
  **[W19, C24]**.
- Chemistry: speciation and lifetimes for reactive systems **[AL20, HN20]**.
- Statistics: several independent trajectories, with standard errors
  **[W19]**. Test transferability at nearby state points **[C17]**.

## Accuracy and cost benchmarks (compare with `evaluate`'s `reduced_force_rmse`)

- The literature's reduced RMSE is RMSE / ⟨|F_DFT|⟩ **[C17, W19, C24]**.
  Published values:
  - molten C 2017: 2-body 0.678, 3-body 0.439 **[C17]**;
  - water: 0.24-0.31 **[W19]**;
  - Carbon 2.0 Large: force 0.28, energy 0.01, stress 0.41 **[C24]**.
- The toolkit's `relative_force_error` divides by the RMS force instead and
  is smaller: 0.31 vs 0.46 reduced on the Cu-Zr example. Never compare
  across the two conventions.
- Cost: for 5 ps of 256-atom liquid carbon, DFT took ~50,000 CPU-h,
  ChIMES-2B ~0.15 and ChIMES-3B ~5 CPU-h **[C17]**. 3-body terms dominate
  cost.

## Multi-element models

- Parameters split into element blocks:
  - pure blocks {X, XX, XXX, XXXX} can be fitted from single-element data;
  - cross blocks {XY, XXY, XYY, …} are fitted afterwards to mixtures, with
    the pure blocks fixed.

  This hierarchical approach performed as well as fitting everything at
  once, and blocks are reusable across systems **[HT25]**. al_driver 2.0
  implements it (`codes/al_driver-LLfork/src/hierarch.py`); the toolkit does
  not expose it yet. It is a principled alternative to excluding cross-type
  clusters.

## Fingerprinting and coverage

- A cluster-graph fingerprint is built from histograms (100 bins, Morse
  transform) of dissimilarities between same-order clusters, for 2-, 3- and
  4-body clusters. Mahalanobis D² compares datasets, and D_j² tests whether
  one configuration is novel **[FP26]**.
- Uses: coverage checks for dataset curation, novelty detection for active
  learning, detecting equilibration, and the active-learning stopping
  criterion **[FP26]**.
- Currently single-element. The tool ships in chimes_calculator
  (`chimesFF/src/FP`, example `etc/lmp/tests/example-fingerprint`; build
  LAMMPS with `etc/lmp/install.sh FINGERPRINT`). Not yet wrapped by the
  toolkit.

## ChIMES with DFTB

- ChIMES can Δ-learn the repulsive and many-body correction to DFTB,
  reaching near-hybrid/CC accuracy with few parameters **[DF23]**. It is
  relevant only when the target method is DFTB-based.
