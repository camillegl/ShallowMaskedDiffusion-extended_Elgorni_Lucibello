# Equation-to-code map

Traceability from mathematical components to their implementation, at commit `2e2db70`.
"notes-mem" = `notes/notes_memorization.typ`; "notes-hm" = `notes/notes_hiddenmanifold.typ`;
"NB-corr" = `experiments-analysis/analysis_mmd_distribution_distance_corrected.ipynb`.
Discrepancy IDs refer to `docs/UPSTREAM_DISCREPANCIES.md`.

| Concept / equation | Paper or note location | Current file & symbol | Current behavior | Intended future module | Discrepancy / uncertainty |
|---|---|---|---|---|---|
| Forward masking: `t~U(0,1)`, Bernoulli(t) masks | notes-mem:122,159 | `diffusion.py:72-81` `_compute_loss` | One t per sequence, per-coordinate Bernoulli, mask token 0 | core diffusion module | — |
| Mask variable `m_i ∈ {0,1}`, state `(x, m)` | notes-mem:158 | `models.py:60` `mask = (xt==0)` | In-band scalar 0 encodes mask; indicator recomputed | — | D4 (representation differs, semantics equivalent) |
| Masked objective `−E_t (1/t) E Σ_{i masked} log p_θ` | notes-mem:124-126 | `diffusion.py:83-94` | Sum of BCE·(1/t) over masked / (L·B), mc_samples avg | core diffusion module | D5 (explicit 1/(L·B) implicit in notes) |
| Linear score `σ((1/√L)Σ_j[W_ij(1−m_j)x_j + V_ij m_j])` | notes-mem:164-166 | `models.py:58-64` `LinearBackbone.forward` | `xt@Wᵀ + m@Vᵀ (+b)`; 1/√L only in W init (`models.py:51`) | — | **D3** (no runtime 1/√L) |
| Bias via `W_{ii0}` argument | notes-mem:168 | `models.py:53-56` optional `--bias`, zero-init | Bias off by default; no `W_{ii0}` augmentation | — | minor |
| `V → 0` for uniform data | notes-mem:299 ("[Make this statement more precise!]"), :483 | `models.py:52` V zero-init; `models.py:102-103` `freeze_mask_weights` | V learnable by default; freezable to 0 | — | **D7** (RS-level assertion, not a proof; no HMM derivation) |
| L2 with replica scaling `−½βλ‖w‖²`, M-scaled loss | notes-mem:182 | `diffusion.py:36,46,56-60` `l2coeff`, `sqnorm` | `0.5·λ/(L·α_legacy)·Σ_all-params p²` (includes frozen params) | — | D6 (frozen params regularized) |
| Sampler Algorithm 1: k tokens/step, uniform positions, sample p_θ | notes-mem:137-153 | `diffusion.py:147-231` `sample`, `_sample_k_update` | Fair only; monotone; no revision | sampler module | — |
| Greedy / verygreedy / one-shot reconstruction | not in notes (code-only) | `diffusion.py:233-298` `mask_and_sample`; `:300-379` `mask_and_sample_oneshot` | Threshold decodings; confidence ordering; single-pass reveal | sampler module | code-only extensions |
| U-turn retrieval experiment | notes-mem (U-turn discussion) | `diffusion.py:119-126` `test_step` (driver script `experiments-analysis/run_uturn_experiments.py` retired on `guthlac`, see `docs/archive/HISTORICAL_NOTEBOOKS_ARCHIVE.md`) | Mask fraction t0 of training datum, reconstruct, overlap | — | — |
| Mean-field accuracy ODE (correlations neglected) | notes-mem:702 | not implemented in Python | — | — | theory-only |
| Teacher `F_ia~N(0,1/D)`, `x=sign(Fz)` | notes-hm:43,56,443-444; NB-corr MD cell 2 | `datasets.py:39-53` `RandomFeaturesDataset` | `F=randn(N,D)/√D`, `data=sign(z@Fᵀ)` | — | **D8** (`sign(0)` unpatched in datasets.py) |
| `sign(0) := +1` convention | NB-corr ("Resolve sign ambiguity") | NB-corr cells only (`x[x==0]=1`) | Notebook-only patch | dataset module | **D8** |
| Quenched F across train/val/eval | NB-corr MD cell 2 | NB-corr data cells; `train.py:48-52` | Single dataset object per repeat; fresh-z reference via same F | — | — |
| Ratios `γ=N/D`, `α=M/D` | notes-hm:44; NB-corr MD cells 0-2 | NB-corr (`M_train=round(α·D)`, `config.alpha=M/N`) | Explicit conversion in notebook | config module with contract names | **D1** (alpha conventions) |
| Legacy load `alpha = M/L` | — | `train.py:31,157` | `M=round(alpha·L)` | rename to `visible_load` semantics | **D1** |
| MMD² with kernel `exp(−λ(1−q)/2)`, λ∈{4,8} | NB-corr MD cells 4-5,9 | `src/maskeddiffusion/metrics/mmd.py` (`compute_mmd`); NB-corr cell 11 (historical) | Biased V-stat + unbiased U-stat, chunked | — (migrated Phase 2) | D9 resolved; equivalence tested on small inputs |
| Rényi/Shannon entropy of `P_F` (RS saddle point) | notes-hm:271 (final-RS), :417 (s-hartley), :397 (s-min heur.) | not implemented in Python (`julia-code/hiddenmanifold/` reference) | — | — | theory; RS/ansatz status per notes |
| Clamped-Hopfield RS + MCMC | `notes/notes_hopfield.typ` | `src-hopfield/*.py` | Self-contained side-track | — | — |
| Time-binned linear model | not in notes | `models.py:131-213` `TensorBackbone` | Bins by unmasked count | — | undocumented extension |
| Random-feature score model | notes (generalization TODO) | `models.py:106-129` `RandomFeatureScore` | Fixed random layer + readout | — | D10 (CLI hyphen vs parser underscore) |

Every component above either has an implementation reference or is explicitly marked
missing/theory-only.

**Phase-2 note (2026-07-20).** The active implementations now live in
`src/maskeddiffusion/`: teacher (`teacher.py`, with `sign(0):=+1`), corruption
(`masking.py`), objective (`objectives.py`), linear score (`models.py`, runtime `1/√N`
default + `legacy_init_only` compat mode), samplers (`samplers.py`, named per
`docs/ORIGINAL_ARCHITECTURE.md` correspondence), MMD (`metrics/mmd.py`). The "Current file
& symbol" columns above continue to describe the legacy modules, which remain frozen for
the protected notebooks; regression fixtures tie the two together
(`tests/fixtures/original_architecture_v1/`).
