# Research specification: hidden-manifold extension

Canonical scientific contract for the hidden-manifold (random-features, sign-channel)
extension of the shallow masked diffusion study. Notation: `docs/NOTATION.md`. Upstream
model: `docs/ORIGINAL_ARCHITECTURE.md`. Each section is tagged:
**[DEF]** definition, **[VERIFIED]** verified repository behavior (commit `2e2db70`),
**[INTENDED]** intended future behavior, **[CONJ]** conjecture/open.

## Problem statement [DEF]

How do shallow (linear) masked diffusion models trained on data from a hidden low-dimensional
manifold memorize versus generalize, as a function of the sample ratio `α = M/D` and aspect
ratio `γ = N/D`? The diagnostic is the MMD between generated samples and the finite-F data
law.

## Teacher distribution [DEF, VERIFIED]

- `F ∈ R^{N×D}`, `F_ia ~ N(0, 1/D)`; `z^μ ~ N(0, I_D)`; `x^μ = sign(F z^μ) ∈ {−1,+1}^N`,
  with the convention `sign(0) := +1`.
- Verified: `RandomFeaturesDataset` implements exactly this (`datasets.py:51-53`), except
  that `datasets.py` does **not** apply the `sign(0) := +1` patch — the corrected notebook
  patches `x[x==0]=1` while `train.py` does not (open discrepancy D8). [INTENDED]: patch at
  the dataset level.

## Quenched vs sampled randomness [DEF, VERIFIED]

- Within one repeat: sample `F` **once** (quenched); generate training, validation, and
  fresh evaluation samples with independent latents `z` through the **same** `F`.
- Verified: one dataset object per (D, γ, repeat) holds a single `F`; train/test are
  disjoint row subsets and the MMD reference draws fresh `z` through `full_dataset.F`
  (corrected notebook, data cells; `train.py:48-52` likewise shares `F` across train/val).
- Three distinct distributions, never to be conflated [DEF]:
  1. the empirical training distribution (M samples);
  2. the finite-F law `P_F`;
  3. the disorder average `E_F P_F` (approached only by averaging over explicit repeats).

## Train / validation / evaluation construction [VERIFIED]

- Notebook protocol: `M_train = round(α·D)` rows of the fixed-F dataset; test rows start
  above the largest training load so nested-α training sets never overlap the test rows;
  MMD reference batches are freshly drawn `z` through the same `F` with dedicated seeds.
- `train.py` path: `M = round(alpha_legacy · L)` with `alpha_legacy = M/N` — the legacy
  convention (see `docs/NOTATION.md`; conversion `alpha_legacy = sample_ratio/aspect_ratio`).

## Model family [VERIFIED]

- Linear masked-diffusion denoiser (`LinearBackbone`): `logit = xt Wᵀ + m Vᵀ (+ b)`, no
  runtime `1/√N` factor (init-only; see D3).
- In all hidden-manifold MMD runs to date: `V` frozen at 0 (`freeze_mask_weights=True`) and
  no bias — the effective score is `logit_i = Σ_j W_ij x_{t,j}`. **This is a modeling
  choice/ablation, not a derived property** (see "Open questions").

## Objective [VERIFIED]

The upstream masked-BCE objective with `t ~ U(0,1)` per sequence, Bernoulli(t) masks, `1/t`
weighting, `1/(L·B)` normalization (`diffusion.py:63-94`; full statement in
`docs/ORIGINAL_ARCHITECTURE.md`).

## Sampler-indexed terminal law [DEF, VERIFIED]

- The learned generative object is `P_{θ,A,k}`: the terminal law of sampling algorithm `A`
  resolving `k` tokens per step from the trained conditionals. Default in the MMD study:
  `A` = random-sequential stochastic ("fair"), `k = 1`.
- The independently trained masked single-site conditionals are **not** known to be mutually
  consistent with any joint distribution; no sampler here may be described as exact
  ancestral sampling. Distinct samplers (fair sequential, multi-token, greedy, verygreedy,
  one-shot) define distinct terminal laws and must be named.

## Regimes and order of limits [DEF]

- Finite-size regime: everything measured in this repository is at finite (D, N, M) —
  finite-dimensional empirical evidence.
- Proportional limit: `D → ∞` with `γ = N/D` and `α = M/D` fixed. Any asymptotic statement
  must state which quantities are held fixed and whether it is at fixed `F` (quenched,
  almost-sure) or disorder-averaged; these need not coincide and no self-averaging claim is
  established here. [CONJ] where used.

## Target observables [VERIFIED]

- MMD² between sample sets under the exponential normalized-Hamming kernel
  `k_λ(x,y) = exp(−λ(1−q)/2)`, `q = x·y/N`, kernel scales `λ ∈ {4, 8}` (restricted from a
  larger sweep; λ=16 sits at its own noise floor, small λ measures only mean distance).
  Curves: Model-vs-True (`P_{θ,seq,1}` vs fresh `P_F`), True-vs-True (noise floor),
  Train-vs-True (memorization reference). Both biased V-statistic and unbiased U-statistic
  are computed; negative unbiased values are legitimate finite-sample behavior.
- Retrieval/U-turn overlap diagnostics (upstream observables).

## Permitted claims [DEF]

- "The Model-vs-True MMD decreases with α under kernel scales λ ∈ {4,8}" (empirical).
- "Approaches the finite-F target under this diagnostic."
- "The sampler-induced terminal law `P_{θ,seq,1}` ..."
- "Consistent with improved distributional agreement."
- "Finite-dimensional empirical evidence for ..."

## Prohibited overclaims [DEF]

- "The model learns the distribution" (from an MMD estimate).
- Any sampler-free "model distribution" for generated samples.
- "Exact ancestral sampling" / treating the conditionals as a coherent joint law.
- Capacity thresholds or phase transitions asserted from finite-size plateaus — the
  persistent Model-vs-True gap above the noise floor (commit `2e2db70`) is an **open
  empirical observation**; the notebook itself lists model misspecification, optimization
  error, objective/sampler mismatch, and kernel finite-sample effects as live explanations.
- Transferring the uniform-data `V → 0` result to hidden-manifold data (below).

## Open questions [CONJ]

1. **V and bias under fixed F.** The uniform-data claim that mask weights vanish is, in the
   notes themselves, an RS-level assertion flagged "[Make this statement more precise!]"
   (`notes/notes_memorization.typ:299`) — not a theorem even for uniform data. For a fixed
   hidden-manifold teacher `F`, coordinate exchangeability is broken and no derivation
   exists; `V ≡ 0` in the experiments is an unresolved restriction and a candidate
   contributor to the residual MMD gap.
2. **Source of the persistent MMD gap** (capacity vs optimization vs sampler mismatch vs
   kernel effects) — open.
3. **Joint-law consistency** of the trained single-site conditionals — open; determines
   whether any sampler-free description of the model could ever be justified.
4. **Theory–experiment link**: `notes/notes_hiddenmanifold.typ` computes Rényi/Shannon
   entropies of `P_F` (with its own RS ansätze and open 1RSB questions); it makes **no
   prediction for the MMD**. The connection is currently thematic, not quantitative.
5. **Proportional-limit behavior** of `P_{θ,seq,1}` — no asymptotic result exists.
