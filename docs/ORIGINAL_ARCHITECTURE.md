# Original masked-diffusion architecture (as implemented)

Canonical description of the upstream model, reconstructed from source at commit `2e2db70`.
"Notes" references are to `notes/notes_memorization.typ`, which holds the authoritative
upstream equations. At the Phase 2 baseline, `paper/main-neuralnetworks.typ` was a
non-authoritative stub (see `docs/PROVENANCE.md`); it was removed in Phase 3A and remains
recoverable from the stable tag `phase2-hidden-manifold-foundation`. Where the notes and code
disagree, both are stated; discrepancies are catalogued in `docs/UPSTREAM_DISCREPANCIES.md`.

## Data model

- Clean data are Ising vectors `x0 ∈ {−1,+1}^L` (`datasets.py`): `UniformIsingDataset`
  (i.i.d. uniform ±1), `RandomFeaturesDataset` (hidden-manifold: `x = sign(z Fᵀ)`,
  `F = randn(N,D)/√D`, `datasets.py:39-53` — note `sign(0)=0` is *not* patched here),
  `BinarizedMNIST`.
- Legacy `L` is the visible dimension (`N` in the contract notation of `docs/NOTATION.md`).

## Mask representation

- The mask token is the in-band scalar `0` (`self.mask_index = 0`, `diffusion.py:15`), a
  value clean ±1 data never takes. Masked positions of `xt` are set to 0 (`diffusion.py:81`).
- Backbones recompute the mask indicator from the input: `mask = (xt == 0)`
  (`models.py:60`). The notes instead describe a tuple `(x, m)` with an explicit mask
  variable (notes:158); semantically equivalent, representationally different.

## Forward corruption

- Per training sequence: `t ~ Uniform(0,1)` once, shared across all L positions
  (`diffusion.py:72,76`); each coordinate masked independently with probability `t`
  (`mask = rand_like(x0) < t`, `diffusion.py:79`). Matches notes:122,159.
- All masked coordinates are prediction targets simultaneously; there is no single
  distinguished target position.

## Objective (`_compute_loss`, `diffusion.py:63-94`)

Per Monte-Carlo sample (default `mc_samples=1` in training; 10 in test):

```
loss = (1/(L·B)) · Σ_batch Σ_{i masked} (1/t) · BCE_with_logits(logit_i, (x0_i+1)/2)
```

- `1/t` weighting per masked position, using that sequence's `t` (`diffusion.py:88`).
- Sum over masked entries, normalized by `L · batch_size` (`diffusion.py:89`), then averaged
  over `mc_samples` (`diffusion.py:94`).
- Notes objective (notes:124-126): `−E_{t~U(0,1)} (1/t) E Σ_{i masked} log p_θ(x0_i|x^t)` —
  matches up to the explicit `1/(L·B)` normalization, which the notes leave implicit.
- Accuracy metric: `logits > 0` versus targets, mean over masked entries.

## Linear score (`LinearBackbone`, `models.py:40-103`)

Implemented forward (`models.py:58-64`):

```
logit = xt @ Wᵀ + m @ Vᵀ  (+ b if bias)      # m = 1[xt == 0]
```

- Because masked entries of `xt` are 0, `W` effectively acts on unmasked tokens only.
- **No runtime `1/√L` factor.** The notes write `σ((1/√L) Σ_j [W_ij(1−m_j)x_j + V_ij m_j])`
  (notes:164-166); the code carries `1/√L` only in the initialization `W = randn(L,L)/√L`
  (`models.py:51`). At fixed L this is a reparametrization of the trained model, but it
  changes effective learning-rate/regularization scaling across L. (Discrepancy D3.)
- `V` initialized to zeros (`models.py:52`), learnable by default;
  `--freeze-mask-weights` pins it at 0 (`models.py:102-103`).
- Bias off by default (`--bias`); zero-initialized if enabled. The notes argue a bias is
  unnecessary because `W_{ii0}` can play its role (notes:168); no such augmentation exists
  in code.
- No diagonal constraint (Hebbian option can zero the diagonal; `train.py` calls with
  `zero_diagonal=False`). No time conditioning: `t` never enters the backbone.
- Optional Hebbian initialization `W = (XᵀX)/L`, then frozen (`models.py:66-93`,
  `--hebbian`).
- Logged order parameters: `qW = mean(W²)·L`, `qV = mean(V²)·L` (`models.py:96-100`).

## Model variants

- `RandomFeatureScore` (`models.py:106-129`): fixed random first layer
  `W1 = randn(H,L)/√L`, `V1 = 0` (both frozen), learnable readout `W2 = randn(L,H)/√H`;
  `out = act(xt W1ᵀ + m V1ᵀ) W2ᵀ`. Selected by `--model rfs<expansion>_<act>` — note the
  parser splits on underscore (`diffusion.py:21`) while the CLI help shows a hyphen
  (discrepancy D10).
- `TensorBackbone` (`models.py:131-213`): per-bin linear models; the bin is chosen by the
  count of *unmasked* tokens (`models.py:154-157`) — a discretized time conditioning with no
  counterpart in the notes.

## Regularization

- `l2coeff = 0.5 · l2reg / (L · alpha_legacy)` ≈ `0.5 · λ/M` (`diffusion.py:36`), applied to
  the squared norm of **all** registered parameters, including `V`, bias, and even frozen
  parameters such as `RandomFeatureScore.W1/V1` (`diffusion.py:56-60`) — frozen parameters
  inflate the reported `l2loss` without being optimizable.
- AdamW `weight_decay=0`; regularization is purely in the loss (`diffusion.py:132-134`).
- The `1/M` scaling matches the replica convention `−½βλ‖w‖²` with M-scaled data loss
  (notes:182).

## Samplers (each distinct; none ever revises a committed token)

1. **`sample` / `_sample_k_update`** (`diffusion.py:147-231`) — generative sampler,
   `P_{θ,seq,k}`. Starts fully masked (unless given a partial `xt`); per step chooses
   `min(k, #masked)` masked positions uniformly at random and samples each token
   `2·Bernoulli(σ(logit))−1`. Always stochastic ("fair"); no greedy option in this path.
   Implements the notes' Algorithm 1 (notes:137-153).
2. **`mask_and_sample`** (`diffusion.py:233-298`) — sequential reconstruction (U-turn):
   masks `T0` positions, unmasks one per step. Decodings: `fair` (Bernoulli), `greedy`
   (threshold at 0.5, pre-scheduled random order), `verygreedy` (dynamically picks the
   still-masked position with max |logit|, then thresholds). Records per-step
   (frac_masked, frac_correct, frac_errors).
3. **`mask_and_sample_oneshot`** (`diffusion.py:300-379`) — one forward pass on the masked
   input, then reveals all masked positions simultaneously (fair or thresholded).
4. **U-turn experiments** (`test_step` `diffusion.py:119-126`): mask a fraction `t0` of a
   training datum, reconstruct with `mask_and_sample` (default greedy), measure retrieval
   overlap. The driver script `experiments-analysis/run_uturn_experiments.py` that invoked
   this on batches of legacy-CLI checkpoints was retired on `guthlac`; the reused mechanism
   (`test_step`) is unaffected — see `docs/archive/HISTORICAL_NOTEBOOKS_ARCHIVE.md`.

All samplers are strictly monotone absorbing-state unmaskers: only positions still equal to
the mask token are candidates, and written tokens are final.

**Objective/sampler relationship.** Training fits all single-site masked conditionals
simultaneously and independently (the notes note the model "can be factorized over output
positions", notes:173). Sequential sampling composes these conditionals one site at a time.
Nothing in the repository establishes that these independently trained conditionals are
mutually consistent with a single joint law, so the sampler must not be described as exact
ancestral sampling; the terminal law is sampler-indexed (`P_{θ,A,k}`). The notes acknowledge
neglected correlations only in the mean-field ODE section (notes:702), and only for the
accuracy observable.

## Training loop and load

- Lightning `MaskedDiffusion` module, AdamW, `train.py` argparse CLI. `--alpha` = `M/L`
  (legacy convention; `M = round(alpha·L)`, `train.py:31`) — see `docs/NOTATION.md`.
- `--seed` defaults to −1 = unseeded. Logs to `logs/{model}_{dataset}_L{L}_alpha{alpha}_...`.

## Known upstream mismatches

See `docs/UPSTREAM_DISCREPANCIES.md` for the full catalogue (runtime `1/√L`, mask encoding,
`alpha` conventions, `sign(0)`, hyphen/underscore model strings, stray `turtle` import,
TensorBackbone's undocumented time conditioning, V=0 status).
