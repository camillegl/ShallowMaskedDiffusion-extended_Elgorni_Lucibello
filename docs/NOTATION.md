# Notation and naming (canonical)

This document is the single authority for symbols and code names. Where legacy code
disagrees, the legacy meaning is documented here and must be converted explicitly, never
reinterpreted.

## Symbols (hidden-manifold extension)

| Symbol | Meaning | Notes |
|---|---|---|
| `D` | latent dimension | teacher input dimension |
| `N` | visible dimension | data dimension; legacy code calls this `L` |
| `M` | training-set size | |
| `γ = N/D` | aspect ratio | fixed geometry of the teacher |
| `α = M/D` | sample ratio | primary training-load parameter of the extension |
| `M/N = α/γ` | visible load | derived metadata only |
| `F ∈ R^{N×D}` | teacher matrix, `F_ia ~ N(0, 1/D)` | quenched within a repeat |
| `z ∈ R^D` | latent vector, `z ~ N(0, I_D)` | fresh per sample |
| `x = sign(Fz) ∈ {−1,+1}^N` | visible sample | convention `sign(0) := +1` |
| `P_F` | finite-F data law | distinct from empirical train set and from `E_F P_F` |
| `P_{θ,A,k}` | sampler-indexed terminal law of the trained model | `A` = algorithm, `k` = tokens resolved per step |
| `t ∈ [0,1]` | masking time / mask density | |
| `m ∈ {0,1}^N` | mask indicator, 1 = masked | code derives it as `x_t == 0` |
| `W, V, b` | linear-score weights: unmasked-token, mask-channel, bias | see `docs/ORIGINAL_ARCHITECTURE.md` |
| `λ` | (i) L2 strength `l2reg`; (ii) MMD kernel scale in the notebooks | context-disambiguated; new code must use `l2reg` / `kernel_scale` |

## Code names (mandatory in new interfaces)

| Code name | Symbol | Definition |
|---|---|---|
| `latent_dim` | `D` | int |
| `visible_dim` | `N` | int |
| `train_size` | `M` | int |
| `aspect_ratio` | `γ` | `N/D` |
| `sample_ratio` | `α` | `M/D` |
| `visible_load` | `M/N` | `sample_ratio / aspect_ratio`; **derived metadata only**, never a primary control parameter unless explicitly justified |

Bare `alpha` and `gamma` identifiers are **banned** in new active Python code and
configuration. They may appear only in mathematical documentation and in untouched legacy
code.

## Rounding rules

Integer dimensions are obtained by rounding, in this order:

- `visible_dim = round(aspect_ratio * latent_dim)`
- `train_size = round(sample_ratio * latent_dim)`
- `visible_load = train_size / visible_dim` (computed from the realized integers, so it may
  differ slightly from `sample_ratio/aspect_ratio`; record the realized value).

Legacy `train.py` uses `M = round(alpha * L)` (`train.py:31`) — same rounding rule, but with
its own `alpha` convention (below).

## Deprecated / legacy names and the alpha conflict

`alpha` currently has **four different meanings** in this repository (the notes use it both
as the Rényi order and as a disorder-replica index; the table collapses those into one row):

| Location | Meaning | Canonical equivalent |
|---|---|---|
| `train.py --alpha`, `config.alpha`, log-dir names, `l2coeff` scaling (`diffusion.py:36`) | `M/L = M/N` | `visible_load = sample_ratio / aspect_ratio` |
| Hidden-manifold notebooks (`analysis_mmd_distribution_distance_corrected.ipynb`) and this contract | `M/D` | `sample_ratio` |
| `notes/notes_hiddenmanifold.typ` | Rényi order `α = n+1` (and separately a replica index) | unrelated to data load |

Rules:

- **Never silently reinterpret** a stored value or CLI argument. A value that flowed through
  `train.py --alpha` or `config.alpha` is `M/N`; a notebook-axis `alpha` is `M/D`.
- Conversion: `legacy_alpha (M/N) = sample_ratio / aspect_ratio`.
- Legacy `L` means `N` (`visible_dim`). `RandomFeaturesDataset` uses `n_visible = N` and
  `n_hidden = D` (`datasets.py:39-53`) — note `n_hidden` there is the *latent* dimension,
  not a network width.

## Examples

With `latent_dim D = 100`, `aspect_ratio γ = 5`, `sample_ratio α = 8`:

- `visible_dim N = 500`, `train_size M = 800`, `visible_load M/N = 1.6`.
- To reproduce this run through the legacy CLI: `--L 500 --alpha 1.6` (legacy alpha = M/N).

## Validation invariants

Any config or artifact using these names must satisfy:

1. `visible_dim == round(aspect_ratio * latent_dim)`
2. `train_size == round(sample_ratio * latent_dim)`
3. `abs(visible_load - train_size/visible_dim) < 1e-12` (realized value)
4. If a legacy `alpha` is present alongside contract names:
   `abs(alpha_legacy - train_size/visible_dim) < 1e-9`
5. All samples in `{−1,+1}` exactly — no zeros (`sign(0) := +1` applied).
