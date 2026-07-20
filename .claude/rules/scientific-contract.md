# Scientific contract rules

These rules apply to all scientific code, notebooks, notes, and documentation in this
repository. Authoritative definitions live in `docs/RESEARCH_SPEC.md` and `docs/NOTATION.md`.

## Notation

- Hidden-manifold quantities: `D` latent dim, `N` visible dim, `M` training-set size,
  `γ = N/D`, `α = M/D`, `M/N = α/γ`.
- New code and configuration must use `latent_dim`, `visible_dim`, `train_size`,
  `aspect_ratio` (N/D), `sample_ratio` (M/D), `visible_load` (M/N, derived metadata only).
- Never use a bare `alpha` or `gamma` identifier in new Python interfaces. The legacy CLI
  `train.py --alpha` means **M/N** (M/L) and must not be reinterpreted; convert explicitly:
  `legacy alpha = sample_ratio / aspect_ratio`.

## Finite-F target and quenched disorder

- Within one repeat: sample the teacher `F` once, hold it fixed, and generate training,
  validation, and fresh evaluation samples with independent latents `z` through the same `F`.
- The finite-`F` law `P_F` is distinct from the empirical training distribution and from the
  disorder average `E_F P_F`. Always say which one a statement is about.
- Do not average over `F` implicitly; disorder averaging happens only across explicit repeats.

## Sampler-indexed laws

- The learned generative object is a sampler-indexed terminal law `P_{θ,A,k}` (algorithm `A`,
  `k` tokens resolved per step). Never write a bare "model distribution" for generated samples.
- Do not call any sampler here "exact ancestral sampling" — compatibility of the independently
  trained masked conditionals with a single coherent joint law has not been established.
- Do not collapse the distinct samplers (fair sequential, multi-token, greedy, verygreedy,
  one-shot) into one generic "reverse process".

## Claim discipline

- A decreasing MMD estimate supports "approaches the finite-F target under this diagnostic"
  or "consistent with improved distributional agreement" — never "the model learns the
  distribution".
- Label every claim as theorem, derivation, RS/ansatz result, conjecture, or empirical
  (finite-dimensional) observation. The "persistent gap above the noise floor" is an open
  empirical observation, not a capacity theorem.

## No silent symmetry assumptions

- The uniform-data argument that mask weights `V` (or biases) vanish does **not** transfer to
  a fixed hidden-manifold teacher `F`: fixed `F` breaks coordinate exchangeability. Treat
  `V ≡ 0` under hidden-manifold data as an experimental restriction (and a candidate cause of
  residual MMD gap), not a derived property, unless a proof is added to the repository.
