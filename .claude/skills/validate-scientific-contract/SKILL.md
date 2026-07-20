---
name: validate-scientific-contract
description: Validate a config, script, notebook, or proposed change against the scientific contract in docs/RESEARCH_SPEC.md and docs/NOTATION.md — dimensions, ratio names, target-law and sampler framing. Use when adding or reviewing hidden-manifold experiment code or docs.
---

# Validate scientific contract

Validate the given target (file, diff, config, or notebook) against
`docs/RESEARCH_SPEC.md` and `docs/NOTATION.md`. Do not edit anything; return a verdict.

Checks, in order:

1. **Naming.** New code/config must use `latent_dim`, `visible_dim`, `train_size`,
   `aspect_ratio`, `sample_ratio`, `visible_load`. Flag any new bare `alpha`/`gamma`
   identifier. Legacy `train.py --alpha` (= M/N) may be *consumed* but only via an explicit
   conversion `sample_ratio / aspect_ratio`.
2. **Dimensions and ratios.** Verify consistency: `visible_dim ≈ round(aspect_ratio ·
   latent_dim)`, `train_size ≈ round(sample_ratio · latent_dim)`, `visible_load =
   train_size / visible_dim` (derived metadata only, never a primary control).
3. **Quenched disorder.** Within a repeat, `F` sampled once and shared by train/val/fresh
   evaluation samples; fresh latents `z` per sample; `sign(0)` handled explicitly (→ +1).
4. **Target law.** Any distributional comparison must name its target: finite-F law `P_F`
   (fresh z, same F), empirical train set, or disorder average — flag ambiguity.
5. **Sampler indexing.** Generated-sample statements must identify the sampler and `k`
   (terminal law `P_{θ,A,k}`); flag "ancestral sampling" or sampler-free "model
   distribution" language.
6. **Claim vocabulary.** Flag "learns the distribution" claims based on MMD; require the
   hedged vocabulary from `docs/RESEARCH_SPEC.md`.

Verdict: **PASS** (all checks satisfied, cite what was checked), **FAIL** (violations, each
with file:line evidence and the exact rule violated), or **BLOCKED** (needed information
missing — say precisely what).
