# MMD distribution-distance baseline

Companion summary for [`analysis_mmd_distribution_distance_corrected.ipynb`](analysis_mmd_distribution_distance_corrected.ipynb).
This is an **empirical diagnostic / baseline experiment**, not a proof of any asymptotic
statement.

## Scientific question

For a fixed random-feature matrix $F$, do samples from the trained shallow masked diffusion
sampler $P_{\theta,k}$ approach the true finite-$F$ hidden-manifold distribution $P_F$ as the
sample load $\alpha=M/D$ increases?

## Target distribution

$$
z\sim\mathcal N(0,I_D),\qquad x=\operatorname{sign}(Fz)\in\{-1,+1\}^N,
$$

with $F\in\mathbb R^{N\times D}$ fixed per run, $\gamma=N/D$, and $\alpha=M/D$. For a given
$(D,\gamma,\text{repeat})$, the *same* sampled $F$ is used to generate the training set, the
held-out test set, and every fresh "true" sample drawn during evaluation — the target is the
finite-$F$ law $P_F$, not an average over feature matrices.

## Estimator

Mixture-kernel Maximum Mean Discrepancy (MMD) with exponential normalized-Hamming kernels
$k_\lambda(x,y)=\exp(-\lambda\,d_H(x,y)/N)$ at $\lambda\in\{0.5,1,2,4,8,16\}$. The
mixture-kernel MMD² is a nonnegative-weighted sum of the per-$\lambda$ MMD² values (uniform
weights by default) — this is a mathematically valid PD kernel combination, not an
approximation. Both a biased V-statistic (used for plotting, always $\ge 0$ after clipping) and
an unbiased U-statistic (used for statistics, can be slightly negative at finite sample size)
are computed. The unbiased two-sample cross term always uses all $mn$ pairs — no diagonal is
excluded there; only the unbiased self-terms exclude their diagonal.

## Primary experiment

Fixed $(D,\gamma)$, varying $\alpha=M/D$ (Experiment A), under the `repo_uniform_t` training
objective (the repo's canonical $t\sim U(0,1)$ masked-diffusion objective) and the default
sequential sampler ($k=1$). `exact_fixed_context` and `bernoulli_fixed_context` are kept as
fixed-masking-level **ablations**, not the primary objective.

## Baselines computed for every row

- **True-vs-true**: MMD between two independent fresh batches from $P_F$ — the finite-sample
  noise floor.
- **Train-vs-true**: MMD between the empirical training set and fresh $P_F$ samples — how good
  an approximation to $P_F$ the training set itself is.
- **Model-vs-train**: MMD between generated samples and the training set — a
  proximity-to-training diagnostic, not a memorization proof by itself.
- **Nearest-training overlap**: largest overlap between generated samples and the training set,
  compared against the same quantity for *fresh* true samples vs. the training set (so that
  overlap increasing with $\alpha$ purely from more training data is not mistaken for
  memorization).

## Success criterion

> Model-vs-True MMD should decrease with $\alpha$ and approach the True-vs-True baseline. A
> plateau above the baseline indicates possible model/sampler/optimization/finite-size
> limitations — **not** proof of an asymptotic capacity threshold.

## What `FAST_DEBUG=True` proves and does not prove

`FAST_DEBUG=True, FINAL_RUN=False` (the committed defaults) runs a tiny CPU-friendly sweep
(hundreds of MMD samples, ~100-300 training steps, 1 repeat). It proves the pipeline runs
end-to-end without exceptions, the estimator sanity checks pass (biased self-MMD ≈ 0,
independent true-vs-true MMD finite, exact-$K$ mask/loss-scale checks pass), and the plots and
result tables are produced correctly. It does **not** prove anything about the scientific
question above — sample counts and training budgets are far too small for quantitative claims.

## What a full run requires

Set `FAST_DEBUG=False` (10k MMD samples, up to 3000 training steps scaled by $N$) and, for
final figures, `FINAL_RUN=True` (3 repeats over feature matrices/datasets/initialization/
sampling noise, so mean ± SEM curves are meaningful). A full run is substantially more
expensive and should use a GPU if available (`train_device`/`mmd_device` auto-detect CUDA).

## Reproducing figures

Notebook cell outputs are stripped before commit to keep diffs reviewable. To regenerate
figures/results locally:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace \
    experiments-analysis/analysis_mmd_distribution_distance_corrected.ipynb
```

This writes `results/results_mmd_distribution_distance_corrected.csv`,
`results/results_mmd_time_sliced.csv`, and PNGs under `figures/`. Checkpoint CSVs
(`results/results_mmd_distribution_distance_{debug,full}_checkpoint.csv`) are resume artifacts
and are not committed.

## Limitations

- **Objective/sampler mismatch**: `exact_fixed_context`/`bernoulli_fixed_context` train the
  model at a single masking level; the sequential sampler visits many masking levels during
  generation, so results under these ablations should be read as an objective/sampler-mismatch
  check, not as a failure of the finite-$K$ conditional objective itself.
- **Experiment B (dimension scaling)** changes $D$, hence $N$, $M$, parameter count, and
  optimization difficulty simultaneously — it is a finite-size / optimization-budget stress
  test, not a clean thermodynamic-limit result. The per-config training-convergence summary
  (`initial_100_train_loss_mean`, `final_100_train_loss_mean`, `loss_decrease_ratio`) is
  included specifically so a $D$-sweep is not over-interpreted without checking whether larger
  models were simply undertrained.
- **Linear backbone**: a time-independent linear visible-coordinate denoiser may not represent
  the full family of masked conditional distributions needed to reproduce $P_F$ exactly.
- **Scalar summary**: the mixture-kernel MMD can hide failures at individual kernel scales;
  always check the per-$\lambda$ curves.
- This baseline is **not sufficient to identify the asymptotic capacity threshold** of the
  shallow model — it is a finite-$(D,\alpha)$ empirical check.
