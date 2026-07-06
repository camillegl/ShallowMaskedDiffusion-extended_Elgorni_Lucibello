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

## Current committed results

The notebook is currently committed with **`FINAL_RUN=True`** and its outputs baked in
(figures, tables, and the execution-status cell are all visible without re-running
anything). This is the full baseline sweep: Experiments A+B+C
($D\in\{40,80,160\}$, $\gamma\in\{1,2,4\}$, $\alpha\in\{0.5,\dots,100\}$, 34 unique
configs), 3 repeats, 10 000 MMD samples per comparison — 102 result rows and 510
time-sliced diagnostic rows, `run_mode=full_final` throughout. On Experiment A
($D=80,\gamma=2$), Model-vs-True MMD decreases from 0.042 at $\alpha=0.5$ to 0.013 at
$\alpha=100$, closing in on the 0.012 True-vs-True noise floor. This full run took ~72
minutes on an M3 MacBook Air CPU (not the 30-minute preview budget below); the checkpoint
in `results/results_mmd_distribution_distance_full_checkpoint.csv` makes it resumable.

To reproduce quickly for local iteration instead, flip the flags in the configuration
cell to the preview mode described next (`MACBOOK_30MIN_RUN=True`, `FINAL_RUN=False`) —
that finishes in about a minute but is explicitly **not** the quantitative baseline shown
above.

## Run modes

Three run modes, selected in the configuration cell. Every result row carries a
`run_mode` column (`fast_debug` / `macbook_30min` / `full_single_repeat` / `full_final`)
plus per-row timing columns (`train_time_sec`, `loss_eval_time_sec`, `sample_time_sec`,
`mmd_time_sec`, `row_time_sec`), so results from different modes are never confusable and
the laptop bottleneck is visible directly in the CSV.

### What `FAST_DEBUG=True` proves and does not prove

`FAST_DEBUG=True` runs a tiny CPU-friendly sweep (hundreds of MMD samples, ~100-300
training steps, 1 repeat) and overrides `MACBOOK_30MIN_RUN`. It proves the pipeline runs
end-to-end without exceptions, the estimator sanity checks pass (biased self-MMD ≈ 0,
independent true-vs-true MMD finite, exact-$K$ mask/loss-scale checks pass), and the plots and
result tables are produced correctly. It does **not** prove anything about the scientific
question above — sample counts and training budgets are far too small for quantitative claims.

### Local MacBook preview mode

`MACBOOK_30MIN_RUN=True` is a reduced local preview mode intended for Apple Silicon
laptops (not the currently-committed configuration — see above). It uses Experiment A only ($D=80$, $\gamma=2$, the full
$\alpha \in \{0.5, 1, 2, 5, 10, 20, 50, 100\}$ grid), one repeat, one sampler value
($k=1$), 1500 MMD samples instead of 10000, a smaller capped training budget (≤1000 steps
per model), and disables the gamma/dimension sweeps and the correlation diagnostic. It is
designed to check the pipeline and obtain a rough qualitative alpha-sweep in laptop-scale
time — measured ~2 minutes end-to-end on an M3 MacBook Air (8 GB), far under the 30-minute
budget it was sized for. **It is not the final
quantitative baseline**: 1500 MMD samples raise the finite-sample noise floor, and one
repeat means no error bars.

Device selection is CUDA > MPS > CPU via `get_preferred_device`, controlled by
`PREFER_MPS_FOR_TRAINING` and `USE_MPS_FOR_MMD`. Both default to **False**: benchmarked on
an M3 MacBook Air (torch 2.12), the linear backbones here are so small that MPS
dispatch/sync overhead makes training ~65× and $k=1$ sequential sampling ~50× *slower* on
MPS than on CPU, and the chunked MMD kernel sums (which sync per chunk) are ~10× slower.
Enable the flags only if timing on your machine shows MPS is faster (e.g. for much larger
backbones). CUDA, when available, is always used automatically.

### What a full run requires

Set `FAST_DEBUG=False, MACBOOK_30MIN_RUN=False` (10k MMD samples, up to 3000 training steps
scaled by $N$) and, for final figures, `FINAL_RUN=True` (3 repeats over feature
matrices/datasets/initialization/sampling noise, so mean ± SEM curves are meaningful). A
full run is substantially more expensive and should run on a GPU/cluster machine
(`train_device`/`mmd_device` auto-detect CUDA); it is not intended for a MacBook Air.

## Reproducing figures

The committed notebook currently has its outputs baked in (see "Current committed
results" above), rather than stripped, so the results are visible without re-running
anything. To regenerate figures/results locally — e.g. after changing a run-mode flag —
run:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace \
    experiments-analysis/analysis_mmd_distribution_distance_corrected.ipynb
```

This writes `results/results_mmd_distribution_distance_corrected.csv`,
`results/results_mmd_time_sliced.csv`, and PNGs under `figures/`. Checkpoint CSVs
(`results/results_mmd_distribution_distance_{debug,macbook,full}_checkpoint.csv`) are resume
artifacts and are not committed. Note: the time-sliced diagnostic is *not* itself
checkpointed — if the main sweep fully resumes from a cached checkpoint, no new
time-sliced rows are generated that run, and the existing CSV on disk is kept as-is.

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
