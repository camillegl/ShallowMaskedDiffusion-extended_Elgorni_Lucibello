"""MMD with the exponential normalized-Hamming kernel — the exact estimator
of the corrected final-run notebook
(experiments-analysis/analysis_mmd_distribution_distance_corrected.ipynb).

Kernel: k_λ(x, y) = exp(-λ (1 - q)/2), q = x·y/N (normalized overlap).

- biased V-statistic:  S_XX/m² + S_YY/n² − 2 S_XY/(mn)  (all pairs)
- raw unbiased U-statistic: off-diagonal self-terms, all-pairs cross term;
  may legitimately be negative and is NEVER clipped here.
- `sqrt_clipped_mmd` is a clearly named visualization transform:
  sqrt(max(mmd2, 0)).
- mixture kernel: nonnegative-weighted sum over λ (uniform by default) —
  positive-definite as a sum of PD kernels.
- chunked computation: never materializes more than chunk_size² kernel
  entries at once (the final run used 100k samples; a full matrix would be
  100k×100k).

An MMD decrease supports "approaches the finite-F target under this
diagnostic" — it is not evidence that the model learns the distribution
(docs/RESEARCH_SPEC.md).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch


def kernel_sums_exponential_hamming(
    x: torch.Tensor,
    y: torch.Tensor,
    lambdas: Sequence[float],
    chunk_size: int = 1024,
) -> dict[float, float]:
    """Sum of k_λ over all pairs in X × Y (diagonal included when X is Y)."""
    n_dim = x.shape[1]
    if y.shape[1] != n_dim:
        raise ValueError("visible dimensions differ")
    sums = {lam: 0.0 for lam in lambdas}
    xf = x.to(torch.float32)
    yf = y.to(torch.float32)
    for i in range(0, xf.shape[0], chunk_size):
        xc = xf[i : i + chunk_size]
        for j in range(0, yf.shape[0], chunk_size):
            yc = yf[j : j + chunk_size]
            overlaps = (xc @ yc.t()) / n_dim
            norm_hamming = (1.0 - overlaps) / 2.0
            for lam in lambdas:
                sums[lam] += float(torch.exp(-lam * norm_hamming).sum().item())
    return sums


def normalized_weights(
    lambdas: Sequence[float], weights: Sequence[float] | None
) -> dict[float, float]:
    if weights is None:
        return {lam: 1.0 / len(lambdas) for lam in lambdas}
    if len(weights) != len(lambdas):
        raise ValueError("weights must match lambdas length")
    ws = [float(w) for w in weights]
    if any(w < 0 for w in ws):
        raise ValueError("weights must be nonnegative")
    total = sum(ws)
    if total <= 0:
        raise ValueError("weights must have positive sum")
    return {lam: w / total for lam, w in zip(lambdas, ws, strict=True)}


@dataclass(frozen=True)
class MMDResult:
    biased_mmd2: dict[float, float]  # per-λ V-statistic
    unbiased_mmd2_raw: dict[float, float]  # per-λ U-statistic, unclipped
    mixture_biased_mmd2: float
    mixture_unbiased_mmd2_raw: float
    weights: dict[float, float]


def sqrt_clipped_mmd(mmd2: float) -> float:
    """Visualization transform only: sqrt(max(mmd2, 0))."""
    return math.sqrt(max(mmd2, 0.0))


def compute_mmd(
    x: torch.Tensor,
    y: torch.Tensor,
    lambdas: Sequence[float] = (4.0, 8.0),
    *,
    chunk_size: int = 1024,
    weights: Sequence[float] | None = None,
) -> MMDResult:
    """Biased and raw unbiased MMD² per λ plus mixture summaries — exactly the
    corrected notebook's compute_mmd_biased_unbiased."""
    m, n = x.shape[0], y.shape[0]
    w = normalized_weights(lambdas, weights)

    sum_xx = kernel_sums_exponential_hamming(x, x, lambdas, chunk_size)
    sum_yy = kernel_sums_exponential_hamming(y, y, lambdas, chunk_size)
    sum_xy = kernel_sums_exponential_hamming(x, y, lambdas, chunk_size)
    # k_λ(x, x) = exp(0) = 1, so each self-diagonal sums to the set size
    diag_xx, diag_yy = float(m), float(n)

    biased: dict[float, float] = {}
    unbiased: dict[float, float] = {}
    for lam in lambdas:
        biased[lam] = sum_xx[lam] / (m * m) + sum_yy[lam] / (n * n) - 2.0 * sum_xy[lam] / (m * n)
        xx_u = (sum_xx[lam] - diag_xx) / (m * (m - 1)) if m > 1 else 0.0
        yy_u = (sum_yy[lam] - diag_yy) / (n * (n - 1)) if n > 1 else 0.0
        unbiased[lam] = xx_u + yy_u - 2.0 * sum_xy[lam] / (m * n)

    return MMDResult(
        biased_mmd2=biased,
        unbiased_mmd2_raw=unbiased,
        mixture_biased_mmd2=sum(w[lam] * biased[lam] for lam in lambdas),
        mixture_unbiased_mmd2_raw=sum(w[lam] * unbiased[lam] for lam in lambdas),
        weights=w,
    )


# -- clearly named evaluation comparisons (docs/RESEARCH_SPEC.md targets) ----


def true_vs_true(
    true_a: torch.Tensor,
    true_b: torch.Tensor,
    lambdas: Sequence[float] = (4.0, 8.0),
    **kwargs: object,
) -> MMDResult:
    """Finite-sample noise floor: two independent fresh P_F batches."""
    return compute_mmd(true_a, true_b, lambdas, **kwargs)  # type: ignore[arg-type]


def train_vs_true(
    train_set: torch.Tensor,
    true_fresh: torch.Tensor,
    lambdas: Sequence[float] = (4.0, 8.0),
    **kwargs: object,
) -> MMDResult:
    """Memorization reference: empirical training rows vs fresh P_F samples."""
    return compute_mmd(train_set, true_fresh, lambdas, **kwargs)  # type: ignore[arg-type]


def model_vs_true(
    model_samples: torch.Tensor,
    true_fresh: torch.Tensor,
    lambdas: Sequence[float] = (4.0, 8.0),
    **kwargs: object,
) -> MMDResult:
    """Sampler-indexed terminal-law samples vs fresh P_F samples."""
    return compute_mmd(model_samples, true_fresh, lambdas, **kwargs)  # type: ignore[arg-type]


def model_vs_train(
    model_samples: torch.Tensor,
    train_set: torch.Tensor,
    lambdas: Sequence[float] = (4.0, 8.0),
    **kwargs: object,
) -> MMDResult:
    """Terminal-law samples vs the empirical training set."""
    return compute_mmd(model_samples, train_set, lambdas, **kwargs)  # type: ignore[arg-type]
