"""Overlap diagnostics: memorization-sensitive nearest-training statistics.

Normalized overlap q(x, y) = x·y / N for ±1 spins. Nearest-training overlap is
computed in chunks: O(A·B·N) time, O(chunk·B) memory for sample sets of sizes
A and B. These metrics are diagnostics only; they do not fully characterize
the distribution.
"""

from __future__ import annotations

import torch


def normalized_overlap(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Pairwise overlaps x·y/N for row-aligned batches (B, N) -> (B,)."""
    if x.shape != y.shape:
        raise ValueError("shapes must match for row-aligned overlap")
    return (x * y).sum(dim=1) / x.shape[1]


def nearest_training_overlap(
    samples: torch.Tensor, train_set: torch.Tensor, *, chunk_size: int = 1024
) -> torch.Tensor:
    """(S,) max over training rows of q(sample, train_row); chunked over samples."""
    n = samples.shape[1]
    if train_set.shape[1] != n:
        raise ValueError("visible dimensions differ")
    out = torch.empty(samples.shape[0], dtype=samples.dtype)
    for start in range(0, samples.shape[0], chunk_size):
        block = samples[start : start + chunk_size]
        q = block @ train_set.t() / n  # (chunk, M)
        out[start : start + chunk_size] = q.max(dim=1).values
    return out


def nearest_training_excess(
    model_samples: torch.Tensor,
    true_samples: torch.Tensor,
    train_set: torch.Tensor,
    *,
    chunk_size: int = 1024,
) -> dict[str, float]:
    """Compare nearest-training overlap of model samples against fresh true
    samples — the excess is a memorization signal (a diagnostic, not proof)."""
    q_model = nearest_training_overlap(model_samples, train_set, chunk_size=chunk_size)
    q_true = nearest_training_overlap(true_samples, train_set, chunk_size=chunk_size)
    return {
        "model_mean_nearest_overlap": float(q_model.mean().item()),
        "true_mean_nearest_overlap": float(q_true.mean().item()),
        "excess": float((q_model.mean() - q_true.mean()).item()),
    }
