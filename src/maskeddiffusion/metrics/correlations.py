"""Empirical spin statistics vs the analytic finite-F arcsine correlation.

These are diagnostics; they do not fully characterize the distribution.
Complexity: pair correlations are O(S·N²) time and O(N²) memory for S samples.
"""

from __future__ import annotations

import torch


def empirical_mean_spin(samples: torch.Tensor) -> torch.Tensor:
    """(N,) mean spin over samples (S, N)."""
    return samples.mean(dim=0)


def empirical_pair_correlation(samples: torch.Tensor) -> torch.Tensor:
    """(N, N) empirical E[x_i x_j] over samples (S, N). O(S·N²)."""
    s = samples.shape[0]
    return samples.t() @ samples / s


def correlation_error(
    empirical: torch.Tensor, analytic: torch.Tensor, *, exclude_diagonal: bool = True
) -> dict[str, float]:
    """Error statistics between empirical and analytic C_ij (off-diagonal by default)."""
    diff = empirical - analytic
    if exclude_diagonal:
        n = diff.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool, device=diff.device)
        diff = diff[mask]
    return {
        "max_abs_error": float(diff.abs().max().item()),
        "rms_error": float(diff.pow(2).mean().sqrt().item()),
        "mean_error": float(diff.mean().item()),
    }
