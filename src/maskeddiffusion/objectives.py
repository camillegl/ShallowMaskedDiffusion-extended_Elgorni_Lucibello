"""Named masked-BCE objective estimators.

The canonical continuous-time objective (docs/RESEARCH_SPEC.md,
docs/ORIGINAL_ARCHITECTURE.md):

    L = (1/(N*B)) * sum_batch sum_{i masked} (1/t) * BCE(h_i, (x_i+1)/2)

with t ~ U(0,1) per sequence and Bernoulli(t) masks. The 1/t factor is applied
exactly; it is never silently replaced by N/num_masked. Numerical behavior
near t = 0 is governed by the sampling distribution: an optional
`min_time` shifts sampling to U(min_time, 1) and must be recorded in the run
config — it changes the estimator's time mixture and is off (0.0) by default.

Estimators with different mask laws have different explicit names.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .masking import (
    MaskedBatch,
    bernoulli_mask,
    continuous_time_mask,
    exact_visible_count_mask,
)
from .models import LinearMaskedScore


def _safe_inverse_time(t: torch.Tensor) -> torch.Tensor:
    """1/t, with t == 0 mapped exactly to 0 instead of inf.

    `t ~ U(min_time, 1)` can draw exactly 0.0 even when min_time == 0.0 (its
    documented default) — torch.rand's range is [0, 1). When it does, every
    coordinate in that row is masked with probability 0 (bernoulli_mask), so
    `batch.is_masked` is all-False for that row; the intended contribution
    of that row to the loss is exactly 0. But `weighted = losses * (1/t) *
    is_masked` computes `finite * inf * 0`, which is NaN under IEEE-754, not
    0 — corrupting the whole batch's loss and gradient (docs/UPSTREAM_DISCREPANCIES.md
    D16). This maps exactly t == 0 to a weight of 0 (its all-False mask row
    then contributes an exact 0 regardless of the multiplier) while leaving
    1/t untouched — to full float precision — for every t > 0, including
    values arbitrarily close to 0; a clamp_min(eps) would instead silently
    distort the estimator for any positive draw below eps.
    """
    is_zero = t == 0
    safe_t = torch.where(is_zero, torch.ones_like(t), t)
    return torch.where(is_zero, torch.zeros_like(t), 1.0 / safe_t)


@dataclass
class ObjectiveResult:
    per_example: torch.Tensor  # (B,) unreduced data-loss per example
    data_loss: torch.Tensor  # scalar batch reduction (mean convention below)
    regularization: torch.Tensor  # scalar
    total: torch.Tensor  # data_loss + regularization
    masked_token_count: int
    diagnostics: dict[str, float]


def l2_regularization(model: LinearMaskedScore, l2reg: float, train_size: int) -> torch.Tensor:
    """0.5 * l2reg / train_size * sum ||p||^2 over trainable regularized params.

    Matches the legacy replica-λ correspondence (l2coeff = 0.5·λ/(L·α_legacy)
    = 0.5·λ/M), restricted to trainable parameters (fixes D6 in the active
    path; the legacy code also penalized frozen parameters).
    """
    params = model.regularized_parameters()
    device = next(model.parameters()).device
    if l2reg == 0.0 or not params:
        return torch.zeros((), device=device)
    sq = torch.zeros((), device=device)
    for p in params:
        sq = sq + (p**2).sum()
    return 0.5 * l2reg / train_size * sq


def _masked_bce(
    model: LinearMaskedScore, batch: MaskedBatch, weight_per_position: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Shared core: weighted BCE over masked positions, /(N*B) normalization."""
    b, n = batch.values.shape
    logits = model(batch.values, batch.is_masked)
    targets = (batch.values + 1.0) / 2.0
    losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    weighted = losses * weight_per_position * batch.is_masked.to(losses.dtype)
    per_example = weighted.sum(dim=1) / n
    data_loss = per_example.sum() / b
    masked_count = int(batch.is_masked.sum().item())
    return per_example, data_loss, masked_count


def continuous_time_masked_bce(
    model: LinearMaskedScore,
    x: torch.Tensor,
    generator: torch.Generator,
    *,
    l2reg: float = 0.0,
    train_size: int = 1,
    min_time: float = 0.0,
) -> ObjectiveResult:
    """Canonical objective: t ~ U(min_time, 1) per sequence, Bernoulli(t) masks,
    per-position weight 1/t."""
    batch_size = x.shape[0]
    t = torch.rand(batch_size, generator=generator, dtype=x.dtype, device=x.device)
    if min_time > 0.0:
        t = min_time + (1.0 - min_time) * t
    batch = bernoulli_mask(x, t, generator)
    weight = _safe_inverse_time(t).unsqueeze(1).expand_as(x)
    return _finalize(
        model, batch, weight, l2reg, train_size, extra={"mean_time": float(t.mean().item())}
    )


def continuous_time_masked_bce_from_batch(
    model: LinearMaskedScore,
    batch: MaskedBatch,
    *,
    l2reg: float = 0.0,
    train_size: int = 1,
) -> ObjectiveResult:
    """Same estimator evaluated on a pre-built MaskedBatch (deterministic tests)."""
    t = batch.mask_probability
    zero_rows_with_mask = (t == 0) & batch.is_masked.any(dim=1)
    if bool(zero_rows_with_mask.any()):
        raise ValueError(
            "batch has masked coordinates in a row with mask_probability t == 0 — "
            "inconsistent with the estimator, which requires t == 0 rows to be "
            "entirely unmasked (weight 1/t is defined as exactly 0 there)"
        )
    weight = _safe_inverse_time(t).unsqueeze(1).expand_as(batch.values)
    return _finalize(
        model, batch, weight, l2reg, train_size, extra={"mean_time": float(t.mean().item())}
    )


def fixed_mask_probability_bce(
    model: LinearMaskedScore,
    x: torch.Tensor,
    mask_probability: float,
    generator: torch.Generator,
    *,
    l2reg: float = 0.0,
    train_size: int = 1,
    time_weighting: bool = False,
) -> ObjectiveResult:
    """Fixed-t ablation: every sequence masked at the same probability.

    By default unweighted (weight 1); `time_weighting=True` applies the 1/t
    factor of the continuous-time estimator at this fixed t.
    """
    if not 0.0 < mask_probability <= 1.0:
        raise ValueError(f"mask_probability must be in (0, 1], got {mask_probability}")
    batch_size = x.shape[0]
    t = torch.full((batch_size,), mask_probability, dtype=x.dtype, device=x.device)
    batch = bernoulli_mask(x, t, generator)
    w = 1.0 / mask_probability if time_weighting else 1.0
    weight = torch.full_like(x, w)
    return _finalize(
        model, batch, weight, l2reg, train_size, extra={"mask_probability": mask_probability}
    )


def exact_visible_count_bce(
    model: LinearMaskedScore,
    x: torch.Tensor,
    visible_count: int,
    generator: torch.Generator,
    *,
    l2reg: float = 0.0,
    train_size: int = 1,
) -> ObjectiveResult:
    """Exact-count ablation: all but `visible_count` coordinates masked, weight 1."""
    batch = exact_visible_count_mask(x, visible_count, generator)
    weight = torch.ones_like(x)
    return _finalize(
        model, batch, weight, l2reg, train_size, extra={"visible_count": float(visible_count)}
    )


def _finalize(
    model: LinearMaskedScore,
    batch: MaskedBatch,
    weight: torch.Tensor,
    l2reg: float,
    train_size: int,
    extra: dict[str, float],
) -> ObjectiveResult:
    per_example, data_loss, masked_count = _masked_bce(model, batch, weight)
    reg = l2_regularization(model, l2reg, train_size)
    diagnostics = {"masked_fraction": masked_count / batch.values.numel(), **extra}
    return ObjectiveResult(
        per_example=per_example,
        data_loss=data_loss,
        regularization=reg,
        total=data_loss + reg,
        masked_token_count=masked_count,
        diagnostics=diagnostics,
    )


__all__ = [
    "ObjectiveResult",
    "continuous_time_masked_bce",
    "continuous_time_masked_bce_from_batch",
    "fixed_mask_probability_bce",
    "exact_visible_count_bce",
    "l2_regularization",
    "continuous_time_mask",
]
