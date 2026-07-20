"""Mask representation and forward corruption.

Clean spin values and masked status are separate tensors; spin value 0 is
never used as a mask sentinel inside core logic (the legacy in-band `0` token
is confined to the legacy modules — discrepancy D4).

The canonical continuous-time corruption (docs/RESEARCH_SPEC.md /
docs/ORIGINAL_ARCHITECTURE.md): per sequence t ~ U(0,1), then each coordinate
is masked independently with probability t. Every masked coordinate is a
prediction target; the target coordinate is not separately conditioned to be
masked beyond this Bernoulli draw.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MaskedBatch:
    """values: clean spins with masked entries zeroed for model input is NOT
    stored here; `values` holds the clean ±1 spins, `is_masked` the mask.
    Model input encoding is the model's concern (see models.py).
    """

    values: torch.Tensor  # (B, N) clean spins in {-1, +1}
    is_masked: torch.Tensor  # (B, N) bool
    mask_probability: torch.Tensor  # (B,) the t used per sequence (or realized fraction)
    clean_targets: torch.Tensor | None = None  # alias of values when training

    def masked_count(self) -> torch.Tensor:
        return self.is_masked.sum(dim=1)

    def visible_count(self) -> torch.Tensor:
        return (~self.is_masked).sum(dim=1)


def bernoulli_mask(
    x: torch.Tensor, mask_probability: torch.Tensor, generator: torch.Generator
) -> MaskedBatch:
    """Mask each coordinate independently with its sequence's probability t."""
    if mask_probability.ndim != 1 or mask_probability.shape[0] != x.shape[0]:
        raise ValueError("mask_probability must be shape (B,)")
    t = mask_probability.unsqueeze(1).expand_as(x)
    u = torch.rand(x.shape, generator=generator, dtype=x.dtype, device=x.device)
    is_masked = u < t
    return MaskedBatch(
        values=x, is_masked=is_masked, mask_probability=mask_probability, clean_targets=x
    )


def continuous_time_mask(x: torch.Tensor, generator: torch.Generator) -> MaskedBatch:
    """Canonical corruption: t ~ U(0,1) per sequence, Bernoulli(t) per coordinate.

    Uses a single generator for both draws (t first, then uniforms), so replay
    with the same generator state is deterministic.
    """
    t = torch.rand(x.shape[0], generator=generator, dtype=x.dtype, device=x.device)
    return bernoulli_mask(x, t, generator)


def exact_visible_count_mask(
    x: torch.Tensor, visible_count: int, generator: torch.Generator
) -> MaskedBatch:
    """Mask all but exactly `visible_count` uniformly chosen coordinates per row."""
    batch, n = x.shape
    if not 0 <= visible_count <= n:
        raise ValueError(f"visible_count {visible_count} not in [0, {n}]")
    scores = torch.rand((batch, n), generator=generator, device=x.device)
    order = scores.argsort(dim=1)
    is_masked = torch.ones((batch, n), dtype=torch.bool, device=x.device)
    keep = order[:, :visible_count]
    is_masked.scatter_(1, keep, False)
    frac = torch.full((batch,), (n - visible_count) / n, dtype=x.dtype, device=x.device)
    return MaskedBatch(values=x, is_masked=is_masked, mask_probability=frac, clean_targets=x)


def mask_from_uniforms(
    x: torch.Tensor, mask_probability: torch.Tensor, uniforms: torch.Tensor
) -> MaskedBatch:
    """Deterministic masking from explicit uniforms (for tests/fixtures)."""
    if uniforms.shape != x.shape:
        raise ValueError("uniforms must have the same shape as x")
    t = mask_probability.unsqueeze(1).expand_as(x)
    is_masked = uniforms < t
    return MaskedBatch(
        values=x, is_masked=is_masked, mask_probability=mask_probability, clean_targets=x
    )
