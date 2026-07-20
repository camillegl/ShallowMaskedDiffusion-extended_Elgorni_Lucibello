"""Linear masked score model.

Canonical form (docs/RESEARCH_SPEC.md, docs/ORIGINAL_ARCHITECTURE.md):

    h_i(x, m) = (1/sqrt(N)) * sum_j [ W_ij (1 - m_j) x_j + V_ij m_j ] + b_i

All modeling choices are explicit and configurable; defaults are documented,
not theorems:

- normalization: "explicit_sqrt_n" puts 1/sqrt(N) in the forward pass (new
  default; resolves the D3 ambiguity for active code). "legacy_init_only"
  reproduces the legacy convention — no runtime factor, W initialized at
  scale 1/sqrt(N) — and exists for regression fixtures.
- V (mask channel): whether V ≡ 0 is optimal for a fixed hidden-manifold
  teacher is unresolved (discrepancy D7). V is implemented cleanly; it can be
  frozen at zero (default, matching all recorded experiments) or trained as an
  ablation. Nothing here asserts symmetry forces V to vanish.
- bias: off by default; configurable, never inferred.
- W diagonal: under the masked objective, output i enters the loss only when
  site i is masked, and then its own input contribution is zero — so W_ii
  never multiplies a nonzero input in any trained or sampled logit. The
  diagonal is unidentifiable; default policy fixes it to zero rather than
  carrying unused parameters. "free" reproduces legacy behavior.
- regularization: `regularized_parameters()` returns only trainable
  parameters selected by the configuration (fixes D6 in the active path).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

Normalization = Literal["explicit_sqrt_n", "legacy_init_only"]
VPolicy = Literal["frozen_zero", "trainable"]
BiasPolicy = Literal["none", "trainable", "frozen_zero"]
DiagonalPolicy = Literal["zero", "free"]


@dataclass(frozen=True)
class LinearScoreConfig:
    visible_dim: int
    normalization: Normalization = "explicit_sqrt_n"
    v_policy: VPolicy = "frozen_zero"
    bias_policy: BiasPolicy = "none"
    diagonal_policy: DiagonalPolicy = "zero"

    def identity(self) -> dict[str, object]:
        return {
            "model": "linear_masked_score",
            "visible_dim": self.visible_dim,
            "normalization": self.normalization,
            "v_policy": self.v_policy,
            "bias_policy": self.bias_policy,
            "diagonal_policy": self.diagonal_policy,
        }


class LinearMaskedScore(nn.Module):
    def __init__(self, config: LinearScoreConfig, generator: torch.Generator):
        super().__init__()
        self.config = config
        n = config.visible_dim

        init_scale = 1.0 if config.normalization == "explicit_sqrt_n" else 1.0 / math.sqrt(n)
        w = torch.randn((n, n), generator=generator) * init_scale
        self.W = nn.Parameter(w)

        self.V = nn.Parameter(torch.zeros((n, n)))
        if config.v_policy == "frozen_zero":
            self.V.requires_grad_(False)

        if config.bias_policy == "none":
            self.b: nn.Parameter | None = None
        else:
            self.b = nn.Parameter(torch.zeros(n))
            if config.bias_policy == "frozen_zero":
                self.b.requires_grad_(False)

        if config.diagonal_policy == "zero":
            with torch.no_grad():
                self.W.fill_diagonal_(0.0)
            self.register_buffer("_diag_mask", 1.0 - torch.eye(n), persistent=False)
        else:
            self._diag_mask = None  # type: ignore[assignment]

    @property
    def runtime_scale(self) -> float:
        if self.config.normalization == "explicit_sqrt_n":
            return 1.0 / math.sqrt(self.config.visible_dim)
        return 1.0

    def effective_W(self) -> torch.Tensor:
        if self._diag_mask is not None:
            return self.W * self._diag_mask
        return self.W

    def forward(self, values: torch.Tensor, is_masked: torch.Tensor) -> torch.Tensor:
        """Logits h. `values`: clean/committed spins (±1); `is_masked`: bool.

        Masked coordinates contribute through V only; their spin value is
        ignored (multiplied by 1 - m). No in-band sentinel value is used.
        """
        m = is_masked.to(values.dtype)
        visible = values * (1.0 - m)
        h = visible @ self.effective_W().t() + m @ self.V.t()
        h = h * self.runtime_scale
        if self.b is not None:
            h = h + self.b
        return h

    def probabilities(self, values: torch.Tensor, is_masked: torch.Tensor) -> torch.Tensor:
        """p(x_i = +1 | visible context) = sigmoid(h_i)."""
        return torch.sigmoid(self.forward(values, is_masked))

    def regularized_parameters(self) -> list[nn.Parameter]:
        """Only trainable parameters are regularized (frozen V/bias excluded)."""
        params: list[nn.Parameter] = []
        if self.W.requires_grad:
            params.append(self.W)
        if self.V.requires_grad:
            params.append(self.V)
        if self.b is not None and self.b.requires_grad:
            params.append(self.b)
        return params

    def order_parameters(self) -> dict[str, float]:
        n = self.config.visible_dim
        return {
            "qW": float((self.effective_W() ** 2).mean().item() * n),
            "qV": float((self.V**2).mean().item() * n),
        }
