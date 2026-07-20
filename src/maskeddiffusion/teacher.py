"""Hidden-manifold teacher: x = sign(F z), F_ia ~ N(0, 1/D), quenched F.

Scientific contract: docs/RESEARCH_SPEC.md. The sign convention is explicit:
sign(0) := +1 (raw torch.sign would emit 0, colliding with downstream mask
conventions — discrepancy D8 is fixed here for the active path). Outputs
contain only -1 and +1.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import torch

from .dimensions import Dimensions


def sign_pm1(h: torch.Tensor) -> torch.Tensor:
    """sign with the contract convention sign(0) := +1; output strictly ±1."""
    return torch.where(h >= 0, 1.0, -1.0).to(h.dtype)


class HiddenManifoldTeacher:
    """Owns the quenched feature matrix F for one repeat.

    Sample F exactly once (via `sample`); hold it fixed; generate every split
    (train/validation/evaluation) through this same instance with independent
    generators. Dataset wrappers must receive a teacher instance and must not
    sample F themselves.
    """

    def __init__(self, dims: Dimensions, F: torch.Tensor, dtype: torch.dtype = torch.float32):
        if F.shape != (dims.visible_dim, dims.latent_dim):
            raise ValueError(
                f"F shape {tuple(F.shape)} != (visible_dim, latent_dim) = "
                f"({dims.visible_dim}, {dims.latent_dim})"
            )
        self.dims = dims
        self.dtype = dtype
        self.F = F.to(dtype=dtype, device="cpu")

    @staticmethod
    def sample(
        dims: Dimensions, generator: torch.Generator, dtype: torch.dtype = torch.float32
    ) -> HiddenManifoldTeacher:
        F = torch.randn(
            (dims.visible_dim, dims.latent_dim), generator=generator, dtype=dtype
        ) / math.sqrt(dims.latent_dim)
        return HiddenManifoldTeacher(dims, F, dtype=dtype)

    # -- identity ---------------------------------------------------------

    @property
    def teacher_id(self) -> str:
        """Stable content hash of dimensions + F bytes (float32, CPU)."""
        h = hashlib.sha256()
        h.update(repr(self.dims.to_dict()).encode())
        h.update(self.F.to(torch.float32).contiguous().numpy().tobytes())
        return "hmt-" + h.hexdigest()[:16]

    # -- sampling ---------------------------------------------------------

    def sample_latents(self, n: int, generator: torch.Generator) -> torch.Tensor:
        return torch.randn((n, self.dims.latent_dim), generator=generator, dtype=self.dtype)

    def sample_from_latents(self, z: torch.Tensor) -> torch.Tensor:
        return sign_pm1(z @ self.F.t())

    def sample_batch(self, n: int, generator: torch.Generator) -> torch.Tensor:
        """n fresh samples from the finite-F law P_F (fresh z, same F)."""
        return self.sample_from_latents(self.sample_latents(n, generator))

    # -- analytics --------------------------------------------------------

    def correlation_matrix(self) -> torch.Tensor:
        """Analytic pair correlation at fixed F:

        C_ij = (2/pi) * arcsin( F_i·F_j / (|F_i| |F_j|) ), clipped before arcsin.
        """
        norms = self.F.norm(dim=1, keepdim=True)
        cos = (self.F @ self.F.t()) / (norms * norms.t())
        cos = cos.clamp(-1.0, 1.0)
        # C_ii = 1 by definition; float32 gram diagonals can land at 1 - O(1e-4)
        cos.fill_diagonal_(1.0)
        return (2.0 / math.pi) * torch.asin(cos)

    # -- serialization (device-independent) --------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "format": "maskeddiffusion.teacher.v1",
            "dimensions": self.dims.to_dict(),
            "dtype": str(self.dtype).replace("torch.", ""),
            "F": self.F.to(torch.float32).cpu(),
            "teacher_id": self.teacher_id,
        }

    @staticmethod
    def from_state_dict(state: dict[str, Any]) -> HiddenManifoldTeacher:
        if state.get("format") != "maskeddiffusion.teacher.v1":
            raise ValueError(f"unknown teacher format {state.get('format')!r}")
        dims = Dimensions.from_dict(state["dimensions"])
        dtype = getattr(torch, state["dtype"])
        teacher = HiddenManifoldTeacher(dims, state["F"], dtype=dtype)
        recorded = state.get("teacher_id")
        if recorded is not None and recorded != teacher.teacher_id:
            raise ValueError(
                f"teacher_id mismatch: stored {recorded}, recomputed {teacher.teacher_id}"
            )
        return teacher

    def save(self, path: str | Path) -> None:
        torch.save(self.state_dict(), Path(path))

    @staticmethod
    def load(path: str | Path) -> HiddenManifoldTeacher:
        state = torch.load(Path(path), map_location="cpu", weights_only=False)
        return HiddenManifoldTeacher.from_state_dict(state)
