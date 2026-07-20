"""Resolved problem dimensions.

One `Dimensions` object is authoritative for a run; no other module may recompute
N or M from the ratios. Rounding rule (docs/NOTATION.md):

    visible_dim = round(aspect_ratio * latent_dim)
    train_size  = round(sample_ratio * latent_dim)

`visible_load` is derived metadata (M/N), never a primary control parameter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Dimensions:
    latent_dim: int
    aspect_ratio: float
    sample_ratio: float
    # Derived; filled by `resolve`. Authoritative integers for the run.
    visible_dim: int
    train_size: int
    visible_load: float

    @staticmethod
    def resolve(latent_dim: int, aspect_ratio: float, sample_ratio: float) -> Dimensions:
        if not isinstance(latent_dim, int) or isinstance(latent_dim, bool):
            raise TypeError(f"latent_dim must be int, got {type(latent_dim).__name__}")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be > 0, got {latent_dim}")
        for name, value in (("aspect_ratio", aspect_ratio), ("sample_ratio", sample_ratio)):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")
        visible_dim = round(aspect_ratio * latent_dim)
        train_size = round(sample_ratio * latent_dim)
        if visible_dim < 1:
            raise ValueError(
                f"visible_dim = round({aspect_ratio} * {latent_dim}) = {visible_dim} < 1"
            )
        if train_size < 1:
            raise ValueError(
                f"train_size = round({sample_ratio} * {latent_dim}) = {train_size} < 1"
            )
        return Dimensions(
            latent_dim=latent_dim,
            aspect_ratio=float(aspect_ratio),
            sample_ratio=float(sample_ratio),
            visible_dim=visible_dim,
            train_size=train_size,
            visible_load=train_size / visible_dim,
        )

    def __post_init__(self) -> None:
        expected_load = self.train_size / self.visible_dim
        if abs(self.visible_load - expected_load) > 1e-12:
            raise ValueError(
                f"visible_load {self.visible_load} != train_size/visible_dim {expected_load}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "latent_dim": self.latent_dim,
            "aspect_ratio": self.aspect_ratio,
            "sample_ratio": self.sample_ratio,
            "visible_dim": self.visible_dim,
            "train_size": self.train_size,
            "visible_load": self.visible_load,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Dimensions:
        dims = Dimensions.resolve(
            latent_dim=d["latent_dim"],
            aspect_ratio=d["aspect_ratio"],
            sample_ratio=d["sample_ratio"],
        )
        for key in ("visible_dim", "train_size"):
            if key in d and d[key] != getattr(dims, key):
                raise ValueError(
                    f"stored {key}={d[key]} disagrees with resolved {getattr(dims, key)}"
                )
        return dims
