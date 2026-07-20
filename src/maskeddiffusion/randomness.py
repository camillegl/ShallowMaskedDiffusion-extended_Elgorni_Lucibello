"""Hierarchical seed configuration and independent torch generators.

Core scientific functions never touch global RNG state; every stream is an
independent `torch.Generator` derived from a named seed. Run manifests record
the full hierarchy.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

import torch

STREAMS: tuple[str, ...] = (
    "teacher_seed",
    "train_data_seed",
    "validation_data_seed",
    "evaluation_data_seed",
    "model_seed",
    "mask_seed",
    "dataloader_seed",
    "sampler_order_seed",
    "sampler_token_seed",
    "metric_seed",
)

# Fixed per-stream offsets so `SeedHierarchy.from_base(s)` yields distinct,
# reproducible seeds per stream.
_STREAM_OFFSETS: dict[str, int] = {name: 1000 + 17 * i for i, name in enumerate(STREAMS)}


@dataclass(frozen=True)
class SeedHierarchy:
    teacher_seed: int
    train_data_seed: int
    validation_data_seed: int
    evaluation_data_seed: int
    model_seed: int
    mask_seed: int
    dataloader_seed: int
    sampler_order_seed: int
    sampler_token_seed: int
    metric_seed: int
    base_seed: int | None = field(default=None)

    @staticmethod
    def from_base(base_seed: int) -> SeedHierarchy:
        return SeedHierarchy(
            base_seed=base_seed,
            **{name: base_seed + _STREAM_OFFSETS[name] for name in STREAMS},
        )

    def generator(self, stream: str, device: str | torch.device = "cpu") -> torch.Generator:
        if stream not in STREAMS:
            raise KeyError(f"unknown seed stream {stream!r}; known: {STREAMS}")
        gen = torch.Generator(device=device)
        gen.manual_seed(int(getattr(self, stream)))
        return gen

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> SeedHierarchy:
        return SeedHierarchy(**{k: d[k] for k in [*STREAMS, "base_seed"] if k in d})


def generator_state(gen: torch.Generator) -> torch.Tensor:
    return gen.get_state()


def set_generator_state(gen: torch.Generator, state: torch.Tensor) -> None:
    gen.set_state(state)
