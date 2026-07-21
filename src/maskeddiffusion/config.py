"""Typed run configuration: dataclasses + TOML in, resolved JSON out.

ADR 002. Contract naming is enforced here: bare `alpha`/`gamma` keys are
rejected with an explanatory error (docs/NOTATION.md); there is no silent
mapping from legacy conventions.
"""

from __future__ import annotations

import json
import math
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .dimensions import Dimensions
from .models import LinearScoreConfig
from .randomness import SeedHierarchy
from .samplers import SamplerConfig

_FORBIDDEN_KEYS = {"alpha", "gamma", "L"}


def _require_positive_int(name: str, value: Any) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be int, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")


def _require_nonnegative_int(name: str, value: Any) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")


@dataclass(frozen=True)
class TrainingConfig:
    max_steps: int = 1000
    batch_size: int = 64
    learning_rate: float = 1e-3
    l2reg: float = 0.0
    min_time: float = 0.0
    validation_size: int = 0
    validation_every: int = 100
    checkpoint_every: int = 0  # 0 = only final checkpoint
    log_every: int = 10

    def __post_init__(self) -> None:
        _require_positive_int("max_steps", self.max_steps)
        _require_positive_int("batch_size", self.batch_size)
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be finite and > 0, got {self.learning_rate}")
        if not math.isfinite(self.l2reg) or self.l2reg < 0:
            raise ValueError(f"l2reg must be finite and >= 0, got {self.l2reg}")
        # min_time shifts t ~ U(min_time, 1) (docs/RESEARCH_SPEC.md); must
        # leave a nonempty sampling interval, and t=1 identically would mean
        # every position is always masked, degenerate for the objective.
        if not math.isfinite(self.min_time) or not (0.0 <= self.min_time < 1.0):
            raise ValueError(f"min_time must satisfy 0 <= min_time < 1, got {self.min_time}")
        _require_nonnegative_int("validation_size", self.validation_size)
        _require_positive_int("validation_every", self.validation_every)
        _require_nonnegative_int("checkpoint_every", self.checkpoint_every)
        _require_positive_int("log_every", self.log_every)


@dataclass(frozen=True)
class RunConfig:
    dimensions: Dimensions
    seeds: SeedHierarchy
    model: LinearScoreConfig
    training: TrainingConfig = field(default_factory=TrainingConfig)
    sampler: SamplerConfig = field(
        default_factory=lambda: SamplerConfig("sequential_random_stochastic")
    )
    n_generate: int = 0  # samples to generate after training (0 = none)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimensions": self.dimensions.to_dict(),
            "seeds": self.seeds.to_dict(),
            "model": asdict(self.model),
            "training": asdict(self.training),
            "sampler": asdict(self.sampler),
            "n_generate": self.n_generate,
        }

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")


def _reject_forbidden(table: dict[str, Any], where: str) -> None:
    for key in table:
        if key in _FORBIDDEN_KEYS:
            raise ValueError(
                f"config key {key!r} in [{where}] is forbidden by docs/NOTATION.md: "
                "use latent_dim / aspect_ratio / sample_ratio (alpha is ambiguous — "
                "legacy CLI used it for M/N, the contract uses M/D)"
            )
        if isinstance(table[key], dict):
            _reject_forbidden(table[key], f"{where}.{key}")


def load_config(path: str | Path) -> RunConfig:
    raw = tomllib.loads(Path(path).read_text())
    _reject_forbidden(raw, "root")

    dims_raw = raw.get("dimensions")
    if dims_raw is None:
        raise ValueError("config must contain a [dimensions] table")
    dims = Dimensions.resolve(
        latent_dim=dims_raw["latent_dim"],
        aspect_ratio=dims_raw["aspect_ratio"],
        sample_ratio=dims_raw["sample_ratio"],
    )

    seeds_raw = raw.get("seeds", {})
    if "base_seed" in seeds_raw and len(seeds_raw) == 1:
        seeds = SeedHierarchy.from_base(int(seeds_raw["base_seed"]))
    elif seeds_raw:
        seeds = SeedHierarchy.from_dict(seeds_raw)
    else:
        raise ValueError("config must set [seeds] (base_seed or the full hierarchy)")

    model_raw = raw.get("model", {})
    model = LinearScoreConfig(visible_dim=dims.visible_dim, **model_raw)

    training = TrainingConfig(**raw.get("training", {}))
    sampler_raw = raw.get("sampler", {})
    sampler = (
        SamplerConfig(**sampler_raw)
        if sampler_raw
        else SamplerConfig("sequential_random_stochastic")
    )

    return RunConfig(
        dimensions=dims,
        seeds=seeds,
        model=model,
        training=training,
        sampler=sampler,
        n_generate=int(raw.get("n_generate", 0)),
    )
