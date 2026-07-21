"""Serializable specification of one arm of one paired comparison group.

An `ExperimentSpec` pins every quantity that defines a single run of the
Phase 4C paired-experiment engine (docs/PHASE4C_EXPERIMENT_PROTOCOL.md):
identity (experiment / pair / repeat / condition), resolved dimensions, the
full ten-stream seed hierarchy, model/training/sampler configuration,
evaluation settings, and the generation count.

Specs are frozen dataclasses; `to_dict` / `from_dict` / `to_json` give a
deterministic serialization and `spec_fingerprint` content-hashes a spec so
that a completed run's manifest can be checked for provenance consistency
on resume. `to_dict` deliberately contains only configuration state —
derived identities (`LinearScoreConfig.identity()`, `SamplerConfig.identity()`,
`training.optimizer_identity`) are added at render time by
`spec_identities`, so a recursive diff of two `to_dict()` outputs differs
exactly where the underlying configuration differs (see
`experiment.pairs.validate_pair`).

Paired-disorder discipline is NOT enforced here but in
`experiment.pairs`: arms of one comparison group must differ only in the
fields the intervention explicitly permits; the full seed hierarchy is
always identical across arms.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..config import RunConfig, TrainingConfig
from ..dimensions import Dimensions
from ..models import LinearScoreConfig
from ..randomness import SeedHierarchy
from ..samplers import SamplerConfig
from ..training import optimizer_identity
from ..uturn import UTurnConfig

# Path-safe identity labels: used as single directory components, so no
# separators, no leading dots, no whitespace.
_SLUG = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")

_SPEC_KEYS = (
    "experiment_id",
    "pair_id",
    "repeat_id",
    "intervention",
    "condition",
    "dimensions",
    "seeds",
    "model",
    "training",
    "sampler",
    "evaluation",
    "n_generate",
    "uturn",
)


def _uturn_to_dict(uturn: UTurnConfig | None) -> dict[str, Any] | None:
    if uturn is None:
        return None
    return {
        "t_values": list(uturn.t_values),
        "n_examples": uturn.n_examples,
        "sources": list(uturn.sources),
    }


def _uturn_from_dict(d: dict[str, Any] | None) -> UTurnConfig | None:
    if d is None:
        return None
    return UTurnConfig(
        t_values=tuple(d["t_values"]), n_examples=d["n_examples"], sources=tuple(d["sources"])
    )


def _require_slug(name: str, value: Any) -> None:
    if not isinstance(value, str) or not _SLUG.match(value):
        raise ValueError(f"{name} must be a path-safe slug matching {_SLUG.pattern}, got {value!r}")


@dataclass(frozen=True)
class EvaluationConfig:
    """Fresh P_F evaluation draws and MMD kernel scales for one run.

    `n_true` fresh samples are drawn twice per evaluation (the
    `evaluation_data_seed` and `metric_seed` streams), once as the Model-True
    reference and once as the True-True noise-floor batch.
    """

    n_true: int = 1000
    lambdas: tuple[float, ...] = (4.0, 8.0)

    def __post_init__(self) -> None:
        if not isinstance(self.n_true, int) or isinstance(self.n_true, bool):
            raise TypeError(f"n_true must be int, got {type(self.n_true).__name__}")
        if self.n_true < 2:
            raise ValueError(f"n_true must be >= 2 (MMD estimator), got {self.n_true}")
        if not self.lambdas:
            raise ValueError("lambdas must be a nonempty tuple of kernel scales")
        for lam in self.lambdas:
            if not isinstance(lam, (int, float)) or isinstance(lam, bool):
                raise TypeError(f"kernel scale must be a number, got {type(lam).__name__}")
            if not math.isfinite(lam) or lam <= 0:
                raise ValueError(f"kernel scale must be finite and > 0, got {lam}")

    def to_dict(self) -> dict[str, Any]:
        return {"n_true": self.n_true, "lambdas": list(self.lambdas)}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> EvaluationConfig:
        return EvaluationConfig(n_true=d["n_true"], lambdas=tuple(d["lambdas"]))


@dataclass(frozen=True)
class ExperimentSpec:
    """One arm of one paired comparison group (one runnable unit)."""

    experiment_id: str
    pair_id: str
    repeat_id: int
    intervention: str
    condition: str
    dimensions: Dimensions
    seeds: SeedHierarchy
    model: LinearScoreConfig
    training: TrainingConfig = field(default_factory=TrainingConfig)
    sampler: SamplerConfig = field(
        default_factory=lambda: SamplerConfig("sequential_random_stochastic")
    )
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    n_generate: int = 1
    uturn: UTurnConfig | None = None

    def __post_init__(self) -> None:
        _require_slug("experiment_id", self.experiment_id)
        _require_slug("pair_id", self.pair_id)
        _require_slug("condition", self.condition)
        if not isinstance(self.intervention, str) or not self.intervention:
            raise ValueError(f"intervention must be a nonempty string, got {self.intervention!r}")
        if not isinstance(self.repeat_id, int) or isinstance(self.repeat_id, bool):
            raise TypeError(f"repeat_id must be int, got {type(self.repeat_id).__name__}")
        if self.repeat_id < 0:
            raise ValueError(f"repeat_id must be >= 0, got {self.repeat_id}")
        if not isinstance(self.n_generate, int) or isinstance(self.n_generate, bool):
            raise TypeError(f"n_generate must be int, got {type(self.n_generate).__name__}")
        if self.n_generate < 1:
            raise ValueError(
                f"n_generate must be >= 1 — an experiment arm that generates no "
                f"samples has no terminal-law diagnostic, got {self.n_generate}"
            )

    def to_run_config(self) -> RunConfig:
        """The deterministic resolved run configuration for this arm."""
        return RunConfig(
            dimensions=self.dimensions,
            seeds=self.seeds,
            model=self.model,
            training=self.training,
            sampler=self.sampler,
            n_generate=self.n_generate,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "pair_id": self.pair_id,
            "repeat_id": self.repeat_id,
            "intervention": self.intervention,
            "condition": self.condition,
            "dimensions": self.dimensions.to_dict(),
            "seeds": self.seeds.to_dict(),
            "model": asdict(self.model),
            "training": asdict(self.training),
            "sampler": asdict(self.sampler),
            "evaluation": self.evaluation.to_dict(),
            "n_generate": self.n_generate,
            "uturn": _uturn_to_dict(self.uturn),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ExperimentSpec:
        unknown = set(d) - set(_SPEC_KEYS)
        if unknown:
            raise ValueError(f"unknown ExperimentSpec keys {sorted(unknown)}")
        missing = [k for k in _SPEC_KEYS if k not in d]
        if missing:
            raise ValueError(f"ExperimentSpec missing keys {missing}")
        model_raw = dict(d["model"])
        return ExperimentSpec(
            experiment_id=d["experiment_id"],
            pair_id=d["pair_id"],
            repeat_id=d["repeat_id"],
            intervention=d["intervention"],
            condition=d["condition"],
            dimensions=Dimensions.from_dict(d["dimensions"]),
            seeds=SeedHierarchy.from_dict(d["seeds"]),
            model=LinearScoreConfig(**model_raw),
            training=TrainingConfig(**d["training"]),
            sampler=SamplerConfig(**d["sampler"]),
            evaluation=EvaluationConfig.from_dict(d["evaluation"]),
            n_generate=d["n_generate"],
            uturn=_uturn_from_dict(d["uturn"]),
        )

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")


def spec_identities(spec: ExperimentSpec) -> dict[str, Any]:
    """Derived model/optimizer/sampler identities, for manifests and dry runs.

    Kept out of `ExperimentSpec.to_dict` so pair diffs reflect only actual
    configuration state (see module docstring).
    """
    return {
        "model": spec.model.identity(),
        "optimizer": optimizer_identity(spec.training),
        "sampler": spec.sampler.identity(),
    }


def spec_fingerprint(spec: ExperimentSpec) -> str:
    """Stable content hash of the full spec (identity, dims, seeds, configs)."""
    payload = json.dumps(spec.to_dict(), sort_keys=True, separators=(",", ":"))
    return "expspec-" + hashlib.sha256(payload.encode()).hexdigest()[:16]
