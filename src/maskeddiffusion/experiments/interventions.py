"""Intervention registry for the Phase 4C paired-experiment engine.

An intervention names the ONE permitted difference between the arms of a
paired comparison group; every other spec field — including the full
ten-stream seed hierarchy — must be identical across arms
(docs/PHASE4C_EXPERIMENT_PROTOCOL.md).

Four interventions are supported:

- ``v_trainability``: frozen-zero V versus trainable V
  (``model.v_policy``; open question Q1 of docs/RESEARCH_SPEC.md).
- ``sampler_stochasticity``: a stochastic sampler versus its corresponding
  greedy sampler (``sampler.sampler_name``; the pair must be one of the
  declared correspondences in ``SAMPLER_FAMILIES`` — e.g.
  ``sequential_random_stochastic`` ↔ ``sequential_random_greedy``).
- ``optimization_budget``: identical training configurations at different
  ``training.max_steps`` (same seeds, so the shorter trajectory is a prefix
  of the longer one).
- ``finite_d``: different ``latent_dim`` at fixed ``aspect_ratio`` /
  ``sample_ratio`` (and fixed seed VALUES). A different `latent_dim` means a
  different-shape quenched teacher `F` — two arms of this intervention
  CANNOT share disorder, no matter what seed values they were given, because
  `F` has a different shape in each arm and is therefore necessarily a
  distinct draw with a distinct `teacher_id`. This is fundamentally not the
  same comparison as the other three interventions and must never be
  described or treated as one: see `comparison_type` below.

`comparison_type` names what kind of comparison an intervention produces,
and is the single source of truth other layers key off of (never re-derive
it from `strict_disorder`, a name that no longer exists):

- ``"paired_disorder"`` (`v_trainability`, `sampler_stochasticity`,
  `optimization_budget`): arms share the full seed hierarchy AND, on CPU,
  therefore share content-identical disorder — same quenched `F`, same
  training/validation/evaluation draws. `teacher_id` is required equal
  across arms (`experiments.runner` asserts and records this).
- ``"matched_seed_finite_size"`` (`finite_d` only): arms share every seed
  VALUE (so any two runs at the same repeat index used the same random
  streams) but NOT disorder content, because `latent_dim` differs and so
  does the teacher's shape. Rules (enforced across `pairs.py`,
  `runner.py`, and `analysis.rows`): same repeat index; same seed hierarchy
  values; same `aspect_ratio`/`sample_ratio`; different `(D, N, M)`;
  DISTINCT `teacher_id` (asserted, not merely tolerated); never labeled or
  consumed as a paired same-disorder (`"paired_disorder"`) comparison —
  `analysis.statistics.paired_differences` excludes it by default, and any
  matched-seed comparison the analysis layer does produce is written to a
  separately named, explicitly labeled non-paired output
  (`analysis.statistics.matched_seed_finite_size_frame`,
  `p4c_matched_seed_finite_size.csv`).

`permitted_diff_fields` are leaf paths into `ExperimentSpec.to_dict()`;
`experiments.pairs.validate_pair` rejects any arm-to-arm difference outside
this set (plus the mandatory `condition` label).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from .spec import _require_slug

ComparisonType = Literal["paired_disorder", "matched_seed_finite_size"]
COMPARISON_TYPES: tuple[str, ...] = ("paired_disorder", "matched_seed_finite_size")

# Stochastic sampler -> its corresponding greedy sampler (same coordinate
# selection, same tokens-per-step semantics; only the token decode differs).
SAMPLER_FAMILIES: dict[str, tuple[str, str]] = {
    "sequential_random": ("sequential_random_stochastic", "sequential_random_greedy"),
    "parallel_random": ("parallel_random_stochastic", "parallel_random_greedy"),
    "one_shot": ("one_shot_stochastic", "one_shot_greedy"),
}
STOCHASTIC_TO_GREEDY: dict[str, str] = {s: g for s, g in SAMPLER_FAMILIES.values()}


@dataclass(frozen=True)
class ArmDefinition:
    """One condition label plus the base-table overrides that define it."""

    condition: str
    overrides: dict[str, dict[str, Any]]  # e.g. {"model": {"v_policy": "trainable"}}


@dataclass(frozen=True)
class Intervention:
    name: str
    comparison_type: ComparisonType
    permitted_diff_fields: frozenset[str]
    build_arms: Callable[[dict[str, Any]], tuple[ArmDefinition, ...]]

    def __post_init__(self) -> None:
        if self.comparison_type not in COMPARISON_TYPES:
            raise ValueError(
                f"comparison_type must be one of {COMPARISON_TYPES}, got {self.comparison_type!r}"
            )


def _reject_unknown_params(name: str, params: dict[str, Any], known: set[str]) -> None:
    unknown = set(params) - known
    if unknown:
        raise ValueError(
            f"intervention {name!r} got unknown parameters {sorted(unknown)}; "
            f"known: {sorted(known)}"
        )


def _v_trainability_arms(params: dict[str, Any]) -> tuple[ArmDefinition, ...]:
    _reject_unknown_params("v_trainability", params, set())
    return (
        ArmDefinition("frozen_zero_v", {"model": {"v_policy": "frozen_zero"}}),
        ArmDefinition("trainable_v", {"model": {"v_policy": "trainable"}}),
    )


def _sampler_stochasticity_arms(params: dict[str, Any]) -> tuple[ArmDefinition, ...]:
    _reject_unknown_params("sampler_stochasticity", params, {"sampler_family"})
    family = params.get("sampler_family")
    if family not in SAMPLER_FAMILIES:
        raise ValueError(
            f"sampler_family must be one of {sorted(SAMPLER_FAMILIES)}, got {family!r}"
        )
    stochastic, greedy = SAMPLER_FAMILIES[family]
    return (
        ArmDefinition("stochastic", {"sampler": {"sampler_name": stochastic}}),
        ArmDefinition("greedy", {"sampler": {"sampler_name": greedy}}),
    )


def _optimization_budget_arms(params: dict[str, Any]) -> tuple[ArmDefinition, ...]:
    _reject_unknown_params("optimization_budget", params, {"budgets"})
    budgets = params.get("budgets")
    if not isinstance(budgets, dict) or len(budgets) < 2:
        raise ValueError(
            "optimization_budget requires 'budgets': a table of >= 2 "
            "{condition_label: max_steps}, e.g. budgets = { short = 60, long = 600 }"
        )
    arms: list[ArmDefinition] = []
    for label, steps in budgets.items():
        _require_slug("budget condition label", label)
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
            raise ValueError(f"budget {label!r}: max_steps must be an int >= 1, got {steps!r}")
        arms.append(ArmDefinition(label, {"training": {"max_steps": steps}}))
    if len({a.overrides["training"]["max_steps"] for a in arms}) < 2:
        raise ValueError("optimization_budget budgets must contain at least two distinct values")
    return tuple(arms)


def _finite_d_arms(params: dict[str, Any]) -> tuple[ArmDefinition, ...]:
    _reject_unknown_params("finite_d", params, {"latent_dims"})
    latent_dims = params.get("latent_dims")
    if (
        not isinstance(latent_dims, list)
        or len(latent_dims) < 2
        or len(set(latent_dims)) != len(latent_dims)
    ):
        raise ValueError(
            "finite_d requires 'latent_dims': a list of >= 2 distinct latent dimensions"
        )
    arms: list[ArmDefinition] = []
    for d in latent_dims:
        if not isinstance(d, int) or isinstance(d, bool) or d < 1:
            raise ValueError(f"finite_d latent_dim must be an int >= 1, got {d!r}")
        arms.append(ArmDefinition(f"d{d}", {"dimensions": {"latent_dim": d}}))
    return tuple(arms)


INTERVENTIONS: dict[str, Intervention] = {
    "v_trainability": Intervention(
        name="v_trainability",
        comparison_type="paired_disorder",
        permitted_diff_fields=frozenset({"model.v_policy"}),
        build_arms=_v_trainability_arms,
    ),
    "sampler_stochasticity": Intervention(
        name="sampler_stochasticity",
        comparison_type="paired_disorder",
        permitted_diff_fields=frozenset({"sampler.sampler_name"}),
        build_arms=_sampler_stochasticity_arms,
    ),
    "optimization_budget": Intervention(
        name="optimization_budget",
        comparison_type="paired_disorder",
        permitted_diff_fields=frozenset({"training.max_steps"}),
        build_arms=_optimization_budget_arms,
    ),
    "finite_d": Intervention(
        name="finite_d",
        comparison_type="matched_seed_finite_size",
        permitted_diff_fields=frozenset(
            {
                "dimensions.latent_dim",
                "dimensions.visible_dim",
                "dimensions.train_size",
                "dimensions.visible_load",
                "model.visible_dim",
            }
        ),
        build_arms=_finite_d_arms,
    ),
}


def get_intervention(name: str) -> Intervention:
    if name not in INTERVENTIONS:
        raise ValueError(f"unknown intervention {name!r}; known: {sorted(INTERVENTIONS)}")
    return INTERVENTIONS[name]
