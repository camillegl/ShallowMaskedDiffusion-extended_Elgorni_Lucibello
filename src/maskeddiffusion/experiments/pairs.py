"""Paired-disorder enforcement for the Phase 4C engine.

`validate_pair` is the strict guard required by
docs/PHASE4C_EXPERIMENT_PROTOCOL.md: within one comparison group the two
arms must share the teacher seed, every train/validation/evaluation latent
stream, the metric seeds, the sampler seeds, the dimensions (except under
`finite_d`), and every model/training/sampler field not under intervention.
Only the fields named by the intervention's `permitted_diff_fields` (plus
the `condition` label itself) may differ; anything else is reported as a
problem with the exact differing leaf path.

The check is a recursive diff of the two specs' `to_dict()` output against
the allowlist — it cannot be bypassed by renaming a field, and it
automatically covers future fields added to `ExperimentSpec`.
"""

from __future__ import annotations

from typing import Any

from .interventions import STOCHASTIC_TO_GREEDY, get_intervention
from .spec import ExperimentSpec

# Identity fields that must be EQUAL across arms (condition must DIFFER).
_IDENTITY_FIELDS = ("experiment_id", "pair_id", "repeat_id", "intervention")


def diff_leaves(x: Any, y: Any, path: str = "") -> set[str]:
    """Leaf paths at which two nested dict/list/scalar structures differ."""
    if isinstance(x, dict) and isinstance(y, dict):
        out: set[str] = set()
        for key in x.keys() | y.keys():
            p = f"{path}.{key}" if path else str(key)
            if key not in x or key not in y:
                out.add(p)
            else:
                out |= diff_leaves(x[key], y[key], p)
        return out
    if isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
        out = set()
        for i, (a, b) in enumerate(zip(x, y, strict=True)):
            out |= diff_leaves(a, b, f"{path}[{i}]")
        return out
    if x != y:
        return {path or "<root>"}
    return set()


def validate_pair(a: ExperimentSpec, b: ExperimentSpec) -> list[str]:
    """Problems with the arm pair (a, b); an empty list means the pair is valid."""
    problems: list[str] = []

    for key in _IDENTITY_FIELDS:
        if getattr(a, key) != getattr(b, key):
            problems.append(
                f"paired arms must share {key}: {getattr(a, key)!r} != {getattr(b, key)!r}"
            )
    if a.condition == b.condition:
        problems.append(f"paired arms must have distinct conditions, both are {a.condition!r}")
    if problems:
        return problems

    intervention = get_intervention(a.intervention)  # raises on unknown intervention
    allowed = set(intervention.permitted_diff_fields) | {"condition"}
    differing = diff_leaves(a.to_dict(), b.to_dict())

    unpermitted = differing - allowed
    for path in sorted(unpermitted):
        problems.append(
            f"paired arms under intervention {intervention.name!r} may differ only in "
            f"{sorted(intervention.permitted_diff_fields)}, but found a difference at "
            f"{path!r}"
        )

    substantive = differing - {"condition"}
    if not unpermitted and not substantive:
        problems.append(
            f"arms {a.condition!r} and {b.condition!r} are identical apart from the "
            f"condition label — intervention {intervention.name!r} has no effect"
        )

    # Sampler-pair correspondence: a stochasticity comparison must oppose a
    # stochastic sampler and ITS greedy counterpart, not an arbitrary greedy
    # sampler with a different coordinate schedule.
    if intervention.name == "sampler_stochasticity" and not unpermitted:
        names = {a.sampler.sampler_name, b.sampler.sampler_name}
        paired = any(
            names == {stochastic, greedy} for stochastic, greedy in STOCHASTIC_TO_GREEDY.items()
        )
        if not paired:
            problems.append(
                f"sampler_stochasticity requires a corresponding stochastic/greedy pair "
                f"from {sorted(STOCHASTIC_TO_GREEDY)}, got {sorted(names)}"
            )

    return problems


def validate_group(specs: list[ExperimentSpec] | tuple[ExperimentSpec, ...]) -> list[str]:
    """Problems with a whole comparison group; empty list means valid."""
    problems: list[str] = []
    if len(specs) < 2:
        return [f"a comparison group needs at least 2 arms, got {len(specs)}"]
    conditions = [s.condition for s in specs]
    if len(set(conditions)) != len(conditions):
        problems.append(f"duplicate conditions in group: {conditions}")
    for i in range(len(specs)):
        for j in range(i + 1, len(specs)):
            problems += validate_pair(specs[i], specs[j])
    return problems
