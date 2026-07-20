"""Named samplers over the trained conditionals.

Each sampler defines its own sampler-indexed terminal law P_{θ,A,k}
(docs/RESEARCH_SPEC.md). None of these is ancestral sampling of a joint law;
the trained single-site conditionals are not known to be mutually consistent.

All samplers here are irreversible: committed tokens are never revised
(revision_policy = "never"). Coordinate selection and token selection use
separate generators.

Legacy correspondence (docs/ORIGINAL_ARCHITECTURE.md):
- sequential_random_stochastic  == legacy `sample`/"fair", k=1 (the notes'
  Algorithm 1 with k=1; the original paper-style stochastic sampler)
- parallel_random_stochastic    == legacy `sample` with k>1 (Algorithm 1)
- sequential_random_greedy      == legacy `mask_and_sample` "greedy"
- sequential_confidence_greedy  == legacy `mask_and_sample` "verygreedy"
- one_shot_stochastic/greedy    == legacy `mask_and_sample_oneshot` fair/greedy
- parallel_random_greedy        == no legacy counterpart (new, explicit)

`temperature` has no legacy or spec provenance; it must stay at 1.0 for any
contract-comparable run, and a non-unit value is a new sampler identity that
must be justified and recorded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch

from .models import LinearMaskedScore

SamplerName = Literal[
    "sequential_random_stochastic",
    "sequential_random_greedy",
    "sequential_confidence_greedy",
    "parallel_random_stochastic",
    "parallel_random_greedy",
    "one_shot_stochastic",
    "one_shot_greedy",
]


@dataclass(frozen=True)
class SamplerConfig:
    sampler_name: SamplerName
    tokens_per_step: int = 1
    temperature: float = 1.0
    revision_policy: Literal["never"] = "never"

    def identity(self) -> dict[str, object]:
        coordinate_selection = {
            "sequential_random_stochastic": "uniform_random",
            "sequential_random_greedy": "uniform_random",
            "sequential_confidence_greedy": "max_abs_logit",
            "parallel_random_stochastic": "uniform_random",
            "parallel_random_greedy": "uniform_random",
            "one_shot_stochastic": "all_at_once",
            "one_shot_greedy": "all_at_once",
        }[self.sampler_name]
        token_selection = (
            "threshold_at_half" if "greedy" in self.sampler_name else "bernoulli_sigmoid"
        )
        return {
            "sampler_name": self.sampler_name,
            "tokens_per_step": self.tokens_per_step,
            "temperature": self.temperature,
            "revision_policy": self.revision_policy,
            "coordinate_selection": coordinate_selection,
            "token_selection": token_selection,
        }


@dataclass
class SampleResult:
    values: torch.Tensor  # (B, N) final spins, all ±1
    trajectory: list[torch.Tensor] = field(default_factory=list)  # optional states
    trajectory_masks: list[torch.Tensor] = field(default_factory=list)


def _decode_tokens(
    logits: torch.Tensor,
    stochastic: bool,
    temperature: float,
    token_generator: torch.Generator | None,
) -> torch.Tensor:
    if stochastic:
        if token_generator is None:
            raise ValueError("stochastic decoding requires a token generator")
        p = torch.sigmoid(logits / temperature)
        u = torch.rand(p.shape, generator=token_generator, device=p.device)
        return torch.where(u < p, 1.0, -1.0)
    return torch.where(logits >= 0, 1.0, -1.0)


def sample(
    model: LinearMaskedScore,
    config: SamplerConfig,
    batch_size: int,
    *,
    order_generator: torch.Generator,
    token_generator: torch.Generator | None = None,
    initial_values: torch.Tensor | None = None,
    initial_mask: torch.Tensor | None = None,
    record_trajectory: bool = False,
) -> SampleResult:
    """Run the named sampler until no masked coordinates remain.

    Starts fully masked unless (initial_values, initial_mask) provide a
    partially observed state (reconstruction setting). Committed/observed
    tokens are never revised.
    """
    n = model.config.visible_dim
    device = next(model.parameters()).device
    if initial_values is None:
        values = torch.zeros((batch_size, n), device=device)
        is_masked = torch.ones((batch_size, n), dtype=torch.bool, device=device)
    else:
        if initial_mask is None:
            raise ValueError("initial_values requires initial_mask")
        values = initial_values.clone().to(device)
        is_masked = initial_mask.clone().to(device)

    stochastic = "stochastic" in config.sampler_name
    result = SampleResult(values=values)
    if record_trajectory:
        result.trajectory.append(values.clone())
        result.trajectory_masks.append(is_masked.clone())

    name = config.sampler_name
    if name in ("one_shot_stochastic", "one_shot_greedy"):
        logits = model(values, is_masked)
        decoded = _decode_tokens(logits, stochastic, config.temperature, token_generator)
        values = torch.where(is_masked, decoded, values)
        is_masked = torch.zeros_like(is_masked)
        result.values = values
        if record_trajectory:
            result.trajectory.append(values.clone())
            result.trajectory_masks.append(is_masked.clone())
        return result

    if name in (
        "sequential_random_stochastic",
        "sequential_random_greedy",
        "sequential_confidence_greedy",
    ):
        k = 1
    else:  # parallel_*
        k = config.tokens_per_step
        if k < 1:
            raise ValueError("tokens_per_step must be >= 1")

    while bool(is_masked.any()):
        logits = model(values, is_masked)
        if name == "sequential_confidence_greedy":
            # pick the still-masked coordinate with max |logit| per row
            confidence = logits.abs().masked_fill(~is_masked, float("-inf"))
            chosen = confidence.argmax(dim=1, keepdim=True)  # (B, 1)
            select = torch.zeros_like(is_masked)
            select.scatter_(1, chosen, True)
            select &= is_masked
        else:
            # choose up to k masked coordinates uniformly at random per row
            scores = torch.rand(is_masked.shape, generator=order_generator, device=device)
            scores = scores.masked_fill(~is_masked, float("inf"))
            order = scores.argsort(dim=1)
            nk = torch.minimum(is_masked.sum(dim=1), torch.full_like(is_masked.sum(dim=1), k))
            select = torch.zeros_like(is_masked)
            for row in range(select.shape[0]):
                take = int(nk[row].item())
                if take > 0:
                    select[row, order[row, :take]] = True
            select &= is_masked

        decoded = _decode_tokens(logits, stochastic, config.temperature, token_generator)
        values = torch.where(select, decoded, values)
        is_masked = is_masked & ~select
        if record_trajectory:
            result.trajectory.append(values.clone())
            result.trajectory_masks.append(is_masked.clone())

    result.values = values
    return result
