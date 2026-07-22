"""U-turn / reconstruction experiment (Phase 4C).

Protocol (contract observables: docs/RESEARCH_SPEC.md "Target observables" —
retrieval/U-turn overlap diagnostics; legacy counterpart `diffusion.py`'s
`test_step`, see docs/UPSTREAM_DISCREPANCIES.md D14):

1. select a clean example — a training row (source ``"train"``) or a fresh
   draw from the same finite-F teacher (source ``"fresh"``);
2. mask each coordinate independently with probability ``t``
   (`masking.bernoulli_mask`, the canonical Bernoulli(t) corruption);
3. start the named reverse sampler from that partially observed state via
   `samplers.sample(initial_values=..., initial_mask=...)` — the sampler is
   reused as-is, not redesigned;
4. originally observed coordinates are never revised — guaranteed by the
   sampler's absorbing-state design (`samplers.py`), not by anything here;
5. only the initially masked coordinates are generated;
6. measure the retrieval overlap ``q_U(t) = mean_i[x_hat_i * x_clean_i]``
   over all N coordinates — the legacy `(xnew * x0).mean()`
   (`diffusion.py:123`).

Every reconstruction follows the sampler-indexed terminal law `P_{θ,A,k}` of
the configured sampler, conditioned on the partially observed state; the
sampler identity is recorded in every result and every artifact.

Paired seeds. For each (example_index, t) cell the mask, sampler-order, and
sampler-token generators are derived deterministically from the run's seed
hierarchy and the cell key ONLY — never from the source — so the train and
fresh cells at the same (example_index, t) see identical masks and identical
sampler randomness (a paired comparison). Seeds are derived by SHA-256
hashing of the cell key (`uturn_cell_seed`), so each cell's randomness
depends neither on sweep order nor on which other t values or sources are
present. Fresh clean draws derive per-example from `evaluation_data_seed`;
the training set is reconstructed deterministically by the caller (same
finite-F teacher, same `train_data_seed`, same `train_size`).

Memorization discipline. A train-source result alone is never labelled
memorization. The memorization-sensitive diagnostic is the paired
train-minus-fresh comparison of q_U and of nearest-training overlap, emitted
by `summarize_uturn` only when both sources were run (docs/RESEARCH_SPEC.md
claim discipline). No quantity here establishes memorization or
generalization by itself; all are finite-dimensional diagnostics.
"""

from __future__ import annotations

import hashlib
import math
import statistics
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch

from .masking import bernoulli_mask
from .metrics.overlaps import nearest_training_overlap
from .models import LinearMaskedScore
from .randomness import SeedHierarchy
from .samplers import SamplerConfig, sample
from .teacher import HiddenManifoldTeacher

UTurnSource = Literal["train", "fresh"]
UTURN_SOURCES: tuple[str, ...] = ("train", "fresh")

_SEED_DOMAIN = "maskeddiffusion.uturn.v1"


def uturn_cell_seed(purpose: str, stream_seed: int, *key_parts: object) -> int:
    """Derive a deterministic 63-bit seed for one experiment cell.

    SHA-256 over a domain separator, the `purpose` ("mask" / "order" /
    "token" / "fresh_data"), the base stream seed, and the cell key. The
    source (train/fresh) is never a key part: both sources share mask, order,
    and token seeds at the same (example_index, t) — the paired-seed design.
    """
    h = hashlib.sha256()
    h.update(_SEED_DOMAIN.encode())
    h.update(purpose.encode())
    h.update(str(stream_seed).encode())
    for part in key_parts:
        h.update(repr(part).encode())
    return int.from_bytes(h.digest()[:8], "big") >> 1


def _cell_generator(
    purpose: str, stream_seed: int, device: str | torch.device, *key_parts: object
) -> torch.Generator:
    gen = torch.Generator(device=device)
    gen.manual_seed(uturn_cell_seed(purpose, stream_seed, *key_parts))
    return gen


@dataclass(frozen=True)
class UTurnConfig:
    """Experiment grid: mask probabilities, examples per source, sources."""

    t_values: tuple[float, ...]
    n_examples: int
    sources: tuple[str, ...] = UTURN_SOURCES

    def __post_init__(self) -> None:
        if not isinstance(self.n_examples, int) or isinstance(self.n_examples, bool):
            raise TypeError(f"n_examples must be int, got {type(self.n_examples).__name__}")
        if self.n_examples < 1:
            raise ValueError(f"n_examples must be >= 1, got {self.n_examples}")
        if not self.t_values:
            raise ValueError("t_values must be non-empty")
        seen: set[float] = set()
        normalized: list[float] = []
        for t in self.t_values:
            if isinstance(t, bool) or not isinstance(t, (int, float)):
                raise TypeError(f"t values must be real numbers, got {type(t).__name__}")
            t = float(t)
            if not math.isfinite(t) or not 0.0 <= t <= 1.0:
                raise ValueError(f"t values must satisfy 0 <= t <= 1, got {t}")
            if t in seen:
                raise ValueError(f"duplicate t value {t} (cells are keyed by t)")
            seen.add(t)
            normalized.append(t)
        object.__setattr__(self, "t_values", tuple(normalized))
        if not self.sources:
            raise ValueError("sources must be non-empty")
        for s in self.sources:
            if s not in UTURN_SOURCES:
                raise ValueError(f"source {s!r} not in {list(UTURN_SOURCES)}")
        if len(set(self.sources)) != len(self.sources):
            raise ValueError(f"duplicate sources in {self.sources!r}")


@dataclass(frozen=True)
class UTurnCellResult:
    """Scalar record for one (source, example_index, t) cell."""

    source: str
    example_index: int
    t_value: float
    realized_mask_fraction: float
    n_masked: int
    q_u: float  # mean_i[x_hat_i * x_clean_i] over all N coordinates
    excess_over_baseline: float  # q_u - (1 - t), the no-recovery baseline
    hamming_error: float  # mean_i 1[x_hat_i != x_clean_i] over all N
    hamming_error_masked: float | None  # restricted to initially masked coords
    nearest_train_overlap_recon: float  # max over train rows of q(x_hat, row)
    nearest_train_overlap_clean: float  # max over train rows of q(x_clean, row)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UTurnRunResult:
    """Full output of one U-turn run (one sampler identity)."""

    sampler_identity: dict[str, Any]
    t_values: tuple[float, ...]
    n_examples: int
    sources: tuple[str, ...]
    cells: list[UTurnCellResult]
    clean: dict[str, torch.Tensor]  # source -> (n_examples, N), cpu
    masks: torch.Tensor  # (len(t_values), n_examples, N) bool, cpu — one
    # tensor because paired seeds make the mask source-independent
    reconstructions: dict[str, torch.Tensor]  # source -> (T, n_examples, N), cpu


def run_uturn(
    model: LinearMaskedScore,
    sampler: SamplerConfig,
    teacher: HiddenManifoldTeacher,
    train_set: torch.Tensor,
    uturn_config: UTurnConfig,
    seeds: SeedHierarchy,
) -> UTurnRunResult:
    """Run the U-turn protocol for every (source, example, t) cell.

    `model` must already live on its compute device; all generators are
    created on that device (torch requires generator.device == tensor.device).
    The model is set to eval mode for the duration of the run and restored.
    """
    n = model.config.visible_dim
    if teacher.dims.visible_dim != n:
        raise ValueError(f"teacher visible_dim {teacher.dims.visible_dim} != model visible_dim {n}")
    if train_set.ndim != 2 or train_set.shape[1] != n:
        raise ValueError(f"train_set must be (train_size, {n}), got {tuple(train_set.shape)}")
    if train_set.shape[0] < uturn_config.n_examples:
        raise ValueError(
            f"n_examples {uturn_config.n_examples} exceeds the reconstructed "
            f"training set size {train_set.shape[0]} — the U-turn protocol "
            "draws train-source examples from the actual training set"
        )
    device = next(model.parameters()).device
    batch = uturn_config.n_examples
    sources = uturn_config.sources
    train_set = train_set.to(device)

    clean: dict[str, torch.Tensor] = {}
    if "train" in sources:
        # The first n_examples rows of the reconstructed training set —
        # deterministic, and exactly the rows the model was trained on.
        clean["train"] = train_set[:batch].clone()
    if "fresh" in sources:
        # Fresh draws from the same finite-F law P_F: fresh z through the
        # same quenched F, one derived generator per example so the draws do
        # not depend on n_examples or on sweep order.
        rows = [
            teacher.sample_batch(
                1, _cell_generator("fresh_data", seeds.evaluation_data_seed, "cpu", i)
            )
            for i in range(batch)
        ]
        clean["fresh"] = torch.cat(rows, dim=0).to(device)

    t_list = list(uturn_config.t_values)
    masks = torch.zeros((len(t_list), batch, n), dtype=torch.bool, device=device)
    reconstructions = {s: torch.zeros((len(t_list), batch, n), device=device) for s in sources}
    cells: list[UTurnCellResult] = []

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for t_idx, t in enumerate(t_list):
                t_tensor = torch.tensor([t], dtype=torch.float32, device=device)
                for i in range(batch):
                    # The mask draw depends only on the input's shape, dtype,
                    # and device — never on its values (bernoulli_mask draws
                    # fresh uniforms). A canonical probe row keeps the mask
                    # source-independent by construction; the seed already
                    # guarantees it (paired seeds), this makes it structural.
                    probe = torch.empty((1, n), dtype=torch.float32, device=device)
                    mask_row = bernoulli_mask(
                        probe, t_tensor, _cell_generator("mask", seeds.mask_seed, device, i, t)
                    ).is_masked
                    masks[t_idx, i] = mask_row[0]
                    n_masked = int(mask_row.sum().item())
                    for source in sources:
                        clean_row = clean[source][i : i + 1]
                        # Explicitly blank the masked entries: the sampler's
                        # model forward ignores masked spin values anyway
                        # (they are multiplied by 1 - m), so this changes no
                        # logit — it makes the partially observed state
                        # explicit instead of carrying clean values in-band.
                        observed = torch.where(mask_row, torch.zeros_like(clean_row), clean_row)
                        res = sample(
                            model,
                            sampler,
                            1,
                            order_generator=_cell_generator(
                                "order", seeds.sampler_order_seed, device, i, t
                            ),
                            token_generator=_cell_generator(
                                "token", seeds.sampler_token_seed, device, i, t
                            ),
                            initial_values=observed,
                            initial_mask=mask_row,
                        )
                        recon = res.values
                        reconstructions[source][t_idx, i] = recon[0]
                        errors = recon != clean_row
                        q_u = float((recon * clean_row).mean().item())
                        cells.append(
                            UTurnCellResult(
                                source=source,
                                example_index=i,
                                t_value=t,
                                realized_mask_fraction=n_masked / n,
                                n_masked=n_masked,
                                q_u=q_u,
                                excess_over_baseline=q_u - (1.0 - t),
                                hamming_error=float(errors.float().mean().item()),
                                hamming_error_masked=(
                                    float(errors[mask_row].float().mean().item())
                                    if n_masked > 0
                                    else None
                                ),
                                nearest_train_overlap_recon=float(
                                    nearest_training_overlap(recon, train_set)[0].item()
                                ),
                                nearest_train_overlap_clean=float(
                                    nearest_training_overlap(clean_row, train_set)[0].item()
                                ),
                            )
                        )
    finally:
        if was_training:
            model.train()

    return UTurnRunResult(
        sampler_identity=sampler.identity(),
        t_values=uturn_config.t_values,
        n_examples=batch,
        sources=sources,
        cells=cells,
        clean={s: c.cpu() for s, c in clean.items()},
        masks=masks.cpu(),
        reconstructions={s: r.cpu() for s, r in reconstructions.items()},
    )


def _sem(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return statistics.stdev(values) / math.sqrt(len(values))


def summarize_uturn(result: UTurnRunResult) -> dict[str, Any]:
    """Aggregate per-example cells into per-(source, t) curve points.

    Emits the no-recovery baseline ``1 - t`` and the excess recovery
    ``q_U(t) - (1 - t)`` per point. The train-minus-fresh comparison block —
    the only memorization-sensitive diagnostic here — is emitted only when
    both sources were run; a train-source-only summary carries no
    memorization-labelled quantity at all.
    """
    points: list[dict[str, Any]] = []
    by_key: dict[tuple[str, float], dict[str, Any]] = {}
    for source in result.sources:
        for t in result.t_values:
            cells = [c for c in result.cells if c.source == source and c.t_value == t]
            q_us = [c.q_u for c in cells]
            hamming_masked = [
                c.hamming_error_masked for c in cells if c.hamming_error_masked is not None
            ]
            q_u_mean = statistics.fmean(q_us)
            point = {
                "source": source,
                "t_value": t,
                "n_examples": len(cells),
                "realized_mask_fraction_mean": statistics.fmean(
                    [c.realized_mask_fraction for c in cells]
                ),
                "q_u_mean": q_u_mean,
                "q_u_sem": _sem(q_us),
                "no_recovery_baseline": 1.0 - t,
                "excess_recovery_mean": q_u_mean - (1.0 - t),
                "hamming_error_mean": statistics.fmean([c.hamming_error for c in cells]),
                "hamming_error_masked_mean": (
                    statistics.fmean(hamming_masked) if hamming_masked else None
                ),
                "nearest_train_overlap_recon_mean": statistics.fmean(
                    [c.nearest_train_overlap_recon for c in cells]
                ),
                "nearest_train_overlap_clean_mean": statistics.fmean(
                    [c.nearest_train_overlap_clean for c in cells]
                ),
            }
            points.append(point)
            by_key[(source, t)] = point

    summary: dict[str, Any] = {
        "experiment": "uturn_reconstruction",
        "sampler": result.sampler_identity,
        "t_values": list(result.t_values),
        "n_examples": result.n_examples,
        "sources": list(result.sources),
        "points": points,
    }
    if all(s in result.sources for s in UTURN_SOURCES):
        comparison: list[dict[str, Any]] = []
        for t in result.t_values:
            p_train = by_key[("train", t)]
            p_fresh = by_key[("fresh", t)]
            comparison.append(
                {
                    "t_value": t,
                    "excess_q_u_train_minus_fresh": (p_train["q_u_mean"] - p_fresh["q_u_mean"]),
                    "excess_nearest_train_overlap_train_minus_fresh": (
                        p_train["nearest_train_overlap_recon_mean"]
                        - p_fresh["nearest_train_overlap_recon_mean"]
                    ),
                    "interpretation_note": (
                        "train-minus-fresh excess is a memorization-sensitive "
                        "diagnostic (finite-dimensional, sampler-indexed); a "
                        "train-source value alone does not establish "
                        "memorization — docs/RESEARCH_SPEC.md claim discipline"
                    ),
                }
            )
        summary["train_fresh_comparison"] = comparison
    return summary
