"""Deterministic synthetic Phase 4C fixtures — development and tests only.

Every number here is invented; nothing derives from a training run. Rows are
built through the real `AnalysisRow` constructor and are expected to pass
`validate_rows` cleanly, so tests exercise the same path the (deferred)
`run_record.json` parser will feed — not handwritten approximations of it
(docs/PHASE4C_ANALYSIS_SPEC.md §0).
"""

from __future__ import annotations

import hashlib
from typing import Any, TypedDict

import numpy as np

from ..dimensions import Dimensions
from ..randomness import SeedHierarchy
from .rows import AnalysisRow, MMDSummary, UTurnSummary

_KERNEL_SCALES = (4.0, 8.0)

_CELLS = ((16, 4.0, 6.0), (16, 4.0, 12.0))  # (latent_dim, aspect_ratio, sample_ratio)


class _InterventionSpec(TypedDict):
    conditions: tuple[str, str]
    excess: dict[str, float]
    sampler: tuple[str, str]
    model_digest: dict[str, str]
    uturn: bool


# Synthetic condition effects on the Model-True excess above the floor.
# Intervention and condition labels match the engine registry
# (`maskeddiffusion.experiments.interventions.INTERVENTIONS`) exactly, so
# these fixtures exercise the same names real run records will carry.
_INTERVENTIONS: dict[str, _InterventionSpec] = {
    "v_trainability": {
        "conditions": ("frozen_zero_v", "trainable_v"),
        "excess": {"frozen_zero_v": 2.5e-2, "trainable_v": 1.8e-2},
        "sampler": ("sequential_random_stochastic", "sequential_random_stochastic"),
        "model_digest": {
            "frozen_zero_v": "modelcfg-frozen_zero",
            "trainable_v": "modelcfg-trainable",
        },
        "uturn": True,
    },
    "sampler_stochasticity": {
        "conditions": ("stochastic", "greedy"),
        "excess": {"stochastic": 2.2e-2, "greedy": 4.0e-2},
        "sampler": ("sequential_random_stochastic", "sequential_random_greedy"),
        "model_digest": {
            "stochastic": "modelcfg-baseline",
            "greedy": "modelcfg-baseline",
        },
        "uturn": False,
    },
}


def _mmd_summary(rng: np.random.Generator, center: float, *, floor: bool = False) -> MMDSummary:
    jitter = float(rng.uniform(0.9, 1.1))
    biased = center * jitter
    # Raw unbiased U-statistic: noisier, legitimately allowed to dip negative
    # near the floor (docs/RESEARCH_SPEC.md).
    unbiased = biased - float(rng.uniform(0.0, 0.4)) * (center if not floor else 1.5e-3)
    per_lam_biased = {lam: biased * f for lam, f in zip(_KERNEL_SCALES, (0.8, 1.2), strict=True)}
    per_lam_unbiased = {
        lam: unbiased * f for lam, f in zip(_KERNEL_SCALES, (0.8, 1.2), strict=True)
    }
    return MMDSummary(
        mixture_biased_mmd2=biased,
        mixture_unbiased_mmd2_raw=unbiased,
        per_lambda_biased=per_lam_biased,
        per_lambda_unbiased_raw=per_lam_unbiased,
    )


def make_synthetic_row(**overrides: Any) -> AnalysisRow:
    """One valid synthetic row; keyword overrides feed `dataclasses.replace`-style
    construction in tests (same constructor, same validation path)."""
    dims = Dimensions.resolve(latent_dim=16, aspect_ratio=4.0, sample_ratio=6.0)
    rng = np.random.default_rng(np.random.PCG64(0))
    floor = 8e-4
    defaults: dict[str, Any] = {
        "experiment_id": "p4c-v_trainability-D16-M96-frozen_zero_v-r0",
        "pair_id": "p4c-v_trainability-D16-M96",
        "repeat_id": 0,
        "latent_dim": dims.latent_dim,
        "visible_dim": dims.visible_dim,
        "train_size": dims.train_size,
        "aspect_ratio": dims.aspect_ratio,
        "sample_ratio": dims.sample_ratio,
        "visible_load": dims.visible_load,
        "intervention": "v_trainability",
        "condition": "frozen_zero_v",
        "teacher_id": "hmt-synthetic-0-v_trainability-0",
        "checkpoint_id": "ckpt-synthetic-p4c-v_trainability-D16-M96-frozen_zero_v-r0",
        "sampler_name": "sequential_random_stochastic",
        "sampler_tokens_per_step": 1,
        "model_config_digest": "modelcfg-frozen_zero",
        "seeds": SeedHierarchy.from_base(1000),
        "mmd": {
            "model_vs_true": _mmd_summary(rng, floor + 2.5e-2),
            "true_vs_true": _mmd_summary(rng, floor, floor=True),
            "train_vs_true": _mmd_summary(rng, 5e-2),
            "model_vs_train": _mmd_summary(rng, 2.0e-2),
        },
        "nearest_training_excess": 0.05,
        "nearest_training_model_mean": 0.62,
        "nearest_training_true_mean": 0.57,
        "pair_correlation_rms_error": 0.03,
        "pair_correlation_max_abs_error": 0.11,
        "uturn": None,
        "optimization_step": 1000,
        "examples_seen": 64000,
        "artifact_path": (
            "synthetic://run_records/p4c-v_trainability-D16-M96-frozen_zero_v-r0.json"
        ),
        "artifact_sha256": hashlib.sha256(b"synthetic:default-row").hexdigest(),
        "record_validated": True,
    }
    defaults.update(overrides)
    return AnalysisRow(**defaults)


def synthetic_rows(seed: int = 20260721) -> list[AnalysisRow]:
    """The full valid synthetic set: 2 interventions × 2 design cells ×
    2 conditions × 3 repeats = 24 rows. Deterministic in `seed`."""
    rng = np.random.default_rng(np.random.PCG64(seed))
    rows: list[AnalysisRow] = []
    floor = 8e-4
    for cell_idx, (latent_dim, aspect_ratio, sample_ratio) in enumerate(_CELLS):
        dims = Dimensions.resolve(
            latent_dim=latent_dim, aspect_ratio=aspect_ratio, sample_ratio=sample_ratio
        )
        for intervention, spec in _INTERVENTIONS.items():
            pair_id = f"p4c-{intervention}-D{dims.latent_dim}-M{dims.train_size}"
            for repeat_id in range(3):
                seeds = SeedHierarchy.from_base(seed + 101 * repeat_id + 7 * cell_idx)
                teacher_id = f"hmt-synthetic-{cell_idx}-{intervention}-{repeat_id}"
                for arm_idx, condition in enumerate(spec["conditions"]):
                    experiment_id = f"{pair_id}-{condition}-r{repeat_id}"
                    excess = spec["excess"][condition] * float(rng.uniform(0.9, 1.1))
                    uturn = None
                    if spec["uturn"]:
                        t_grid = tuple(float(t) for t in np.linspace(0.1, 0.9, 9))
                        start = 0.92 if arm_idx == 0 else 0.95
                        decay = 0.5 if arm_idx == 0 else 0.35
                        uturn = UTurnSummary(
                            mask_densities=t_grid,
                            overlap=tuple(
                                start - decay * float(t) + float(rng.uniform(-0.01, 0.01))
                                for t in t_grid
                            ),
                            baseline_recovery=0.60 if arm_idx == 0 else 0.62,
                            excess_recovery=0.15 if arm_idx == 0 else 0.22,
                        )
                    rows.append(
                        AnalysisRow(
                            experiment_id=experiment_id,
                            pair_id=pair_id,
                            repeat_id=repeat_id,
                            latent_dim=dims.latent_dim,
                            visible_dim=dims.visible_dim,
                            train_size=dims.train_size,
                            aspect_ratio=dims.aspect_ratio,
                            sample_ratio=dims.sample_ratio,
                            visible_load=dims.visible_load,
                            intervention=intervention,
                            condition=condition,
                            teacher_id=teacher_id,
                            checkpoint_id=f"ckpt-synthetic-{experiment_id}",
                            sampler_name=spec["sampler"][arm_idx],
                            sampler_tokens_per_step=1,
                            model_config_digest=spec["model_digest"][condition],
                            seeds=seeds,
                            mmd={
                                "model_vs_true": _mmd_summary(rng, floor + excess),
                                "true_vs_true": _mmd_summary(rng, floor, floor=True),
                                "train_vs_true": _mmd_summary(rng, 5e-2),
                                "model_vs_train": _mmd_summary(rng, 0.8 * (floor + excess)),
                            },
                            nearest_training_excess=(0.05 if arm_idx == 0 else 0.02)
                            * float(rng.uniform(0.9, 1.1)),
                            nearest_training_model_mean=0.62,
                            nearest_training_true_mean=0.57,
                            pair_correlation_rms_error=0.03 * float(rng.uniform(0.9, 1.1)),
                            pair_correlation_max_abs_error=0.11,
                            uturn=uturn,
                            optimization_step=1000,
                            examples_seen=64000,
                            artifact_path=f"synthetic://run_records/{experiment_id}.json",
                            artifact_sha256=hashlib.sha256(
                                f"synthetic:{experiment_id}".encode()
                            ).hexdigest(),
                            record_validated=True,
                        )
                    )
    return rows
