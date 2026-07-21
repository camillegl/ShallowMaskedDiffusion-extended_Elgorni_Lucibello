"""Phase 4C analysis: tidy-row contract and structural validation.

The authoritative contract is `docs/PHASE4C_ANALYSIS_SPEC.md`. One `AnalysisRow`
corresponds to one completed, validated engine run record (`Phase4CRunRecord`,
written by `maskeddiffusion.experiments.schema`). The parser from
`run_record.json` to `AnalysisRow` lives in `analysis.ingest`; this module
defines the parser's target and the validation every row must pass before
any statistics, table, or figure is produced (spec §9).

Rejection philosophy: a rejected row is excluded from every downstream table and
figure, and every rejection is reported with rule, reason, and affected
experiment ids. Nothing is dropped silently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..dimensions import Dimensions
from ..experiments.schema import COMPARISONS
from ..randomness import SeedHierarchy

# Imported, not redefined, from experiments.schema — Phase4CRunRecord.mmd is
# the single source of truth for which comparisons exist; a second
# independent tuple here would be exactly the kind of duplicated field
# registry the schema is meant to prevent. Tidy column prefixes (spec §1):
COMPARISON_PREFIX: dict[str, str] = {
    "model_vs_true": "model_true",
    "true_vs_true": "true_true",
    "train_vs_true": "train_true",
    "model_vs_train": "model_train",
}

MMD_MIXTURE_COLUMNS: tuple[str, ...] = tuple(
    f"{COMPARISON_PREFIX[cmp]}_mmd2_{stat}"
    for cmp in COMPARISONS
    for stat in ("biased", "unbiased_raw")
)

#: Metrics carried through paired differencing and aggregation (spec §5).
AGGREGATED_METRICS: tuple[str, ...] = MMD_MIXTURE_COLUMNS + (
    "nearest_training_excess",
    "pair_correlation_rms_error",
)

#: Optional U-turn scalar summaries (curves live in the report JSON only).
UTURN_SCALAR_COLUMNS: tuple[str, ...] = ("uturn_baseline_recovery", "uturn_excess_recovery")

SEED_COLUMNS: tuple[str, ...] = (
    "base_seed",
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

#: Exact column order of `p4c_per_run.csv` (spec §2).
PER_RUN_COLUMNS: tuple[str, ...] = (
    "experiment_id",
    "pair_id",
    "repeat_id",
    "latent_dim",
    "visible_dim",
    "train_size",
    "aspect_ratio",
    "sample_ratio",
    "visible_load",
    "intervention",
    "condition",
    "teacher_id",
    "checkpoint_id",
    "sampler_name",
    "sampler_tokens_per_step",
    "model_config_digest",
    *SEED_COLUMNS,
    *MMD_MIXTURE_COLUMNS,
    "nearest_training_excess",
    "nearest_training_model_mean",
    "nearest_training_true_mean",
    "pair_correlation_rms_error",
    "pair_correlation_max_abs_error",
    *UTURN_SCALAR_COLUMNS,
    "optimization_step",
    "examples_seen",
    "artifact_path",
    "artifact_sha256",
    "record_validated",
)

#: Fields that must be identical inside a matched pair, keyed by the
#: intervention that legitimately varies them (spec §9, rule 7). Unknown
#: interventions are strict: nothing may differ. Keys are exactly the
#: engine's registered intervention names
#: (`maskeddiffusion.experiments.interventions.INTERVENTIONS`). `finite_d`
#: legitimately varies `model_config_digest` because `visible_dim` — part of
#: the model identity — necessarily differs across `latent_dim` by
#: construction (`Dimensions.resolve`), not because of any intervened model
#: field.
INTERVENTION_VARYING_FIELDS: dict[str, tuple[str, ...]] = {
    "sampler_stochasticity": ("sampler_name",),
    "v_trainability": ("model_config_digest",),
    "optimization_budget": (),
    "finite_d": ("model_config_digest",),
}

#: Interventions whose `comparison_type` (`maskeddiffusion.experiments.
#: interventions.INTERVENTIONS`) is `"matched_seed_finite_size"`, not
#: `"paired_disorder"`: arms differ in realized dimensions and therefore in
#: `teacher_id` (a teacher of a different shape is necessarily a distinct
#: draw) — DISTINCT `teacher_id` is required, not merely tolerated (rule 4
#: below). Arms still share every seed VALUE and the aspect/sample ratios,
#: but this is never the same kind of comparison as a `"paired_disorder"`
#: group: `analysis.statistics.paired_differences` excludes these rows by
#: default (`docs/PHASE4C_EXPERIMENT_PROTOCOL.md` §2).
MATCHED_SEED_FINITE_SIZE_INTERVENTIONS: frozenset[str] = frozenset({"finite_d"})


def comparison_type(intervention: str) -> str:
    """`"matched_seed_finite_size"` or `"paired_disorder"` for a row's
    `intervention` name — the same classification the engine registry uses
    (`maskeddiffusion.experiments.interventions.Intervention.comparison_type`),
    duplicated here as a name-keyed lookup because the analysis layer never
    imports the engine (artifact-level truth lives only in
    `Phase4CRunRecord`; this module classifies by the name already carried
    on every row)."""
    return (
        "matched_seed_finite_size"
        if intervention in MATCHED_SEED_FINITE_SIZE_INTERVENTIONS
        else "paired_disorder"
    )


#: Dimension fields still required to match inside a dimension-varying pair.
RATIO_FIELDS: tuple[str, ...] = ("aspect_ratio", "sample_ratio")

#: Identity fields checked inside a pair unless the intervention varies them.
IDENTITY_FIELDS: tuple[str, ...] = (
    "sampler_name",
    "sampler_tokens_per_step",
    "model_config_digest",
)

DIMENSION_FIELDS: tuple[str, ...] = (
    "latent_dim",
    "visible_dim",
    "train_size",
    "aspect_ratio",
    "sample_ratio",
    "visible_load",
)

#: Data and metric seeds that must match inside a pair (spec §9, rule 5).
DATA_METRIC_SEED_FIELDS: tuple[str, ...] = (
    "train_data_seed",
    "validation_data_seed",
    "evaluation_data_seed",
    "metric_seed",
)


@dataclass(frozen=True)
class MMDSummary:
    """Mixture-level MMD² for one comparison, with per-kernel-scale detail.

    `mixture_unbiased_mmd2_raw` and `per_lambda_unbiased_raw` may legitimately
    be negative (finite-sample U-statistic, docs/RESEARCH_SPEC.md); they are
    stored raw and never clipped.
    """

    mixture_biased_mmd2: float
    mixture_unbiased_mmd2_raw: float
    per_lambda_biased: dict[float, float] = field(default_factory=dict)
    per_lambda_unbiased_raw: dict[float, float] = field(default_factory=dict)


@dataclass(frozen=True)
class UTurnSummary:
    """Optional U-turn retrieval diagnostic: q_U(t) plus scalar recoveries."""

    mask_densities: tuple[float, ...]
    overlap: tuple[float, ...]
    baseline_recovery: float
    excess_recovery: float

    def __post_init__(self) -> None:
        if len(self.mask_densities) != len(self.overlap):
            raise ValueError(
                f"mask_densities and overlap must have equal length, got "
                f"{len(self.mask_densities)} vs {len(self.overlap)}"
            )
        if not self.mask_densities:
            raise ValueError("UTurnSummary requires at least one (t, q_U) point")


@dataclass(frozen=True)
class AnalysisRow:
    """One tidy per-run row: the parser target for `Phase4CRunRecord` (spec §2).

    This is the analysis layer's tidy-table contract only — it is not an
    artifact format and must never be serialized back into a run record.
    """

    experiment_id: str
    pair_id: str
    repeat_id: int
    latent_dim: int
    visible_dim: int
    train_size: int
    aspect_ratio: float
    sample_ratio: float
    visible_load: float
    intervention: str
    condition: str
    teacher_id: str
    checkpoint_id: str
    sampler_name: str
    sampler_tokens_per_step: int
    model_config_digest: str
    seeds: SeedHierarchy
    mmd: dict[str, MMDSummary]  # keyed by COMPARISONS names
    nearest_training_excess: float
    nearest_training_model_mean: float
    nearest_training_true_mean: float
    pair_correlation_rms_error: float
    pair_correlation_max_abs_error: float
    uturn: UTurnSummary | None
    optimization_step: int
    examples_seen: int
    artifact_path: str
    artifact_sha256: str
    record_validated: bool

    def __post_init__(self) -> None:
        if not self.experiment_id:
            raise ValueError("experiment_id must be nonempty")
        if not self.pair_id:
            raise ValueError("pair_id must be nonempty")
        if self.repeat_id < 0:
            raise ValueError(f"repeat_id must be >= 0, got {self.repeat_id}")
        if not self.condition:
            raise ValueError("condition must be nonempty")
        if self.sampler_tokens_per_step < 1:
            raise ValueError(
                f"sampler_tokens_per_step must be >= 1, got {self.sampler_tokens_per_step}"
            )

    def flat_dict(self) -> dict[str, Any]:
        """Flat one-row mapping with exactly PER_RUN_COLUMNS keys."""
        d: dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "pair_id": self.pair_id,
            "repeat_id": self.repeat_id,
            "latent_dim": self.latent_dim,
            "visible_dim": self.visible_dim,
            "train_size": self.train_size,
            "aspect_ratio": self.aspect_ratio,
            "sample_ratio": self.sample_ratio,
            "visible_load": self.visible_load,
            "intervention": self.intervention,
            "condition": self.condition,
            "teacher_id": self.teacher_id,
            "checkpoint_id": self.checkpoint_id,
            "sampler_name": self.sampler_name,
            "sampler_tokens_per_step": self.sampler_tokens_per_step,
            "model_config_digest": self.model_config_digest,
        }
        seed_dict = self.seeds.to_dict()
        for col in SEED_COLUMNS:
            d[col] = seed_dict.get(col)
        for cmp in COMPARISONS:
            summary = self.mmd.get(cmp)
            prefix = COMPARISON_PREFIX[cmp]
            d[f"{prefix}_mmd2_biased"] = (
                summary.mixture_biased_mmd2 if summary is not None else None
            )
            d[f"{prefix}_mmd2_unbiased_raw"] = (
                summary.mixture_unbiased_mmd2_raw if summary is not None else None
            )
        d.update(
            {
                "nearest_training_excess": self.nearest_training_excess,
                "nearest_training_model_mean": self.nearest_training_model_mean,
                "nearest_training_true_mean": self.nearest_training_true_mean,
                "pair_correlation_rms_error": self.pair_correlation_rms_error,
                "pair_correlation_max_abs_error": self.pair_correlation_max_abs_error,
                "uturn_baseline_recovery": (
                    self.uturn.baseline_recovery if self.uturn is not None else None
                ),
                "uturn_excess_recovery": (
                    self.uturn.excess_recovery if self.uturn is not None else None
                ),
                "optimization_step": self.optimization_step,
                "examples_seen": self.examples_seen,
                "artifact_path": self.artifact_path,
                "artifact_sha256": self.artifact_sha256,
                "record_validated": self.record_validated,
            }
        )
        if tuple(d.keys()) != PER_RUN_COLUMNS:
            missing = set(PER_RUN_COLUMNS) - set(d)
            extra = set(d) - set(PER_RUN_COLUMNS)
            raise AssertionError(f"flat_dict drifted from PER_RUN_COLUMNS: {missing=} {extra=}")
        return d


@dataclass(frozen=True)
class Rejection:
    """One validation rejection (spec §9); reported, never silent."""

    rule: str
    reason: str
    experiment_ids: tuple[str, ...]


@dataclass(frozen=True)
class ValidationResult:
    accepted: tuple[AnalysisRow, ...]
    rejections: tuple[Rejection, ...]


def _is_finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def _row_problems(row: AnalysisRow) -> list[tuple[str, str]]:
    """Per-row checks (spec §9 rules 1, 3-per-row, 8)."""
    problems: list[tuple[str, str]] = []
    if not row.record_validated:
        problems.append(
            (
                "unvalidated_artifact",
                f"row from unvalidated record {row.artifact_path!r}",
            )
        )
    try:
        dims = Dimensions.resolve(row.latent_dim, row.aspect_ratio, row.sample_ratio)
        if dims.visible_dim != row.visible_dim or dims.train_size != row.train_size:
            problems.append(
                (
                    "dimension_mismatch",
                    f"stored (visible_dim={row.visible_dim}, train_size={row.train_size}) "
                    f"disagrees with resolved ({dims.visible_dim}, {dims.train_size})",
                )
            )
        elif abs(row.visible_load - row.train_size / row.visible_dim) > 1e-12:
            problems.append(
                (
                    "dimension_mismatch",
                    f"visible_load {row.visible_load} != train_size/visible_dim "
                    f"{row.train_size / row.visible_dim}",
                )
            )
    except (TypeError, ValueError) as e:
        problems.append(("dimension_mismatch", f"dimensions fail to resolve: {e}"))
    for cmp in COMPARISONS:
        summary = row.mmd.get(cmp)
        if summary is None:
            problems.append(("missing_metric", f"missing MMD comparison {cmp!r}"))
            continue
        if not _is_finite(summary.mixture_biased_mmd2):
            problems.append(("nonfinite_metric", f"{cmp} mixture_biased_mmd2 not finite"))
        if not _is_finite(summary.mixture_unbiased_mmd2_raw):
            problems.append(("nonfinite_metric", f"{cmp} mixture_unbiased_mmd2_raw not finite"))
    for name, value in (
        ("nearest_training_excess", row.nearest_training_excess),
        ("nearest_training_model_mean", row.nearest_training_model_mean),
        ("nearest_training_true_mean", row.nearest_training_true_mean),
        ("pair_correlation_rms_error", row.pair_correlation_rms_error),
        ("pair_correlation_max_abs_error", row.pair_correlation_max_abs_error),
    ):
        if not _is_finite(value):
            problems.append(("nonfinite_metric", f"{name} not finite"))
    return problems


def validate_rows(rows: list[AnalysisRow] | tuple[AnalysisRow, ...]) -> ValidationResult:
    """Apply every rejection rule of spec §9; return accepted rows + rejections."""
    rejections: list[Rejection] = []
    rejected_ids: set[str] = set()

    def reject(rule: str, reason: str, doomed: list[AnalysisRow]) -> None:
        rejections.append(
            Rejection(
                rule=rule, reason=reason, experiment_ids=tuple(r.experiment_id for r in doomed)
            )
        )
        rejected_ids.update(r.experiment_id for r in doomed)

    # Rule 1/3-per-row/8: per-row checks.
    survivors: list[AnalysisRow] = []
    for row in rows:
        problems = _row_problems(row)
        if problems:
            for rule, reason in problems:
                reject(rule, reason, [row])
        else:
            survivors.append(row)

    # Rule 2: duplicate experiment ids — first (lexicographic) occurrence kept.
    by_eid: dict[str, list[AnalysisRow]] = {}
    for row in survivors:
        by_eid.setdefault(row.experiment_id, []).append(row)
    survivors = []
    for eid in sorted(by_eid):
        group = by_eid[eid]
        survivors.append(group[0])
        if len(group) > 1:
            reject(
                "duplicate_experiment_id",
                f"experiment_id {eid!r} appears {len(group)} times; first occurrence kept",
                group[1:],
            )

    # Pair-structure checks (spec §9 rules 3–7).
    by_pair: dict[str, list[AnalysisRow]] = {}
    for row in survivors:
        by_pair.setdefault(row.pair_id, []).append(row)

    pair_ok: list[AnalysisRow] = []
    for pair_id in sorted(by_pair):
        pair_rows = by_pair[pair_id]
        interventions = {r.intervention for r in pair_rows}
        if len(interventions) != 1:
            reject(
                "unmatched_conditions",
                f"pair {pair_id!r} mixes interventions {sorted(interventions)}",
                pair_rows,
            )
            continue
        intervention = pair_rows[0].intervention
        # Each condition must map to exactly one identity across repeats.
        identity_by_condition: dict[str, set[tuple[Any, ...]]] = {}
        for r in pair_rows:
            key = (r.sampler_name, r.sampler_tokens_per_step, r.model_config_digest)
            identity_by_condition.setdefault(r.condition, set()).add(key)
        bad_conditions = sorted(c for c, ids in identity_by_condition.items() if len(ids) > 1)
        if bad_conditions:
            reject(
                "unmatched_conditions",
                f"pair {pair_id!r} condition(s) {bad_conditions} map to more than one "
                "sampler identity / model config across repeats",
                pair_rows,
            )
            continue

        by_repeat: dict[int, list[AnalysisRow]] = {}
        for r in pair_rows:
            by_repeat.setdefault(r.repeat_id, []).append(r)
        for repeat_id in sorted(by_repeat):
            arms = by_repeat[repeat_id]
            conditions = sorted({r.condition for r in arms})
            if len(arms) != 2 or len(conditions) != 2:
                reject(
                    "unmatched_conditions",
                    f"pair {pair_id!r} repeat {repeat_id} has {len(arms)} row(s) with "
                    f"conditions {conditions}; expected exactly two distinct conditions",
                    arms,
                )
                continue
            a, b = arms[0], arms[1]
            dim_varying = intervention in MATCHED_SEED_FINITE_SIZE_INTERVENTIONS
            if a.seeds.teacher_seed != b.seeds.teacher_seed or (
                not dim_varying and a.teacher_id != b.teacher_id
            ):
                reject(
                    "mixed_teachers",
                    f"pair {pair_id!r} repeat {repeat_id} mixes teachers "
                    f"({a.teacher_id!r}/{a.seeds.teacher_seed} vs "
                    f"{b.teacher_id!r}/{b.seeds.teacher_seed})",
                    arms,
                )
                continue
            if dim_varying and a.teacher_id == b.teacher_id:
                # matched_seed_finite_size REQUIRES distinct teacher_id: a
                # different latent_dim is a different-shape teacher, so equal
                # teacher_id here means the arms were not actually at
                # different dimensions — a real bug, not a valid comparison
                # (docs/PHASE4C_EXPERIMENT_PROTOCOL.md, comparison_type).
                reject(
                    "finite_size_teacher_collision",
                    f"pair {pair_id!r} repeat {repeat_id} is matched_seed_finite_size "
                    f"but both arms have teacher_id {a.teacher_id!r} — different "
                    "latent_dim must produce a distinct-shape teacher",
                    arms,
                )
                continue
            seed_diffs = [
                s for s in DATA_METRIC_SEED_FIELDS if getattr(a.seeds, s) != getattr(b.seeds, s)
            ]
            if seed_diffs:
                reject(
                    "mixed_data_or_metric_seeds",
                    f"pair {pair_id!r} repeat {repeat_id} differs in {seed_diffs}",
                    arms,
                )
                continue
            dim_fields = RATIO_FIELDS if dim_varying else DIMENSION_FIELDS
            dim_diffs = [f for f in dim_fields if getattr(a, f) != getattr(b, f)]
            if dim_diffs:
                reject(
                    "dimension_mismatch",
                    f"pair {pair_id!r} repeat {repeat_id} differs in dimensions {dim_diffs}",
                    arms,
                )
                continue
            varying = INTERVENTION_VARYING_FIELDS.get(intervention, ())
            id_diffs = [
                f for f in IDENTITY_FIELDS if f not in varying and getattr(a, f) != getattr(b, f)
            ]
            if id_diffs:
                reject(
                    "identity_mismatch",
                    f"pair {pair_id!r} repeat {repeat_id} differs in {id_diffs} although "
                    f"intervention {intervention!r} may only vary {varying or 'nothing'}",
                    arms,
                )
                continue
            pair_ok.extend(arms)

    accepted = tuple(r for r in pair_ok if r.experiment_id not in rejected_ids)
    return ValidationResult(accepted=accepted, rejections=tuple(rejections))


def rows_to_frame(rows: list[AnalysisRow] | tuple[AnalysisRow, ...]) -> pd.DataFrame:
    """Tidy per-run frame with exactly PER_RUN_COLUMNS (spec §2)."""
    frame = pd.DataFrame([r.flat_dict() for r in rows], columns=list(PER_RUN_COLUMNS))
    return frame
