"""Phase 4C report writers: three tidy CSVs and the machine-readable report
JSON (docs/PHASE4C_ANALYSIS_SPEC.md §§2–6).

The CSVs carry mixture-level MMD²; the report JSON additionally preserves the
per-kernel-scale MMD² values and the U-turn q_U(t) curves verbatim, so no
per-run information is lost to aggregation.
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .. import __version__
from ..checkpoints import git_metadata
from .plots import check_wording
from .rows import (
    AGGREGATED_METRICS,
    COMPARISONS,
    PER_RUN_COLUMNS,
    AnalysisRow,
    Rejection,
    rows_to_frame,
)
from .statistics import (
    aggregate_paired,
    aggregate_repeats,
    matched_seed_finite_size_frame,
    paired_differences,
)

REPORT_SCHEMA_VERSION = "maskeddiffusion.p4c_analysis.v1"

PER_RUN_CSV = "p4c_per_run.csv"
PAIRED_CSV = "p4c_paired_differences.csv"
AGGREGATE_CSV = "p4c_aggregate.csv"
AGGREGATE_PAIRED_CSV = "p4c_aggregate_paired.csv"
MATCHED_SEED_FINITE_SIZE_CSV = "p4c_matched_seed_finite_size.csv"
REPORT_JSON = "p4c_report.json"

#: Machine-checkable echo of the binding statistics policy (spec §5).
STATISTICS_POLICY: dict[str, Any] = {
    "repeats_retained": True,
    "paired_differences": (
        "computed within (pair_id, repeat_id) before any aggregation, restricted to "
        "comparison_type == paired_disorder rows; "
        "delta_b_minus_a = B - A with condition labels sorted lexicographically "
        "(a deterministic convention, not a scientific ordering)"
    ),
    "matched_seed_finite_size": (
        "comparison_type == matched_seed_finite_size rows (finite_d) are EXCLUDED "
        "from paired_differences/aggregate_paired: different latent_dim means a "
        "different-shape teacher, so a B-A delta is not a same-disorder comparison. "
        "They appear only in the separately labeled matched_seed_finite_size table, "
        "one row per run, never differenced."
    ),
    "spread": "sample standard deviation (ddof=1); empty when fewer than 2 repeats",
    "median": "supplementary only",
    "bootstrap_ci": False,
    "significance_tests": False,
    "floor": "True-True finite-sample floor reported directly per repeat, never modeled",
    "note": (
        "three repeats per cell: no bootstrap intervals and no significance-test "
        "theatre; individual repeats stay visible in every table and figure"
    ),
}


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and value != value:  # NaN -> null
        return None
    if hasattr(value, "item"):  # numpy scalar
        return value.item()
    return value


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {key: _jsonable(value) for key, value in record.items()}
        for record in frame.to_dict(orient="records")
    ]


def write_tables(
    out_dir: str | Path,
    rows: list[AnalysisRow] | tuple[AnalysisRow, ...],
    *,
    metrics: tuple[str, ...] = AGGREGATED_METRICS,
) -> dict[str, Path]:
    """Write the four tidy tables; return {role: path}. Rows must be the
    *accepted* output of `validate_rows`."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame = rows_to_frame(rows)
    paired = paired_differences(frame, metrics)
    aggregate = aggregate_repeats(frame, metrics)
    aggregate_paired_frame = aggregate_paired(paired, metrics)
    matched_seed = matched_seed_finite_size_frame(frame, metrics)

    paths = {
        "per_run": out / PER_RUN_CSV,
        "paired": out / PAIRED_CSV,
        "aggregate": out / AGGREGATE_CSV,
        "aggregate_paired": out / AGGREGATE_PAIRED_CSV,
        "matched_seed_finite_size": out / MATCHED_SEED_FINITE_SIZE_CSV,
    }
    frame.to_csv(paths["per_run"], index=False)
    paired.to_csv(paths["paired"], index=False)
    aggregate.to_csv(paths["aggregate"], index=False)
    aggregate_paired_frame.to_csv(paths["aggregate_paired"], index=False)
    matched_seed.to_csv(paths["matched_seed_finite_size"], index=False)
    return paths


def _record_detail(row: AnalysisRow) -> dict[str, Any]:
    detail = row.flat_dict()
    detail["mmd_per_lambda"] = {
        cmp: {
            "biased_mmd2": {str(lam): v for lam, v in row.mmd[cmp].per_lambda_biased.items()},
            "unbiased_mmd2_raw": {
                str(lam): v for lam, v in row.mmd[cmp].per_lambda_unbiased_raw.items()
            },
        }
        for cmp in COMPARISONS
        if cmp in row.mmd
    }
    if row.uturn is not None:
        detail["uturn"] = {
            "mask_densities": list(row.uturn.mask_densities),
            "overlap": list(row.uturn.overlap),
            "baseline_recovery": row.uturn.baseline_recovery,
            "excess_recovery": row.uturn.excess_recovery,
        }
    return detail


def build_report(
    rows: list[AnalysisRow] | tuple[AnalysisRow, ...],
    rejections: tuple[Rejection, ...] | list[Rejection],
    *,
    metrics: tuple[str, ...] = AGGREGATED_METRICS,
    input_row_count: int | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Machine-readable report (spec §6), schema
    `maskeddiffusion.p4c_analysis.v1`."""
    frame = rows_to_frame(rows)
    paired = paired_differences(frame, metrics)
    aggregate = aggregate_repeats(frame, metrics)
    aggregate_paired_frame = aggregate_paired(paired, metrics)
    matched_seed = matched_seed_finite_size_frame(frame, metrics)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": generated_at or datetime.now(UTC).isoformat(),
        "provenance": {
            "package_version": __version__,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            **git_metadata(),
        },
        "statistics_policy": STATISTICS_POLICY,
        "validation": {
            "input_row_count": input_row_count if input_row_count is not None else len(rows),
            "accepted_row_count": len(rows),
            "rejection_count": len(rejections),
            "rejections": [
                {"rule": r.rule, "reason": r.reason, "experiment_ids": list(r.experiment_ids)}
                for r in rejections
            ],
        },
        "records": [_record_detail(r) for r in rows],
        "paired_differences": _frame_records(paired),
        "aggregate": _frame_records(aggregate),
        "aggregate_paired": _frame_records(aggregate_paired_frame),
        "matched_seed_finite_size": _frame_records(matched_seed),
    }


def check_report_wording(report: dict[str, Any]) -> list[str]:
    """Recursively scan every string value in the report for forbidden
    phrases (spec §7: the wording guard covers "the report JSON's
    human-readable strings", not figures alone). Walking every string leaf —
    rather than a denylist of "human-readable" keys — means a new report
    field can never silently bypass the guard; non-prose strings (ids,
    paths, hashes) are harmless to scan since the forbidden phrases are
    specific English claims that cannot appear in them incidentally."""
    found: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, str):
            found.update(check_wording(value))
        elif isinstance(value, dict):
            for v in value.values():
                walk(v)
        elif isinstance(value, (list, tuple)):
            for v in value:
                walk(v)

    walk(report)
    return sorted(found)


def write_report(report: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    return path


def per_run_frame(rows: list[AnalysisRow] | tuple[AnalysisRow, ...]) -> pd.DataFrame:
    """Convenience passthrough (keeps PER_RUN_COLUMNS import-stable for callers)."""
    frame = rows_to_frame(rows)
    if tuple(frame.columns) != PER_RUN_COLUMNS:
        raise AssertionError("per-run frame drifted from PER_RUN_COLUMNS")
    return frame
