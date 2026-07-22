"""Phase 4C analysis pipeline (docs/PHASE4C_ANALYSIS_SPEC.md).

Tidy rows, validation, statistics, report/figure/provenance writers,
deterministic synthetic fixtures, and — now that
`maskeddiffusion.experiments.schema` has landed — the `run_record.json`
parser (`analysis.ingest`) behind the `maskeddiffusion-p4c-analyze` CLI.
"""

from .ingest import discover_run_records, load_rows, record_to_row
from .rows import (
    AGGREGATED_METRICS,
    PER_RUN_COLUMNS,
    AnalysisRow,
    MMDSummary,
    Rejection,
    UTurnSummary,
    ValidationResult,
    rows_to_frame,
    validate_rows,
)
from .statistics import aggregate_paired, aggregate_repeats, paired_differences

__all__ = [
    "AGGREGATED_METRICS",
    "discover_run_records",
    "load_rows",
    "record_to_row",
    "PER_RUN_COLUMNS",
    "AnalysisRow",
    "MMDSummary",
    "Rejection",
    "UTurnSummary",
    "ValidationResult",
    "aggregate_paired",
    "aggregate_repeats",
    "paired_differences",
    "rows_to_frame",
    "validate_rows",
]
