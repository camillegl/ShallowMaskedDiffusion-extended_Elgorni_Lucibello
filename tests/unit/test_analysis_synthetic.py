"""Synthetic-fixture tests: determinism, internal plausibility of the invented
numbers, and drift protection between the code contract and
docs/PHASE4C_ANALYSIS_SPEC.md."""

from pathlib import Path

from maskeddiffusion.analysis.rows import (
    AGGREGATED_METRICS,
    COMPARISONS,
    PER_RUN_COLUMNS,
    validate_rows,
)
from maskeddiffusion.analysis.synthetic import synthetic_rows

SPEC = Path(__file__).resolve().parents[2] / "docs" / "PHASE4C_ANALYSIS_SPEC.md"


def test_synthetic_rows_deterministic():
    first = [r.flat_dict() for r in synthetic_rows()]
    second = [r.flat_dict() for r in synthetic_rows()]
    assert first == second


def test_synthetic_set_shape_and_validity():
    rows = synthetic_rows()
    assert len(rows) == 24  # 2 interventions × 2 cells × 2 conditions × 3 repeats
    assert len({r.experiment_id for r in rows}) == 24
    result = validate_rows(rows)
    assert result.rejections == ()
    assert len(result.accepted) == 24


def test_synthetic_numbers_internally_plausible():
    for row in synthetic_rows():
        floor = row.mmd["true_vs_true"].mixture_biased_mmd2
        model = row.mmd["model_vs_true"].mixture_biased_mmd2
        assert 0 < floor < model  # synthetic design: model sits above the floor
        for cmp in COMPARISONS:
            summary = row.mmd[cmp]
            assert summary.mixture_biased_mmd2 > 0
            # mixture equals the uniform mean of the two kernel scales
            mean = sum(summary.per_lambda_biased.values()) / len(summary.per_lambda_biased)
            assert abs(summary.mixture_biased_mmd2 - mean) < 1e-12
        if row.intervention == "v_trainability":
            assert row.uturn is not None
        else:
            assert row.uturn is None


def test_every_spec_column_name_documented():
    text = SPEC.read_text()
    missing = [c for c in PER_RUN_COLUMNS if c not in text]
    assert missing == [], f"columns missing from the spec: {missing}"
    missing_metrics = [m for m in AGGREGATED_METRICS if m not in text]
    assert missing_metrics == []


def test_spec_documents_binding_policies():
    text = SPEC.read_text()
    assert "maskeddiffusion.p4c_analysis.v1" in text
    assert "sample standard deviation" in text
    assert "No bootstrap confidence intervals" in text
    assert "p4c_per_run.csv" in text
    assert "p4c_paired_differences.csv" in text
    assert "p4c_aggregate.csv" in text
    assert "p4c_report.json" in text
    assert "p4c_analysis_manifest.json" in text
