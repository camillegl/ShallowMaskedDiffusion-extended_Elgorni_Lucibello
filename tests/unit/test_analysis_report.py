"""Report-writer tests (docs/PHASE4C_ANALYSIS_SPEC.md §§2–6): exact CSV
contracts, machine-readable JSON, per-lambda and U-turn preservation, and the
wording policy applied to generated strings."""

import json

import pandas as pd

from maskeddiffusion.analysis.report import (
    AGGREGATE_CSV,
    AGGREGATE_PAIRED_CSV,
    PAIRED_CSV,
    PER_RUN_CSV,
    REPORT_SCHEMA_VERSION,
    build_report,
    check_report_wording,
    write_report,
    write_tables,
)
from maskeddiffusion.analysis.rows import PER_RUN_COLUMNS, Rejection, validate_rows
from maskeddiffusion.analysis.statistics import paired_difference_columns
from maskeddiffusion.analysis.synthetic import synthetic_rows


def _accepted():
    result = validate_rows(synthetic_rows())
    assert result.rejections == ()
    return result.accepted


def test_write_tables_exact_columns_and_counts(tmp_path):
    rows = _accepted()
    paths = write_tables(tmp_path, rows)
    assert set(paths) == {
        "per_run",
        "paired",
        "aggregate",
        "aggregate_paired",
        "matched_seed_finite_size",
    }
    assert paths["per_run"].name == PER_RUN_CSV
    assert paths["paired"].name == PAIRED_CSV
    assert paths["aggregate"].name == AGGREGATE_CSV
    assert paths["aggregate_paired"].name == AGGREGATE_PAIRED_CSV

    per_run = pd.read_csv(paths["per_run"])
    assert tuple(per_run.columns) == PER_RUN_COLUMNS
    assert len(per_run) == 24  # every individual repeat retained

    paired = pd.read_csv(paths["paired"])
    assert tuple(paired.columns) == tuple(paired_difference_columns())
    assert len(paired) == 4 * 3  # (intervention × cell) pairs × 3 repeats

    aggregate = pd.read_csv(paths["aggregate"])
    assert set(aggregate["n_repeats"]) == {3}
    # one row per (intervention, condition, cell): 2 × 2 × 2
    assert len(aggregate) == 8

    aggregate_paired = pd.read_csv(paths["aggregate_paired"])
    assert set(aggregate_paired["n_pairs"]) == {3}
    assert len(aggregate_paired) == 4


def test_report_json_structure_and_detail(tmp_path):
    rows = _accepted()
    report = build_report(rows, rejections=(), input_row_count=len(rows))
    assert report["schema_version"] == REPORT_SCHEMA_VERSION
    policy = report["statistics_policy"]
    assert policy["bootstrap_ci"] is False
    assert policy["significance_tests"] is False
    assert report["validation"]["accepted_row_count"] == 24
    assert report["records"] and len(report["records"]) == 24

    record = report["records"][0]
    assert "mmd_per_lambda" in record
    per_lam = record["mmd_per_lambda"]["model_vs_true"]
    assert set(per_lam["biased_mmd2"]) == {"4.0", "8.0"}

    uturn_records = [r for r in report["records"] if "uturn" in r]
    assert uturn_records, "U-turn detail must be preserved in the report JSON"
    curve = uturn_records[0]["uturn"]
    assert len(curve["mask_densities"]) == len(curve["overlap"]) == 9
    assert curve["excess_recovery"] > 0

    # The report is JSON-serializable with NaN rendered as null.
    blob = json.dumps(report)
    assert "NaN" not in blob
    again = json.loads(blob)
    assert again["schema_version"] == REPORT_SCHEMA_VERSION

    out = write_report(report, tmp_path / "p4c_report.json")
    assert json.loads(out.read_text())["schema_version"] == REPORT_SCHEMA_VERSION


def test_report_records_rejections_with_reasons():
    rows = list(synthetic_rows())
    rows[1] = rows[0]  # force a duplicate experiment id
    result = validate_rows(rows)
    report = build_report(result.accepted, result.rejections, input_row_count=len(rows))
    validation = report["validation"]
    assert validation["input_row_count"] == len(rows)
    assert validation["rejection_count"] >= 1
    rules = {r["rule"] for r in validation["rejections"]}
    assert "duplicate_experiment_id" in rules
    for rejection in validation["rejections"]:
        assert rejection["reason"]
        assert rejection["experiment_ids"]


def test_report_strings_respect_wording_policy():
    report = build_report(_accepted(), rejections=())
    assert check_report_wording(report) == []


def test_report_wording_guard_catches_a_violation_in_a_rejection_reason():
    report = build_report(
        _accepted(),
        rejections=[
            Rejection(
                rule="dimension_mismatch",
                reason="the sampler exhibits a phase transition at this cell",
                experiment_ids=("e-1",),
            )
        ],
    )
    assert check_report_wording(report) == ["phase transition"]


def test_report_wording_guard_walks_nested_records():
    report = build_report(_accepted(), rejections=())
    report["records"][0]["artifact_path"] = "synthetic://converges-nicely.json"
    assert check_report_wording(report) == ["converg"]
