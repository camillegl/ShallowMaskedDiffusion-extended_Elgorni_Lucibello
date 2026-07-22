"""End-to-end Phase 4C engine integration at smoke scale (D=8, CPU).

Integration check only — none of the numbers produced here are
scientifically interpretable (docs/PHASE4C_EXPERIMENT_PROTOCOL.md)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from maskeddiffusion.analysis.ingest import discover_run_records, load_rows
from maskeddiffusion.analysis.rows import validate_rows
from maskeddiffusion.cli.analyze import main as analyze_main
from maskeddiffusion.experiments.plan import load_experiment_config
from maskeddiffusion.experiments.runner import execute_plan
from maskeddiffusion.experiments.schema import load_run_record

SMOKE_DIR = Path(__file__).resolve().parents[2] / "configs" / "experiments" / "smoke_d8"
CONFIG = SMOKE_DIR / "v_trainability.toml"
UTURN_CONFIG = SMOKE_DIR / "v_trainability_uturn.toml"
FINITE_D_CONFIG = SMOKE_DIR / "finite_d.toml"


@pytest.fixture(scope="module")
def executed(tmp_path_factory):
    out = tmp_path_factory.mktemp("experiment")
    plan = load_experiment_config(CONFIG)
    result = execute_plan(plan, out, device="cpu", command="pytest", config_path=CONFIG)
    return plan, out, result


def test_execute_plan_runs_every_condition(executed):
    plan, out, result = executed
    assert result["n_completed"] == len(plan.specs) == 2
    assert result["n_skipped"] == 0
    for spec in plan.specs:
        run_dir = plan.run_dir(out, spec)
        for name in ("run_manifest.json", "run_record.json"):
            assert (run_dir / name).exists()
        for stage in ("train", "samples", "eval"):
            assert (run_dir / stage / "manifest.json").exists()


def test_run_records_are_valid_and_condition_specific(executed):
    plan, out, _ = executed
    records = [load_run_record(p) for p in discover_run_records(out)]
    assert {r.condition for r in records} == {"frozen_zero_v", "trainable_v"}
    for record in records:
        assert record.record_validated, record.validation_problems
        assert record.intervention == "v_trainability"
        assert record.optimization_step == 4
        assert set(record.mmd) == {
            "model_vs_true",
            "true_vs_true",
            "train_vs_true",
            "model_vs_train",
        }
    frozen, trainable = sorted(records, key=lambda r: r.condition)
    # Paired disorder: same quenched teacher and data; only the intervention differs.
    assert frozen.teacher_id == trainable.teacher_id
    assert frozen.train_data_sha256 == trainable.train_data_sha256
    assert frozen.seeds == trainable.seeds
    assert frozen.model["v_policy"] == "frozen_zero"
    assert trainable.model["v_policy"] == "trainable"
    assert frozen.model_config_digest != trainable.model_config_digest


def test_pair_manifest_records_shared_disorder(executed):
    plan, out, _ = executed
    pair_dir = plan.experiment_root(out) / "smoke-d8-v-trainability-r000"
    pair_manifest = json.loads((pair_dir / "pair_manifest.json").read_text())
    assert pair_manifest["comparison_type"] == "paired_disorder"
    checks = pair_manifest["comparison_checks"]
    assert checks["teacher_id_equal"] is True and checks["passed"] is True
    assert pair_manifest["pair_validation"]["problems"] == []


def test_resume_skips_and_leaves_tree_byte_stable(executed):
    plan, out, _ = executed
    before = {str(p): p.read_bytes() for p in sorted(Path(out).rglob("*")) if p.is_file()}
    result = execute_plan(plan, out, device="cpu", command="pytest", config_path=CONFIG)
    assert result["n_completed"] == 0
    assert result["n_skipped"] == 2
    after = {str(p): p.read_bytes() for p in sorted(Path(out).rglob("*")) if p.is_file()}
    assert before == after


def test_rows_from_real_records_pass_validation(executed):
    _, out, _ = executed
    rows = load_rows(discover_run_records(out))
    result = validate_rows(rows)
    assert len(result.accepted) == 2, [r.reason for r in result.rejections]
    assert not result.rejections


def test_analyze_cli_end_to_end(executed, tmp_path):
    _, out, _ = executed
    analysis_dir = tmp_path / "analysis"
    assert analyze_main(["--records", str(out), "--output", str(analysis_dir)]) == 0
    for name in (
        "p4c_per_run.csv",
        "p4c_paired_differences.csv",
        "p4c_aggregate.csv",
        "p4c_aggregate_paired.csv",
        "p4c_report.json",
        "p4c_analysis_manifest.json",
    ):
        assert (analysis_dir / name).exists()
    report = json.loads((analysis_dir / "p4c_report.json").read_text())
    assert report["validation"]["accepted_row_count"] == 2
    assert report["validation"]["rejection_count"] == 0
    assert report["statistics_policy"]["bootstrap_ci"] is False
    manifest = json.loads((analysis_dir / "p4c_analysis_manifest.json").read_text())
    assert len(manifest["inputs"]) == 2
    assert any(entry["role"] == "figure" for entry in manifest["outputs"])
    figures = list((analysis_dir / "figures").glob("*.png"))
    assert figures


def test_analyze_cli_dry_run_writes_nothing(executed, tmp_path, capsys):
    _, out, _ = executed
    analysis_dir = tmp_path / "dry"
    assert analyze_main(["--records", str(out), "--output", str(analysis_dir), "--dry-run"]) == 0
    assert not analysis_dir.exists()
    printed = json.loads(capsys.readouterr().out)
    assert printed["n_records"] == 2


@pytest.fixture(scope="module")
def executed_with_uturn(tmp_path_factory):
    out = tmp_path_factory.mktemp("experiment_uturn")
    plan = load_experiment_config(UTURN_CONFIG)
    result = execute_plan(plan, out, device="cpu", command="pytest", config_path=UTURN_CONFIG)
    return plan, out, result


def test_uturn_stage_artifact_and_record_block(executed_with_uturn):
    plan, out, result = executed_with_uturn
    assert result["n_completed"] == len(plan.specs) == 2
    for spec in plan.specs:
        run_dir = plan.run_dir(out, spec)
        assert (run_dir / "uturn" / "manifest.json").exists()
        assert (run_dir / "uturn" / "summary.json").exists()
        record = load_run_record(run_dir / "run_record.json")
        assert record.record_validated, record.validation_problems
        assert record.uturn is not None
        assert record.uturn["mask_densities"] == [0.2, 0.5, 0.8]
        assert len(record.uturn["overlap"]) == 3
        assert isinstance(record.uturn["baseline_recovery"], float)
        assert isinstance(record.uturn["excess_recovery"], float)


def test_uturn_resume_skips_and_leaves_uturn_stage_byte_stable(executed_with_uturn):
    plan, out, _ = executed_with_uturn
    before = {str(p): p.read_bytes() for p in sorted(Path(out).rglob("*")) if p.is_file()}
    result = execute_plan(plan, out, device="cpu", command="pytest", config_path=UTURN_CONFIG)
    assert result["n_completed"] == 0
    assert result["n_skipped"] == 2
    after = {str(p): p.read_bytes() for p in sorted(Path(out).rglob("*")) if p.is_file()}
    assert before == after


def test_analyze_cli_produces_uturn_figure(executed_with_uturn, tmp_path):
    _, out, _ = executed_with_uturn
    analysis_dir = tmp_path / "uturn_analysis"
    assert analyze_main(["--records", str(out), "--output", str(analysis_dir)]) == 0
    figures = list((analysis_dir / "figures").glob("p4c_fig_uturn_*.png"))
    assert figures
    report = json.loads((analysis_dir / "p4c_report.json").read_text())
    assert report["validation"]["accepted_row_count"] == 2
    for record in report["records"]:
        assert record["uturn"] is not None
        assert record["uturn"]["mask_densities"] == [0.2, 0.5, 0.8]


def test_uturn_artifact_hash_is_checked(executed_with_uturn):
    """load_run_record (verify_hashes=True default) recomputes the uturn
    stage's manifest.json SHA-256 and catches tampering, exactly like the
    required train/samples/eval stages."""
    plan, out, _ = executed_with_uturn
    spec = plan.specs[0]
    run_dir = plan.run_dir(out, spec)
    uturn_manifest = run_dir / "uturn" / "manifest.json"
    original = uturn_manifest.read_bytes()
    try:
        uturn_manifest.write_bytes(b'{"tampered": true}')
        with pytest.raises(ValueError, match="failed artifact hash verification"):
            load_run_record(run_dir / "run_record.json")
    finally:
        uturn_manifest.write_bytes(original)
    # Restored: verification passes again (sanity that the test didn't
    # leave the fixture directory corrupted for other tests).
    load_run_record(run_dir / "run_record.json")


def test_stage_manifest_tampering_is_rejected_end_to_end(executed):
    """A hash mismatch on any required stage's own manifest.json is caught
    by load_run_record — real files, real tampering, real rejection."""
    plan, out, _ = executed
    spec = plan.specs[0]
    run_dir = plan.run_dir(out, spec)
    eval_manifest = run_dir / "eval" / "manifest.json"
    original = eval_manifest.read_bytes()
    try:
        eval_manifest.write_bytes(original[:-1] + b" ")  # trivial byte flip
        with pytest.raises(ValueError, match="failed artifact hash verification"):
            load_run_record(run_dir / "run_record.json")
    finally:
        eval_manifest.write_bytes(original)


def test_backfilled_record_carries_migration_and_is_then_immutable(executed):
    """Deleting a completed run's run_record.json and resuming rebuilds it
    exactly once with an explicit migration block; a second resume leaves
    it byte-identical (true immutability, not silent rebuild-and-compare)."""
    plan, out, _ = executed
    spec = plan.specs[0]
    run_dir = plan.run_dir(out, spec)
    record_path = run_dir / "run_record.json"

    original = load_run_record(record_path)
    assert original.migration is None

    record_path.unlink()
    result = execute_plan(plan, out, device="cpu", command="pytest", config_path=CONFIG)
    assert result["n_completed"] == 0 and result["n_skipped"] == len(plan.specs)

    backfilled = load_run_record(record_path)
    assert backfilled.migration is not None
    assert backfilled.migration["from_schema"] is None
    assert backfilled.migration["source_artifacts_unchanged"] is True
    assert backfilled == dataclasses.replace(original, migration=backfilled.migration)

    before_bytes = record_path.read_bytes()
    execute_plan(plan, out, device="cpu", command="pytest", config_path=CONFIG)
    assert record_path.read_bytes() == before_bytes


@pytest.fixture(scope="module")
def executed_finite_d(tmp_path_factory):
    out = tmp_path_factory.mktemp("experiment_finite_d")
    plan = load_experiment_config(FINITE_D_CONFIG)
    result = execute_plan(plan, out, device="cpu", command="pytest", config_path=FINITE_D_CONFIG)
    return plan, out, result


def test_finite_d_pair_manifest_is_matched_seed_not_paired_disorder(executed_finite_d):
    plan, out, result = executed_finite_d
    assert result["n_completed"] == len(plan.specs)
    pair_dir = plan.experiment_root(out) / "smoke-d8-finite-d-r000"
    pair_manifest = json.loads((pair_dir / "pair_manifest.json").read_text())
    assert pair_manifest["comparison_type"] == "matched_seed_finite_size"
    assert "comparison_type" in pair_manifest and pair_manifest["comparison_type"] != (
        "paired_disorder"
    )
    checks = pair_manifest["comparison_checks"]
    assert checks["distinct_teacher_id"] is True
    assert checks["same_seed_hierarchy_values"] is True
    assert checks["same_ratios"] is True
    assert checks["distinct_dimensions"] is True
    assert checks["passed"] is True
    # A matched_seed_finite_size group has no teacher_id_equal key at all —
    # it is structurally impossible to confuse with a paired_disorder block.
    assert "teacher_id_equal" not in checks


def test_finite_d_excluded_from_paired_differences_by_default(executed_finite_d):
    _, out, _ = executed_finite_d
    rows = load_rows(discover_run_records(out))
    result = validate_rows(rows)
    assert len(result.accepted) == 2, [r.reason for r in result.rejections]
    from maskeddiffusion.analysis.rows import rows_to_frame
    from maskeddiffusion.analysis.statistics import (
        matched_seed_finite_size_frame,
        paired_differences,
    )

    frame = rows_to_frame(result.accepted)
    paired = paired_differences(frame)
    assert paired.empty
    matched_seed = matched_seed_finite_size_frame(frame)
    assert len(matched_seed) == 2
    assert set(matched_seed["comparison_type"]) == {"matched_seed_finite_size"}
    assert len(set(matched_seed["teacher_id"])) == 2  # distinct per row
