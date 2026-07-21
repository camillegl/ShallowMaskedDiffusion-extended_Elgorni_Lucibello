"""run_record.json → AnalysisRow parser, serialized through Phase4CRunRecord
itself (docs/PHASE4C_ANALYSIS_SPEC.md §10 rule 3 — never handwritten dicts)."""

from __future__ import annotations

import dataclasses

import pytest
from test_run_record_schema import make_record

from maskeddiffusion.analysis.ingest import discover_run_records, load_rows, record_to_row
from maskeddiffusion.analysis.rows import validate_rows
from maskeddiffusion.dimensions import Dimensions
from maskeddiffusion.experiments.schema import write_run_record
from maskeddiffusion.randomness import SeedHierarchy


def _write(tmp_path, record, name):
    """Write a record whose `run_dir` has no real linked files (no
    train/resolved_config.json, artifact manifests, or checkpoint) —
    every `load_rows` call in this module must pass `verify_hashes=False`.
    Real-file hash verification is covered in
    test_run_record_hash_verification.py and the runner integration tests.
    """
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True)
    return write_run_record(dataclasses.replace(record, run_dir=str(run_dir)), run_dir)


def test_record_to_row_maps_every_field(tmp_path):
    record = make_record(
        uturn={
            "mask_densities": [0.2, 0.5],
            "overlap": [0.9, 0.7],
            "baseline_recovery": 0.6,
            "excess_recovery": 0.15,
        }
    )
    path = _write(tmp_path, record, "run")
    row = record_to_row(record, path)
    assert row.experiment_id == "exp-a-r000-frozen_zero_v"
    assert row.pair_id == record.pair_id
    assert (row.latent_dim, row.visible_dim, row.train_size) == (8, 16, 24)
    assert row.sampler_name == "sequential_random_stochastic"
    assert row.model_config_digest == record.model_config_digest
    assert row.seeds == record.seeds
    for cmp in ("model_vs_true", "true_vs_true", "train_vs_true", "model_vs_train"):
        assert row.mmd[cmp].mixture_biased_mmd2 == record.mmd[cmp]["mixture_biased_mmd2"]
        assert row.mmd[cmp].per_lambda_biased[4.0] == record.mmd[cmp]["biased_mmd2"]["4.0"]
    assert row.nearest_training_excess == record.nearest_training["excess"]
    assert row.uturn is not None and row.uturn.excess_recovery == 0.15
    assert row.artifact_path == str(path)
    assert len(row.artifact_sha256) == 64
    assert row.record_validated is True


def test_unvalidated_record_yields_unvalidated_row_and_rejection(tmp_path):
    record = make_record(validation_problems=("stage eval: hash mismatch",))
    path = _write(tmp_path, record, "bad")
    row = record_to_row(record, path)
    assert row.record_validated is False
    result = validate_rows([row])
    assert not result.accepted
    assert any(r.rule == "unvalidated_artifact" for r in result.rejections)


def _paired_records(intervention, conditions, **common):
    a = make_record(intervention=intervention, condition=conditions[0], **common)
    b = make_record(intervention=intervention, condition=conditions[1], **common)
    return a, b


def test_engine_intervention_names_pass_pair_validation(tmp_path):
    a, b = _paired_records("v_trainability", ("frozen_zero_v", "trainable_v"))
    b = dataclasses.replace(
        b, model=dict(b.model, v_policy="trainable"), checkpoint_id="ckpt-other"
    )
    paths = [_write(tmp_path, r, r.condition) for r in (a, b)]
    rows = load_rows(sorted(tmp_path.rglob("run_record.json")), verify_hashes=False)
    result = validate_rows(rows)
    assert len(result.accepted) == 2, [r.reason for r in result.rejections]
    assert not result.rejections
    assert len(paths) == 2


def test_finite_d_pairs_allow_dimension_and_teacher_id_differences(tmp_path):
    a = make_record(intervention="finite_d", condition="d8")
    b = make_record(
        intervention="finite_d",
        condition="d16",
        dimensions=Dimensions.resolve(latent_dim=16, aspect_ratio=2.0, sample_ratio=3.0),
        teacher_id="hmt-other-shape",
        model=dict(make_record().model, visible_dim=32),
        checkpoint_id="ckpt-d16",
    )
    for r in (a, b):
        _write(tmp_path, r, r.condition)
    rows = load_rows(sorted(tmp_path.rglob("run_record.json")), verify_hashes=False)
    result = validate_rows(rows)
    # finite_d is matched_seed_finite_size: dimension and teacher_id
    # differences are expected, not rejected. model_config_digest also
    # legitimately differs (visible_dim is part of the model identity and
    # differs by construction across latent_dim) — finite_d is registered in
    # INTERVENTION_VARYING_FIELDS for exactly this field, so the pair is
    # accepted outright with no rejections at all.
    assert result.rejections == ()
    assert len(result.accepted) == 2


def test_finite_d_pairs_with_mismatched_ratios_are_rejected(tmp_path):
    a = make_record(intervention="finite_d", condition="d8")
    b = make_record(
        intervention="finite_d",
        condition="d16",
        dimensions=Dimensions.resolve(latent_dim=16, aspect_ratio=4.0, sample_ratio=3.0),
        teacher_id="hmt-other-shape",
    )
    for r in (a, b):
        _write(tmp_path, r, r.condition)
    paths = sorted(tmp_path.rglob("run_record.json"))
    result = validate_rows(load_rows(paths, verify_hashes=False))
    assert any(r.rule == "dimension_mismatch" for r in result.rejections)


def test_mixed_teacher_seed_still_rejected_for_finite_d(tmp_path):
    a = make_record(intervention="finite_d", condition="d8")
    b = make_record(
        intervention="finite_d",
        condition="d16",
        dimensions=Dimensions.resolve(latent_dim=16, aspect_ratio=2.0, sample_ratio=3.0),
        teacher_id="hmt-other-shape",
        seeds=SeedHierarchy.from_base(999),
    )
    for r in (a, b):
        _write(tmp_path, r, r.condition)
    paths = sorted(tmp_path.rglob("run_record.json"))
    result = validate_rows(load_rows(paths, verify_hashes=False))
    assert any(
        r.rule in ("mixed_teachers", "mixed_data_or_metric_seeds") for r in result.rejections
    )


def test_discover_run_records_sorted_and_dir_required(tmp_path):
    with pytest.raises(ValueError, match="not a directory"):
        discover_run_records(tmp_path / "absent")
    for name in ("b", "a"):
        _write(tmp_path, make_record(condition=f"{name}0"), name)
    found = discover_run_records(tmp_path)
    assert [p.parent.name for p in found] == ["a", "b"]
