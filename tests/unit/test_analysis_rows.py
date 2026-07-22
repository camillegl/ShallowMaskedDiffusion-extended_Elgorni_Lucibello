"""Validation-rule tests (docs/PHASE4C_ANALYSIS_SPEC.md §9).

All fixtures are synthetic rows built through the real AnalysisRow constructor
(maskeddiffusion.analysis.synthetic) — no handwritten record approximations.
"""

import dataclasses

import pytest

from maskeddiffusion.analysis.rows import (
    COMPARISONS,
    PER_RUN_COLUMNS,
    MMDSummary,
    rows_to_frame,
    validate_rows,
)
from maskeddiffusion.analysis.synthetic import make_synthetic_row, synthetic_rows


def _pair(**b_overrides):
    """A valid two-arm pair (v_trainability: frozen_zero_v vs trainable_v)."""
    a = make_synthetic_row(
        experiment_id="e-a",
        condition="frozen_zero_v",
        model_config_digest="modelcfg-frozen_zero",
    )
    b_kwargs = {
        "experiment_id": "e-b",
        "condition": "trainable_v",
        "model_config_digest": "modelcfg-trainable",
        "checkpoint_id": "ckpt-synthetic-e-b",
    }
    b_kwargs.update(b_overrides)
    b = make_synthetic_row(**b_kwargs)
    return [a, b]


def _rules(result):
    return [r.rule for r in result.rejections]


def test_synthetic_set_validates_clean():
    result = validate_rows(synthetic_rows())
    assert result.rejections == ()
    assert len(result.accepted) == 24


def test_flat_dict_matches_per_run_columns():
    row = make_synthetic_row()
    assert tuple(row.flat_dict().keys()) == PER_RUN_COLUMNS
    frame = rows_to_frame([row])
    assert tuple(frame.columns) == PER_RUN_COLUMNS
    assert len(frame) == 1


def test_unvalidated_artifact_rejected():
    rows = _pair()
    rows[0] = dataclasses.replace(rows[0], record_validated=False)
    result = validate_rows(rows)
    assert "unvalidated_artifact" in _rules(result)
    # Both rows of the pair leave the accepted set: the survivor has no partner.
    assert result.accepted == ()


def test_duplicate_experiment_id_rejected_first_kept():
    a = make_synthetic_row(experiment_id="dup", condition="frozen_zero_v")
    b = dataclasses.replace(a, teacher_id="hmt-other")
    result = validate_rows([a, b])
    assert "duplicate_experiment_id" in _rules(result)
    kept = [r for r in result.accepted]
    assert all(r.teacher_id != "hmt-other" for r in kept)


def test_row_level_dimension_mismatch_rejected():
    rows = _pair()
    rows[0] = dataclasses.replace(rows[0], visible_dim=rows[0].visible_dim + 1)
    result = validate_rows(rows)
    assert "dimension_mismatch" in _rules(result)
    assert result.accepted == ()


def test_pair_level_dimension_mismatch_rejected():
    rows = _pair(train_size=192, sample_ratio=12.0, visible_load=192 / 64)
    result = validate_rows(rows)
    assert "dimension_mismatch" in _rules(result)
    assert result.accepted == ()


def test_mixed_teachers_rejected():
    rows = _pair(teacher_id="hmt-synthetic-other")
    result = validate_rows(rows)
    assert "mixed_teachers" in _rules(result)
    assert result.accepted == ()


def test_mixed_teacher_seed_rejected():
    rows = _pair()
    rows[1] = dataclasses.replace(
        rows[1], seeds=dataclasses.replace(rows[1].seeds, teacher_seed=999999)
    )
    result = validate_rows(rows)
    assert "mixed_teachers" in _rules(result)


def test_mixed_data_or_metric_seeds_rejected():
    rows = _pair()
    rows[1] = dataclasses.replace(
        rows[1], seeds=dataclasses.replace(rows[1].seeds, metric_seed=424242)
    )
    result = validate_rows(rows)
    assert "mixed_data_or_metric_seeds" in _rules(result)
    assert result.accepted == ()


def test_unmatched_single_arm_rejected():
    result = validate_rows(_pair()[:1])
    assert "unmatched_conditions" in _rules(result)
    assert result.accepted == ()


def test_unmatched_same_condition_rejected():
    rows = _pair(condition="frozen_zero_v")
    result = validate_rows(rows)
    assert "unmatched_conditions" in _rules(result)
    assert result.accepted == ()


def test_unmatched_three_arms_rejected():
    rows = _pair() + [make_synthetic_row(experiment_id="e-c", condition="third")]
    result = validate_rows(rows)
    assert "unmatched_conditions" in _rules(result)
    assert result.accepted == ()


def test_identity_mismatch_sampler_under_v_trainability_rejected():
    rows = _pair(sampler_name="one_shot_stochastic")
    result = validate_rows(rows)
    assert "identity_mismatch" in _rules(result)
    assert result.accepted == ()


def test_sampler_stochasticity_intervention_exemption():
    a = make_synthetic_row(
        experiment_id="e-a",
        intervention="sampler_stochasticity",
        condition="stochastic",
        sampler_name="sequential_random_stochastic",
        model_config_digest="modelcfg-baseline",
    )
    b = make_synthetic_row(
        experiment_id="e-b",
        intervention="sampler_stochasticity",
        condition="greedy",
        sampler_name="sequential_random_greedy",
        model_config_digest="modelcfg-baseline",
    )
    result = validate_rows([a, b])
    assert result.rejections == ()
    assert len(result.accepted) == 2


def test_model_digest_mismatch_under_sampler_stochasticity_rejected():
    a = make_synthetic_row(
        experiment_id="e-a",
        intervention="sampler_stochasticity",
        condition="stochastic",
        sampler_name="sequential_random_stochastic",
        model_config_digest="modelcfg-baseline",
    )
    b = make_synthetic_row(
        experiment_id="e-b",
        intervention="sampler_stochasticity",
        condition="greedy",
        sampler_name="sequential_random_greedy",
        model_config_digest="modelcfg-other",
    )
    result = validate_rows([a, b])
    assert "identity_mismatch" in _rules(result)


def test_mixed_interventions_within_pair_rejected():
    rows = _pair(intervention="objective")
    result = validate_rows(rows)
    assert "unmatched_conditions" in _rules(result)


def test_condition_with_two_identities_across_repeats_rejected():
    rows = _pair()
    extra_a = make_synthetic_row(
        experiment_id="e-a2",
        repeat_id=1,
        condition="frozen_zero_v",
        model_config_digest="modelcfg-frozen_zero",
        sampler_name="one_shot_stochastic",
    )
    extra_b = make_synthetic_row(
        experiment_id="e-b2",
        repeat_id=1,
        condition="trainable_v",
        model_config_digest="modelcfg-trainable",
    )
    result = validate_rows(rows + [extra_a, extra_b])
    assert "unmatched_conditions" in _rules(result)
    assert result.accepted == ()


def test_missing_comparison_rejected():
    rows = _pair()
    mmd = {c: m for c, m in rows[0].mmd.items() if c != "model_vs_train"}
    rows[0] = dataclasses.replace(rows[0], mmd=mmd)
    result = validate_rows(rows)
    assert "missing_metric" in _rules(result)


def test_nonfinite_metric_rejected():
    rows = _pair()
    bad = MMDSummary(mixture_biased_mmd2=float("nan"), mixture_unbiased_mmd2_raw=0.0)
    rows[0] = dataclasses.replace(rows[0], mmd={**rows[0].mmd, "model_vs_true": bad})
    result = validate_rows(rows)
    assert "nonfinite_metric" in _rules(result)


def test_negative_unbiased_mmd2_is_legitimate():
    rows = _pair()
    for i, row in enumerate(rows):
        neg = MMDSummary(mixture_biased_mmd2=1e-3, mixture_unbiased_mmd2_raw=-2e-4)
        rows[i] = dataclasses.replace(row, mmd={**row.mmd, "true_vs_true": neg})
    result = validate_rows(rows)
    assert result.rejections == ()
    assert len(result.accepted) == 2


def test_uturn_curve_length_mismatch_raises():
    from maskeddiffusion.analysis.rows import UTurnSummary

    with pytest.raises(ValueError, match="equal length"):
        UTurnSummary(
            mask_densities=(0.1, 0.2), overlap=(0.9,), baseline_recovery=0.5, excess_recovery=0.1
        )


def test_comparisons_constant_covers_four_ways():
    assert COMPARISONS == ("model_vs_true", "true_vs_true", "train_vs_true", "model_vs_train")
