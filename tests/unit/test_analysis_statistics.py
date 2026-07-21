"""Statistics tests (docs/PHASE4C_ANALYSIS_SPEC.md §5): paired differences and
aggregates with hand-computed expectations; no bootstrap, no significance
tests exist anywhere in the module's API."""

import math

import pytest

from maskeddiffusion.analysis.rows import COMPARISONS, MMDSummary, rows_to_frame
from maskeddiffusion.analysis.statistics import (
    aggregate_paired,
    aggregate_repeats,
    paired_difference_columns,
    paired_differences,
)
from maskeddiffusion.analysis.synthetic import make_synthetic_row, synthetic_rows


def _mmd(biased):
    return MMDSummary(mixture_biased_mmd2=biased, mixture_unbiased_mmd2_raw=biased - 0.001)


def _row(eid, pair, repeat, condition, biased, **overrides):
    mmd = {cmp: _mmd(biased) for cmp in COMPARISONS}
    return make_synthetic_row(
        experiment_id=eid,
        pair_id=pair,
        repeat_id=repeat,
        condition=condition,
        mmd=mmd,
        **overrides,
    )


def _known_set(a_values=(0.01, 0.02, 0.03), b_values=(0.005, 0.006, 0.007)):
    """3 repeats of a v_trainability pair with exactly known Model-True MMD² values."""
    rows = []
    for i, (va, vb) in enumerate(zip(a_values, b_values, strict=True)):
        rows.append(
            _row(
                f"a-r{i}",
                "p1",
                i,
                "frozen_zero_v",
                va,
                model_config_digest="modelcfg-frozen_zero",
            )
        )
        rows.append(
            _row(f"b-r{i}", "p1", i, "trainable_v", vb, model_config_digest="modelcfg-trainable")
        )
    return rows


def test_paired_differences_exact_values_and_column_order():
    frame = rows_to_frame(_known_set())
    paired = paired_differences(frame)
    assert tuple(paired.columns) == tuple(paired_difference_columns())
    assert len(paired) == 3
    # condition_a/b are the sorted labels: frozen_zero_v < trainable_v.
    assert set(paired["condition_a"]) == {"frozen_zero_v"}
    assert set(paired["condition_b"]) == {"trainable_v"}
    got = dict(
        zip(paired["repeat_id"], paired["model_true_mmd2_biased_delta_b_minus_a"], strict=True)
    )
    assert math.isclose(got[0], 0.005 - 0.01)
    assert math.isclose(got[1], 0.006 - 0.02)
    assert math.isclose(got[2], 0.007 - 0.03)
    # Arm values are retained alongside the delta.
    got_a = dict(zip(paired["repeat_id"], paired["model_true_mmd2_biased_a"], strict=True))
    assert math.isclose(got_a[1], 0.02)


def test_paired_differences_rejects_malformed_pair():
    rows = _known_set() + [
        _row("extra", "p1", 0, "third_arm", 0.05, model_config_digest="modelcfg-frozen_zero")
    ]
    with pytest.raises(ValueError, match="validate_rows"):
        paired_differences(rows_to_frame(rows))


def test_aggregate_repeats_hand_computed():
    frame = rows_to_frame(_known_set())
    agg = aggregate_repeats(frame)
    assert len(agg) == 2  # one row per condition
    frozen = agg[agg["condition"] == "frozen_zero_v"].iloc[0]
    assert frozen["n_repeats"] == 3
    assert math.isclose(frozen["model_true_mmd2_biased_mean"], 0.02)
    # sample std (ddof=1) of (0.01, 0.02, 0.03) is 0.01
    assert math.isclose(frozen["model_true_mmd2_biased_std_sample"], 0.01)
    assert math.isclose(frozen["model_true_mmd2_biased_median"], 0.02)
    trainable = agg[agg["condition"] == "trainable_v"].iloc[0]
    assert math.isclose(trainable["model_true_mmd2_biased_mean"], 0.006)
    # No bootstrap or significance columns exist anywhere.
    assert not any("ci" in c.lower() or "pval" in c.lower() for c in agg.columns)


def test_aggregate_repeats_std_empty_below_two_repeats():
    rows = _known_set(a_values=(0.01,), b_values=(0.005,))
    agg = aggregate_repeats(rows_to_frame(rows))
    assert len(agg) == 2
    assert agg["n_repeats"].tolist() == [1, 1]
    assert agg["model_true_mmd2_biased_std_sample"].isna().all()
    assert math.isclose(
        agg.iloc[0]["model_true_mmd2_biased_median"], agg.iloc[0]["model_true_mmd2_biased_mean"]
    )


def test_aggregate_paired_hand_computed():
    paired = paired_differences(rows_to_frame(_known_set()))
    agg = aggregate_paired(paired)
    assert len(agg) == 1
    row = agg.iloc[0]
    assert row["n_pairs"] == 3
    deltas = (0.005 - 0.01, 0.006 - 0.02, 0.007 - 0.03)
    assert math.isclose(row["model_true_mmd2_biased_delta_b_minus_a_mean"], sum(deltas) / 3)
    # sample std (ddof=1) of the three deltas
    mean = sum(deltas) / 3
    expect = math.sqrt(sum((d - mean) ** 2 for d in deltas) / 2)
    assert math.isclose(row["model_true_mmd2_biased_delta_b_minus_a_std_sample"], expect)
    assert math.isclose(row["model_true_mmd2_biased_delta_b_minus_a_median"], sorted(deltas)[1])


def test_uturn_columns_appended_only_when_present():
    no_uturn = aggregate_repeats(rows_to_frame(_known_set()))
    assert not any(c.startswith("uturn_") for c in no_uturn.columns)
    with_uturn = aggregate_repeats(rows_to_frame(synthetic_rows()))
    assert "uturn_excess_recovery_mean" in with_uturn.columns
    assert "uturn_baseline_recovery_std_sample" in with_uturn.columns
    # U-turn data exist only under v_trainability in the synthetic set.
    sampler_rows = with_uturn[with_uturn["intervention"] == "sampler_stochasticity"]
    assert sampler_rows["uturn_excess_recovery_mean"].isna().all()
    vtrainability_rows = with_uturn[with_uturn["intervention"] == "v_trainability"]
    assert vtrainability_rows["uturn_excess_recovery_mean"].notna().all()


def test_statistics_module_exposes_no_test_theatre_api():
    import maskeddiffusion.analysis.statistics as stats

    public = [n for n in dir(stats) if not n.startswith("_")]
    assert not any(
        any(token in name for token in ("bootstrap", "pvalue", "p_value", "significance"))
        for name in public
    )
