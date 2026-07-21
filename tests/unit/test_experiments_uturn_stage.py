"""uturn_summary_to_record_block: source selection and scalar reduction."""

from __future__ import annotations

import pytest

from maskeddiffusion.experiments.uturn_stage import uturn_summary_to_record_block


def _point(source, t, q_u_mean):
    return {
        "source": source,
        "t_value": t,
        "q_u_mean": q_u_mean,
        "no_recovery_baseline": 1.0 - t,
        "excess_recovery_mean": q_u_mean - (1.0 - t),
    }


def _summary(sources, points):
    return {"sources": list(sources), "points": points}


def test_prefers_fresh_source_when_available():
    summary = _summary(
        ("train", "fresh"),
        [
            _point("train", 0.2, 0.95),
            _point("train", 0.8, 0.90),
            _point("fresh", 0.2, 0.81),
            _point("fresh", 0.8, 0.19),
        ],
    )
    block = uturn_summary_to_record_block(summary)
    assert block["mask_densities"] == [0.2, 0.8]
    assert block["overlap"] == [0.81, 0.19]
    assert block["baseline_recovery"] == pytest.approx((0.8 + 0.2) / 2)
    assert block["excess_recovery"] == pytest.approx(((0.81 - 0.8) + (0.19 - 0.2)) / 2)


def test_falls_back_to_train_when_fresh_absent():
    summary = _summary(("train",), [_point("train", 0.5, 0.6)])
    block = uturn_summary_to_record_block(summary)
    assert block["mask_densities"] == [0.5]
    assert block["overlap"] == [0.6]
    assert block["baseline_recovery"] == pytest.approx(0.5)
    assert block["excess_recovery"] == pytest.approx(0.1)


def test_points_sorted_by_t_regardless_of_input_order():
    summary = _summary(
        ("fresh",),
        [_point("fresh", 0.8, 0.1), _point("fresh", 0.2, 0.9), _point("fresh", 0.5, 0.5)],
    )
    block = uturn_summary_to_record_block(summary)
    assert block["mask_densities"] == [0.2, 0.5, 0.8]
    assert block["overlap"] == [0.9, 0.5, 0.1]


def test_raises_on_unknown_source():
    with pytest.raises(ValueError, match="no known U-turn source"):
        uturn_summary_to_record_block(_summary(("bogus",), []))


def test_raises_when_headline_source_has_no_points():
    with pytest.raises(ValueError, match="no points"):
        uturn_summary_to_record_block(_summary(("fresh",), []))
