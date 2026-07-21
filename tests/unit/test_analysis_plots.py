"""Figure tests (docs/PHASE4C_ANALYSIS_SPEC.md §7): raw points visible, paired
conditions connected, floor drawn directly, log axes only where justified, and
the wording guard applied to every generated figure."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.collections import PathCollection

from maskeddiffusion.analysis.plots import (
    FORBIDDEN_PHRASES,
    check_figure_wording,
    check_wording,
    plot_metric_by_condition,
    plot_paired_delta,
    plot_uturn,
    save_figure,
)
from maskeddiffusion.analysis.rows import rows_to_frame, validate_rows
from maskeddiffusion.analysis.statistics import paired_differences
from maskeddiffusion.analysis.synthetic import synthetic_rows


@pytest.fixture()
def accepted():
    result = validate_rows(synthetic_rows())
    assert result.rejections == ()
    return result.accepted


@pytest.fixture()
def vtrainability_cell_frame(accepted):
    frame = rows_to_frame(accepted)
    cell = frame[
        (frame["intervention"] == "v_trainability")
        & (frame["latent_dim"] == 16)
        & (frame["train_size"] == 96)
    ]
    assert len(cell) == 6  # 2 conditions × 3 repeats
    return cell


def test_by_condition_shows_raw_points_pairs_and_floor(vtrainability_cell_frame):
    fig = plot_metric_by_condition(vtrainability_cell_frame, metric="model_true_mmd2_biased")
    ax = fig.axes[0]
    # 3 paired connectors (one per repeat) + 1 median floor line.
    assert len(ax.lines) == 4
    for line in ax.lines[:3]:
        assert len(line.get_xdata()) == 2  # connects exactly the two conditions
    # 2 condition scatter collections + 2 floor diamond collections, each
    # holding the 3 individual repeats (raw points visible).
    assert len(ax.collections) == 4
    offsets = sorted(len(c.get_offsets()) for c in ax.collections)
    assert offsets == [3, 3, 3, 3]
    # Justified log axis: biased MMD², all values strictly positive.
    assert ax.get_yscale() == "log"
    assert check_figure_wording(fig) == []
    plt.close(fig)


def test_by_condition_linear_for_raw_unbiased(vtrainability_cell_frame):
    fig = plot_metric_by_condition(vtrainability_cell_frame, metric="model_true_mmd2_unbiased_raw")
    assert fig.axes[0].get_yscale() == "linear"
    plt.close(fig)


def test_by_condition_requires_single_intervention(accepted):
    frame = rows_to_frame(accepted)
    with pytest.raises(ValueError, match="one intervention"):
        plot_metric_by_condition(frame, metric="model_true_mmd2_biased")


def test_paired_delta_figure(accepted):
    frame = rows_to_frame(accepted)
    paired = paired_differences(frame)
    subset = paired[paired["intervention"] == "v_trainability"]
    fig = plot_paired_delta(subset, metric="model_true_mmd2_biased")
    ax = fig.axes[0]
    assert ax.get_yscale() == "linear"  # deltas can be negative
    assert len(ax.get_xticks()) == 2  # two design cells
    # Per-repeat points: 2 cells × 3 repeats = 6 raw scatter points.
    total = sum(len(c.get_offsets()) for c in ax.collections if isinstance(c, PathCollection))
    assert total == 6
    assert check_figure_wording(fig) == []
    plt.close(fig)


def test_uturn_figure(accepted):
    rows = [
        r
        for r in accepted
        if r.intervention == "v_trainability" and r.train_size == 96 and r.repeat_id == 0
    ]
    assert len(rows) == 2 and all(r.uturn is not None for r in rows)
    fig = plot_uturn(rows)
    ax = fig.axes[0]
    assert len(ax.lines) == 2  # one q_U(t) curve per condition
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert all("baseline" in t and "excess" in t for t in labels)
    assert check_figure_wording(fig) == []
    plt.close(fig)


def test_uturn_figure_rejects_missing_data(accepted):
    rows = [r for r in accepted if r.intervention == "sampler_stochasticity"]
    with pytest.raises(ValueError, match="U-turn"):
        plot_uturn(rows)


def test_save_figure_writes_pdf_and_png(vtrainability_cell_frame, tmp_path):
    fig = plot_metric_by_condition(vtrainability_cell_frame, metric="model_true_mmd2_biased")
    paths = save_figure(fig, tmp_path / "figures" / "p4c_fig_test")
    assert [p.suffix for p in paths] == [".pdf", ".png"]
    assert paths[0].read_bytes().startswith(b"%PDF")
    assert paths[1].read_bytes().startswith(b"\x89PNG")
    plt.close(fig)


def test_check_wording_catches_each_forbidden_phrase():
    assert check_wording("the model converges to the target")
    assert check_wording("CONVERGENCE of the loss")
    assert check_wording("the model learns the distribution")
    assert check_wording("a phase transition appears")
    assert check_wording("samples from the model distribution")
    assert check_wording("exact ancestral sampling")
    assert check_wording("approaches the finite-F target under this diagnostic") == []
    assert "converg" in FORBIDDEN_PHRASES
