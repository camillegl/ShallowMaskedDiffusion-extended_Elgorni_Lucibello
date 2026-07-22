"""Phase 4C figures (docs/PHASE4C_ANALYSIS_SPEC.md §7).

Publication-quality, matplotlib-only. Binding rules implemented here:

- raw repeat points are always visible; paired conditions are connected by one
  line per (pair_id, repeat_id);
- one figure per (intervention, design cell, metric) — no hidden aggregation
  across cells or interventions;
- the True-True finite-sample floor is drawn directly (per-repeat markers plus
  a dashed median line);
- log-y only for biased MMD² metrics whose values are all strictly positive
  (MMD² is nonnegative and spans orders of magnitude); everything else linear;
- a wording guard enforces the claim-discipline vocabulary of
  docs/RESEARCH_SPEC.md on every generated figure's text.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.text
import pandas as pd
from matplotlib.figure import Figure

from .rows import UTURN_SCALAR_COLUMNS, AnalysisRow

#: Phrases banned from every generated artifact's text (spec §7; prohibited
#: overclaims of docs/RESEARCH_SPEC.md). Matched case-insensitively as
#: substrings; "converg" covers converges/converged/convergence.
FORBIDDEN_PHRASES: tuple[str, ...] = (
    "converg",
    "learns the distribution",
    "learn the distribution",
    "learned the distribution",
    "phase transition",
    "the model distribution",
    "exact ancestral sampling",
)

METRIC_LABELS: dict[str, str] = {
    "model_true_mmd2_biased": "Model–True MMD² (mixture, biased V-statistic)",
    "model_true_mmd2_unbiased_raw": "Model–True MMD² (mixture, raw unbiased U-statistic)",
    "true_true_mmd2_biased": "True–True MMD² floor (mixture, biased V-statistic)",
    "true_true_mmd2_unbiased_raw": "True–True MMD² floor (mixture, raw unbiased U-statistic)",
    "train_true_mmd2_biased": "Train–True MMD² (mixture, biased V-statistic)",
    "train_true_mmd2_unbiased_raw": "Train–True MMD² (mixture, raw unbiased U-statistic)",
    "model_train_mmd2_biased": "Model–Train MMD² (mixture, biased V-statistic)",
    "model_train_mmd2_unbiased_raw": "Model–Train MMD² (mixture, raw unbiased U-statistic)",
    "nearest_training_excess": "nearest-training excess overlap",
    "pair_correlation_rms_error": "pair-correlation RMS error",
    "uturn_baseline_recovery": "U-turn baseline recovery",
    "uturn_excess_recovery": "U-turn excess recovery",
}


def check_wording(text: str) -> list[str]:
    """Return the forbidden phrases found in `text` (empty = clean)."""
    lowered = text.lower()
    return [phrase for phrase in FORBIDDEN_PHRASES if phrase in lowered]


def check_figure_wording(fig: Figure) -> list[str]:
    """Scan every text artist of a figure for forbidden phrases."""
    found: list[str] = []
    for text in fig.findobj(matplotlib.text.Text):
        found.extend(check_wording(text.get_text()))
    return sorted(set(found))


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric)


def cell_label(row: pd.Series | dict[str, Any]) -> str:
    """Realized-integer cell label; never bare ratio symbols (spec §7)."""
    return f"D={int(row['latent_dim'])}, N={int(row['visible_dim'])}, M={int(row['train_size'])}"


def _new_figure() -> tuple[Figure, Any]:
    fig, ax = plt.subplots(figsize=(5.2, 4.0), constrained_layout=True)
    return fig, ax


def plot_metric_by_condition(
    cell_frame: pd.DataFrame,
    *,
    metric: str,
    floor_metric: str | None = "true_true_mmd2_biased",
) -> Figure:
    """Raw repeat points per condition with paired connecting lines and the
    True-True floor drawn directly (spec §7.1).

    `cell_frame` must contain exactly one (intervention, design cell).
    """
    if cell_frame["intervention"].nunique() != 1:
        raise ValueError("cell_frame must hold exactly one intervention")
    intervention = cell_frame["intervention"].iloc[0]
    conditions = sorted(cell_frame["condition"].unique())
    x_of = {c: i for i, c in enumerate(conditions)}

    fig, ax = _new_figure()
    # Paired connecting lines first (under the points).
    for _key, group in cell_frame.groupby(["pair_id", "repeat_id"], sort=True):
        group = group.sort_values("condition")
        if len(group) == 2:
            ax.plot(
                [x_of[c] for c in group["condition"]],
                group[metric],
                color="0.6",
                linewidth=0.9,
                zorder=1,
            )
    for condition in conditions:
        values = cell_frame.loc[cell_frame["condition"] == condition, metric]
        ax.scatter(
            [x_of[condition]] * len(values),
            values,
            color="tab:blue",
            edgecolor="black",
            linewidths=0.4,
            s=42,
            zorder=3,
            label="individual repeat" if condition == conditions[0] else None,
        )
    if floor_metric is not None and floor_metric in cell_frame:
        floor = cell_frame[floor_metric].dropna()
        for condition in conditions:
            vals = cell_frame.loc[cell_frame["condition"] == condition, floor_metric].dropna()
            ax.scatter(
                [x_of[condition]] * len(vals),
                vals,
                marker="D",
                facecolor="none",
                edgecolor="tab:red",
                s=30,
                zorder=2,
                label="True–True floor (repeat)" if condition == conditions[0] else None,
            )
        if len(floor):
            ax.axhline(
                floor.median(),
                color="tab:red",
                linestyle="--",
                linewidth=1.0,
                label="True–True floor (median)",
            )
    values = cell_frame[metric].dropna()
    if metric.endswith("_mmd2_biased") and len(values) and (values > 0).all():
        # Justified log axis: biased MMD² is nonnegative and spans orders of
        # magnitude (spec §7). Raw-unbiased MMD² can be negative -> linear.
        ax.set_yscale("log")
    ax.set_xticks(list(x_of.values()), conditions)
    ax.set_ylabel(metric_label(metric))
    ax.set_title(
        f"{metric_label(metric)} by condition\n{intervention} ({cell_label(cell_frame.iloc[0])})"
    )
    ax.legend(loc="best", fontsize=8)
    return fig


def plot_paired_delta(
    paired_frame: pd.DataFrame,
    *,
    metric: str,
) -> Figure:
    """Per-repeat paired B−A differences, zero line, mean ± sample-std (spec §7.2).

    `paired_frame` must contain exactly one intervention; cells are placed on
    the x-axis. Deltas can be negative, so the axis is always linear.
    """
    if paired_frame["intervention"].nunique() != 1:
        raise ValueError("paired_frame must hold exactly one intervention")
    intervention = paired_frame["intervention"].iloc[0]
    delta = f"{metric}_delta_b_minus_a"
    cell_keys = ["latent_dim", "visible_dim", "train_size"]
    cells = paired_frame[cell_keys].drop_duplicates().sort_values(cell_keys)
    labels = [cell_label(row) for _, row in cells.iterrows()]

    fig, ax = _new_figure()
    for i, (_, cell) in enumerate(cells.iterrows()):
        mask = (paired_frame[cell_keys] == cell[cell_keys].values).all(axis=1)
        values = paired_frame.loc[mask, delta].dropna()
        ax.scatter(
            [i] * len(values),
            values,
            color="tab:blue",
            edgecolor="black",
            linewidths=0.4,
            s=42,
            zorder=3,
            label="individual repeat" if i == 0 else None,
        )
        if len(values):
            ax.errorbar(
                [i],
                [values.mean()],
                yerr=[values.std(ddof=1)] if len(values) >= 2 else None,
                fmt="_",
                markersize=14,
                color="black",
                capsize=5,
                zorder=4,
                label="mean ± sample std (n repeats)" if i == 0 else None,
            )
    ax.axhline(0.0, color="0.4", linewidth=0.8, linestyle=":")
    ax.set_xticks(range(len(labels)), labels, rotation=15, ha="right")
    conds = paired_frame[["condition_a", "condition_b"]].iloc[0]
    ax.set_ylabel(f"Δ ({conds['condition_b']} − {conds['condition_a']}) {metric_label(metric)}")
    ax.set_title(f"Paired B−A differences per repeat\n{intervention}")
    ax.legend(loc="best", fontsize=8)
    return fig


def plot_uturn(rows: list[AnalysisRow] | tuple[AnalysisRow, ...]) -> Figure:
    """q_U(t) curves for one (pair_id, repeat_id), one line per condition (spec §7.3)."""
    rows = [r for r in rows if r.uturn is not None]
    if not rows:
        raise ValueError("plot_uturn requires rows carrying U-turn data")
    if len({(r.pair_id, r.repeat_id) for r in rows}) != 1:
        raise ValueError("plot_uturn takes exactly one (pair_id, repeat_id)")
    fig, ax = _new_figure()
    for row in sorted(rows, key=lambda r: r.condition):
        assert row.uturn is not None  # narrowed above
        ax.plot(
            row.uturn.mask_densities,
            row.uturn.overlap,
            marker="o",
            markersize=4,
            label=(
                f"{row.condition} (baseline {row.uturn.baseline_recovery:.3f}, "
                f"excess {row.uturn.excess_recovery:.3f})"
            ),
        )
    pair_id, repeat_id = rows[0].pair_id, rows[0].repeat_id
    ax.set_xlabel("mask density t")
    ax.set_ylabel("q_U(t)")
    ax.set_title(
        f"U-turn retrieval overlap\n{pair_id} repeat {repeat_id} ({cell_label(rows[0].__dict__)})"
    )
    ax.legend(loc="best", fontsize=8)
    return fig


def save_figure(fig: Figure, out_base: str | Path) -> list[Path]:
    """Write PDF + 300-dpi PNG side by side; returns the written paths."""
    base = Path(out_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    paths = [base.with_suffix(".pdf"), base.with_suffix(".png")]
    fig.savefig(paths[0])
    fig.savefig(paths[1], dpi=300)
    return paths


def safe_filename_fragment(text: str) -> str:
    """Filesystem-safe fragment for intervention/metric names."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text)


__all__ = [
    "FORBIDDEN_PHRASES",
    "METRIC_LABELS",
    "UTURN_SCALAR_COLUMNS",
    "cell_label",
    "check_figure_wording",
    "check_wording",
    "metric_label",
    "plot_metric_by_condition",
    "plot_paired_delta",
    "plot_uturn",
    "safe_filename_fragment",
    "save_figure",
]
