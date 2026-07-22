"""Phase 4C statistics: paired differences and repeat aggregates (spec §5).

Binding policy (docs/PHASE4C_ANALYSIS_SPEC.md §5): every individual repeat is
retained; paired A−B differences are taken inside `(pair_id, repeat_id)` before
any cross-repeat aggregation; central tendency is the mean with the **sample**
standard deviation (ddof=1; empty when n < 2); the median is supplementary.
There are no bootstrap confidence intervals and no significance tests at n=3 —
these functions deliberately expose no API for either.

`paired_differences` computes a same-disorder A−B delta and is only valid for
`comparison_type == "paired_disorder"` rows (`analysis.rows.comparison_type`):
`matched_seed_finite_size` (`finite_d`) rows are excluded by default — a
different `latent_dim` means a different-shape teacher, so a "B minus A"
delta between them is not a same-disorder comparison and must never be
presented as one. `finite_d` groups can also legitimately hold more than two
arms (a whole D-sweep), which violates `paired_differences`'s two-arm
assumption outright. Anyone who wants a matched-seed, explicitly-labeled,
NON-paired view of a `finite_d` sweep must use
`matched_seed_finite_size_frame` instead, never `paired_differences`.
"""

from __future__ import annotations

import pandas as pd

from .rows import AGGREGATED_METRICS, UTURN_SCALAR_COLUMNS, comparison_type

#: Design-cell keys shared by aggregates and paired tables.
CELL_KEYS: tuple[str, ...] = (
    "latent_dim",
    "aspect_ratio",
    "sample_ratio",
    "visible_dim",
    "train_size",
)

_PAIR_KEY_COLUMNS: tuple[str, ...] = (
    "pair_id",
    "repeat_id",
    "intervention",
    "condition_a",
    "condition_b",
    *CELL_KEYS,
    "visible_load",
    "teacher_id",
)

_PAIR_CONTEXT_COLUMNS: tuple[str, ...] = (
    "sampler_name_a",
    "sampler_name_b",
    "optimization_step_a",
    "optimization_step_b",
    "examples_seen_a",
    "examples_seen_b",
)


def paired_difference_columns(metrics: tuple[str, ...] = AGGREGATED_METRICS) -> list[str]:
    """Exact column order of the paired-difference table (spec §3)."""
    cols: list[str] = list(_PAIR_KEY_COLUMNS)
    for m in metrics:
        cols += [f"{m}_a", f"{m}_b", f"{m}_delta_b_minus_a"]
    cols += list(_PAIR_CONTEXT_COLUMNS)
    return cols


def paired_differences(
    frame: pd.DataFrame,
    metrics: tuple[str, ...] = AGGREGATED_METRICS,
    *,
    include_matched_seed_finite_size: bool = False,
) -> pd.DataFrame:
    """One row per `(pair_id, repeat_id)`: per-metric arm values and B−A.

    `condition_a`/`condition_b` are the two condition labels sorted
    lexicographically (a deterministic, recorded convention — spec §3). Input
    must already be validated (`validate_rows`); a malformed pair raises
    `ValueError` rather than being silently repaired.

    `matched_seed_finite_size` rows (`finite_d`) are excluded before pairing
    unless `include_matched_seed_finite_size=True` is passed explicitly —
    see the module docstring for why a B−A delta across different-`latent_dim`
    arms is not a same-disorder comparison. Passing `True` does not relabel
    the output; callers that need an explicitly-labeled matched-seed view
    must use `matched_seed_finite_size_frame` instead.
    """
    if not include_matched_seed_finite_size:
        frame = frame[frame["intervention"].map(comparison_type) != "matched_seed_finite_size"]
    records: list[dict[str, object]] = []
    for (pair_id, repeat_id), group in frame.groupby(["pair_id", "repeat_id"], sort=True):
        conditions = sorted(group["condition"].unique())
        if len(group) != 2 or len(conditions) != 2:
            raise ValueError(
                f"pair {pair_id!r} repeat {repeat_id} has {len(group)} rows with conditions "
                f"{conditions}; validate_rows must run before paired_differences"
            )
        row_a = group[group["condition"] == conditions[0]].iloc[0]
        row_b = group[group["condition"] == conditions[1]].iloc[0]
        rec: dict[str, object] = {
            "pair_id": pair_id,
            "repeat_id": repeat_id,
            "intervention": row_a["intervention"],
            "condition_a": conditions[0],
            "condition_b": conditions[1],
            "teacher_id": row_a["teacher_id"],
        }
        for key in CELL_KEYS + ("visible_load",):
            if row_a[key] != row_b[key]:
                raise ValueError(
                    f"pair {pair_id!r} repeat {repeat_id} differs in {key!r}; "
                    "validate_rows must run before paired_differences"
                )
            rec[key] = row_a[key]
        for m in metrics:
            rec[f"{m}_a"] = row_a[m]
            rec[f"{m}_b"] = row_b[m]
            rec[f"{m}_delta_b_minus_a"] = row_b[m] - row_a[m]
        rec["sampler_name_a"] = row_a["sampler_name"]
        rec["sampler_name_b"] = row_b["sampler_name"]
        rec["optimization_step_a"] = row_a["optimization_step"]
        rec["optimization_step_b"] = row_b["optimization_step"]
        rec["examples_seen_a"] = row_a["examples_seen"]
        rec["examples_seen_b"] = row_b["examples_seen"]
        records.append(rec)
    return pd.DataFrame(records, columns=paired_difference_columns(metrics))


#: Exact column order of the matched-seed finite-size table: the per-run
#: dimension/identity/metric columns plus an explicit label so this can
#: never be mistaken for a paired_disorder table downstream.
def matched_seed_finite_size_columns(metrics: tuple[str, ...] = AGGREGATED_METRICS) -> list[str]:
    return [
        "comparison_type",
        "pair_id",
        "repeat_id",
        "intervention",
        "condition",
        "teacher_id",
        *CELL_KEYS,
        "visible_load",
        *metrics,
    ]


def matched_seed_finite_size_frame(
    frame: pd.DataFrame, metrics: tuple[str, ...] = AGGREGATED_METRICS
) -> pd.DataFrame:
    """Explicitly-labeled, NON-paired view of `matched_seed_finite_size` rows.

    One row per run (not per pair): distinct arms of a `finite_d` sweep are
    never differenced against each other here — every row simply carries its
    own dimensions, teacher_id, and metrics, plus a `comparison_type` column
    fixed to `"matched_seed_finite_size"` so this table cannot be confused
    with `paired_differences`'s output even if concatenated with it. Input
    must already be validated (`validate_rows`).
    """
    subset = frame[frame["intervention"].map(comparison_type) == "matched_seed_finite_size"]
    columns = matched_seed_finite_size_columns(metrics)
    if subset.empty:
        return pd.DataFrame(columns=columns)
    source_columns = [
        "pair_id",
        "repeat_id",
        "intervention",
        "condition",
        "teacher_id",
        *CELL_KEYS,
        "visible_load",
        *metrics,
    ]
    out = subset[source_columns].copy()
    out.insert(0, "comparison_type", "matched_seed_finite_size")
    out = out.sort_values(["pair_id", "repeat_id", "latent_dim"]).reset_index(drop=True)
    return out[columns]


def _aggregate_block(
    frame: pd.DataFrame, group_keys: list[str], metrics: tuple[str, ...], n_name: str
) -> pd.DataFrame:
    """Mean / sample-std (ddof=1) / median per metric within groups.

    Sample std is NaN for groups with fewer than two non-NaN values — reported
    as an empty CSV cell, never back-filled (spec §4/§5).
    """
    records: list[dict[str, object]] = []
    for keys, group in frame.groupby(group_keys, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec: dict[str, object] = dict(zip(group_keys, keys, strict=True))
        rec[n_name] = len(group)
        for m in metrics:
            values = group[m].dropna()
            rec[f"{m}_mean"] = values.mean() if len(values) else None
            rec[f"{m}_std_sample"] = values.std(ddof=1) if len(values) >= 2 else None
            rec[f"{m}_median"] = values.median() if len(values) else None
        records.append(rec)
    columns = (
        list(group_keys)
        + [n_name]
        + [f"{m}_{stat}" for m in metrics for stat in ("mean", "std_sample", "median")]
    )
    return pd.DataFrame(records, columns=columns)


def aggregate_repeats(
    frame: pd.DataFrame, metrics: tuple[str, ...] = AGGREGATED_METRICS
) -> pd.DataFrame:
    """One row per design cell × condition, over all individual repeats (spec §4).

    U-turn scalar blocks are appended only when at least one row carries U-turn
    data.
    """
    group_keys = ["intervention", "condition", *CELL_KEYS]
    all_metrics: tuple[str, ...] = metrics
    if frame[list(UTURN_SCALAR_COLUMNS)].notna().any().any():
        all_metrics = metrics + UTURN_SCALAR_COLUMNS
    return _aggregate_block(frame, group_keys, all_metrics, "n_repeats")


def aggregate_paired(
    paired: pd.DataFrame, metrics: tuple[str, ...] = AGGREGATED_METRICS
) -> pd.DataFrame:
    """One row per design cell: distribution of the paired B−A differences."""
    group_keys = ["intervention", "condition_a", "condition_b", *CELL_KEYS]
    delta_metrics = tuple(f"{m}_delta_b_minus_a" for m in metrics)
    return _aggregate_block(paired, group_keys, delta_metrics, "n_pairs")
