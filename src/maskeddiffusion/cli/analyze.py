"""maskeddiffusion-p4c-analyze: Phase 4C analysis over canonical run records.

Consumes only validated `run_record.json` files written by the experiment
engine (`maskeddiffusion.experiments.schema`); never scans run directories
for metadata or infers anything from filenames. Produces the tidy CSVs,
machine-readable report JSON, figures, and provenance manifest specified in
docs/PHASE4C_ANALYSIS_SPEC.md §§2-8. Rejected rows are excluded from every
table and figure and reported with rule and reason — nothing is dropped
silently.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from ..analysis.ingest import discover_run_records, load_rows
from ..analysis.plots import (
    check_figure_wording,
    plot_metric_by_condition,
    plot_paired_delta,
    plot_uturn,
    safe_filename_fragment,
    save_figure,
)
from ..analysis.provenance import (
    MANIFEST_JSON,
    build_analysis_manifest,
    file_entry,
    write_manifest,
)
from ..analysis.report import (
    REPORT_JSON,
    STATISTICS_POLICY,
    build_report,
    check_report_wording,
    write_report,
    write_tables,
)
from ..analysis.rows import AGGREGATED_METRICS, AnalysisRow, rows_to_frame, validate_rows
from ..analysis.statistics import paired_differences


def _write_figures(out_dir: Path, rows: tuple[AnalysisRow, ...]) -> tuple[list[Path], list[str]]:
    """All spec-§7 figures; returns (paths, wording problems)."""
    fig_dir = out_dir / "figures"
    paths: list[Path] = []
    problems: list[str] = []
    if not rows:
        return paths, problems
    frame = rows_to_frame(rows)

    def emit(fig, base_name: str) -> None:
        problems.extend(f"{base_name}: {p}" for p in check_figure_wording(fig))
        paths.extend(save_figure(fig, fig_dir / base_name))
        plt.close(fig)

    for intervention in sorted(frame["intervention"].unique()):
        sub = frame[frame["intervention"] == intervention]
        iv = safe_filename_fragment(intervention)
        for (d, n, m), cell_frame in sub.groupby(
            ["latent_dim", "visible_dim", "train_size"], sort=True
        ):
            cell = f"D{d}_N{n}_M{m}"
            for metric in AGGREGATED_METRICS:
                fig = plot_metric_by_condition(cell_frame, metric=metric)
                emit(fig, f"p4c_fig_{iv}_{cell}_{safe_filename_fragment(metric)}_by_condition")
        paired = paired_differences(sub, AGGREGATED_METRICS)
        if not paired.empty:
            for metric in AGGREGATED_METRICS:
                fig = plot_paired_delta(paired, metric=metric)
                emit(fig, f"p4c_fig_{iv}_{safe_filename_fragment(metric)}_paired_delta")

    uturn_groups: dict[tuple[str, int], list[AnalysisRow]] = {}
    for row in rows:
        if row.uturn is not None:
            uturn_groups.setdefault((row.pair_id, row.repeat_id), []).append(row)
    for (pair_id, repeat_id), group in sorted(uturn_groups.items()):
        fig = plot_uturn(group)
        emit(
            fig,
            f"p4c_fig_uturn_{safe_filename_fragment(pair_id)}_r{repeat_id:03d}",
        )
    return paths, problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "p4c-analyze")
    parser.add_argument(
        "--records",
        required=True,
        help="experiment output root; every run_record.json below it is consumed",
    )
    parser.add_argument("--output", required=True, help="analysis output directory")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list discovered records and planned outputs; write nothing",
    )
    args = parser.parse_args(argv)
    out = Path(args.output)

    record_paths = discover_run_records(args.records)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "n_records": len(record_paths),
                    "records": [str(p) for p in record_paths],
                    "planned_outputs": {
                        "tables": [
                            str(out / name)
                            for name in (
                                "p4c_per_run.csv",
                                "p4c_paired_differences.csv",
                                "p4c_aggregate.csv",
                                "p4c_aggregate_paired.csv",
                                "p4c_matched_seed_finite_size.csv",
                            )
                        ],
                        "report": str(out / REPORT_JSON),
                        "figures_dir": str(out / "figures"),
                        "manifest": str(out / MANIFEST_JSON),
                    },
                },
                indent=2,
            )
        )
        return 0
    if not record_paths:
        raise SystemExit(f"no {'run_record.json'} files found under {args.records}")

    rows = load_rows(record_paths)
    result = validate_rows(rows)

    out.mkdir(parents=True, exist_ok=True)
    table_paths = write_tables(out, list(result.accepted))
    report = build_report(list(result.accepted), list(result.rejections), input_row_count=len(rows))
    report_wording_problems = [f"report JSON: {p}" for p in check_report_wording(report)]
    report_path = write_report(report, out / REPORT_JSON)
    figure_paths, figure_wording_problems = _write_figures(out, result.accepted)
    wording_problems = report_wording_problems + figure_wording_problems
    if wording_problems:
        raise SystemExit(
            "wording guard failed (docs/PHASE4C_ANALYSIS_SPEC.md §7): "
            + "; ".join(wording_problems)
        )

    validation_block: dict[str, Any] = {
        "input_record_count": len(rows),
        "accepted_row_count": len(result.accepted),
        "rejection_count": len(result.rejections),
        "rejections": [
            {"rule": r.rule, "reason": r.reason, "experiment_ids": list(r.experiment_ids)}
            for r in result.rejections
        ],
    }
    outputs = [
        file_entry(path, f"tidy table ({role})", "table") for role, path in table_paths.items()
    ]
    outputs.append(file_entry(report_path, "machine-readable analysis report", "report"))
    outputs.extend(file_entry(p, "figure", "figure") for p in figure_paths)
    manifest = build_analysis_manifest(
        command=" ".join(["maskeddiffusion-p4c-analyze", *sys.argv[1:]]),
        inputs=[file_entry(p, "canonical Phase 4C run record", "run_record") for p in record_paths],
        outputs=outputs,
        validation=validation_block,
        statistics_policy=STATISTICS_POLICY,
    )
    write_manifest(manifest, out / MANIFEST_JSON)
    print(
        json.dumps(
            {
                "n_records": len(rows),
                "n_accepted": len(result.accepted),
                "n_rejections": len(result.rejections),
                "output": str(out),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
