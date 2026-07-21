"""`run_record.json` → `AnalysisRow` — the only artifact-reading code in the
analysis layer (docs/PHASE4C_ANALYSIS_SPEC.md §0, §10).

Every field of an `AnalysisRow` comes from a fully validated
`Phase4CRunRecord` (`maskeddiffusion.experiments.schema`); nothing is
inferred from directory layout or filenames. `record_validated` is carried
straight from the record's own `validation_problems` — an invalid record is
never laundered into a clean row; `validate_rows` rejects it downstream with
an explicit `unvalidated_artifact` rejection (spec §9 rule 1).
"""

from __future__ import annotations

from pathlib import Path

from ..artifacts import sha256_file
from ..experiments.schema import RUN_RECORD_FILENAME, Phase4CRunRecord, load_run_record
from .rows import AnalysisRow, MMDSummary, UTurnSummary


def discover_run_records(root: str | Path) -> list[Path]:
    """All `run_record.json` files under an experiment output root, sorted.

    Discovery only locates candidate files; every scrap of metadata comes
    from the records themselves, which are fully re-validated on load.
    """
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"records root {root} is not a directory")
    return sorted(root.rglob(RUN_RECORD_FILENAME))


def _mmd_summary(block: dict) -> MMDSummary:
    return MMDSummary(
        mixture_biased_mmd2=block["mixture_biased_mmd2"],
        mixture_unbiased_mmd2_raw=block["mixture_unbiased_mmd2_raw"],
        per_lambda_biased={float(k): v for k, v in block["biased_mmd2"].items()},
        per_lambda_unbiased_raw={float(k): v for k, v in block["unbiased_mmd2_raw"].items()},
    )


def record_to_row(record: Phase4CRunRecord, record_path: str | Path) -> AnalysisRow:
    """One tidy row from one canonical record (spec §2)."""
    record_path = Path(record_path)
    uturn = None
    if record.uturn is not None:
        uturn = UTurnSummary(
            mask_densities=tuple(record.uturn["mask_densities"]),
            overlap=tuple(record.uturn["overlap"]),
            baseline_recovery=record.uturn["baseline_recovery"],
            excess_recovery=record.uturn["excess_recovery"],
        )
    dims = record.dimensions
    return AnalysisRow(
        # The engine's experiment_id names the whole campaign; the tidy row's
        # experiment_id must identify ONE run. pair_id already embeds the
        # campaign id and repeat (`{experiment_id}-r{repeat:03d}`), so
        # `{pair_id}-{condition}` is unique per run by construction.
        experiment_id=f"{record.pair_id}-{record.condition}",
        pair_id=record.pair_id,
        repeat_id=record.repeat_id,
        latent_dim=dims.latent_dim,
        visible_dim=dims.visible_dim,
        train_size=dims.train_size,
        aspect_ratio=dims.aspect_ratio,
        sample_ratio=dims.sample_ratio,
        visible_load=dims.visible_load,
        intervention=record.intervention,
        condition=record.condition,
        teacher_id=record.teacher_id,
        checkpoint_id=record.checkpoint_id,
        sampler_name=record.sampler["sampler_name"],
        sampler_tokens_per_step=record.sampler["tokens_per_step"],
        model_config_digest=record.model_config_digest,
        seeds=record.seeds,
        mmd={name: _mmd_summary(block) for name, block in record.mmd.items()},
        nearest_training_excess=record.nearest_training["excess"],
        nearest_training_model_mean=record.nearest_training["model_mean_nearest_overlap"],
        nearest_training_true_mean=record.nearest_training["true_mean_nearest_overlap"],
        pair_correlation_rms_error=record.pair_correlation_error["rms_error"],
        pair_correlation_max_abs_error=record.pair_correlation_error["max_abs_error"],
        uturn=uturn,
        optimization_step=record.optimization_step,
        examples_seen=record.examples_seen,
        artifact_path=str(record_path),
        artifact_sha256=sha256_file(record_path),
        record_validated=record.record_validated,
    )


def load_rows(record_paths: list[Path], *, verify_hashes: bool = True) -> list[AnalysisRow]:
    """Load, re-validate, and convert each record. Malformed records raise —
    a file that cannot even be parsed as a `Phase4CRunRecord` is a broken
    input, not a rejectable row. `verify_hashes=True` (default) additionally
    recomputes every linked artifact's SHA-256 and raises on any mismatch or
    missing file (`experiments.schema.verify_artifact_hashes`) — real
    tamper detection, not just structural validation."""
    return [record_to_row(load_run_record(p, verify_hashes=verify_hashes), p) for p in record_paths]
