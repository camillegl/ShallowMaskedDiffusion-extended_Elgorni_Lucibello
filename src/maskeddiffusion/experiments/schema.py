"""Canonical Phase 4C run record (`run_record.json`).

One `Phase4CRunRecord` describes ONE completed engine run — one
(intervention, condition, repeat) evaluation of a sampler-indexed terminal
law `P_{θ,A,k}` against the finite-F teacher law `P_F`. It is the single
artifact-level contract between the paired-experiment engine
(`maskeddiffusion.experiment.runner`, which builds and writes it) and the
analysis layer (`maskeddiffusion.analysis.ingest`, the only code allowed to
turn it into an `AnalysisRow`). Downstream analysis must never scan run
directories or infer metadata from filenames; everything it needs is in
this record, and `validation_problems` says whether the underlying stage
artifacts passed validation at record-build time.

The record carries no new science: every field is copied from the run's own
manifests (`run_manifest.json`, the three ADR-003 stage artifacts) — the
schema describes what the engine emits, it does not invent fields. MMD
values keep both the mixture-level and per-kernel-scale biased /
raw-unbiased estimates; raw unbiased MMD² may legitimately be negative and
is never clipped (docs/RESEARCH_SPEC.md).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..artifacts import sha256_file, validate_artifact
from ..dimensions import Dimensions
from ..randomness import SeedHierarchy

RUN_RECORD_SCHEMA_VERSION = "maskeddiffusion.p4c_run_record.v1"
RUN_RECORD_FILENAME = "run_record.json"

#: The four MMD comparisons every record must carry (docs/PHASE4C_ANALYSIS_SPEC.md §1).
COMPARISONS: tuple[str, ...] = (
    "model_vs_true",
    "true_vs_true",
    "train_vs_true",
    "model_vs_train",
)

_MMD_BLOCK_KEYS = {
    "mixture_biased_mmd2",
    "mixture_unbiased_mmd2_raw",
    "biased_mmd2",
    "unbiased_mmd2_raw",
}
_NEAREST_KEYS = {"model_mean_nearest_overlap", "true_mean_nearest_overlap", "excess"}
_PAIR_CORR_KEYS = {"max_abs_error", "rms_error", "mean_error"}
_UTURN_KEYS = {"mask_densities", "overlap", "baseline_recovery", "excess_recovery"}
_MIGRATION_KEYS = {"from_schema", "to_schema", "performed_at", "source_artifacts_unchanged"}

#: Every completion status a record may carry. A single frozen value today —
#: only completed runs ever get a record — kept as an explicit enum (not a
#: bare truthy field) so a future partial/failed-run record type has
#: somewhere to declare itself rather than overloading `validation_problems`.
COMPLETION_STATUSES: frozenset[str] = frozenset({"completed"})

#: Stages every record links a manifest for; `uturn` is additionally present
#: only when the arm ran the optional U-turn stage.
REQUIRED_ARTIFACT_STAGES: tuple[str, ...] = ("train", "samples", "eval")
OPTIONAL_ARTIFACT_STAGES: tuple[str, ...] = ("uturn",)
_ARTIFACT_ENTRY_KEYS = {"manifest_path", "manifest_sha256"}

_RECORD_KEYS = (
    "schema_version",
    "status",
    "experiment_id",
    "pair_id",
    "repeat_id",
    "intervention",
    "condition",
    "dimensions",
    "seeds",
    "spec_fingerprint",
    "resolved_config",
    "resolved_config_sha256",
    "teacher_id",
    "checkpoint_id",
    "checkpoint_file_sha256",
    "sampler",
    "model",
    "model_config_digest",
    "artifacts",
    "mmd",
    "nearest_training",
    "pair_correlation_error",
    "uturn",
    "optimization_step",
    "examples_seen",
    "train_data_sha256",
    "validation_data_sha256",
    "git_commit",
    "run_dir",
    "run_manifest_sha256",
    "validation_problems",
    "migration",
)


def model_config_digest(model_identity: dict[str, Any]) -> str:
    """Content digest of a model identity dict (`LinearScoreConfig.identity()`)."""
    payload = json.dumps(model_identity, sort_keys=True, separators=(",", ":"))
    return "modelcfg-" + hashlib.sha256(payload.encode()).hexdigest()[:16]


def _check_artifacts(artifacts: Any) -> None:
    if not isinstance(artifacts, dict):
        raise TypeError(f"artifacts must be a dict, got {type(artifacts).__name__}")
    missing = set(REQUIRED_ARTIFACT_STAGES) - set(artifacts)
    if missing:
        raise ValueError(f"artifacts missing required stages {sorted(missing)}")
    unknown = set(artifacts) - set(REQUIRED_ARTIFACT_STAGES) - set(OPTIONAL_ARTIFACT_STAGES)
    if unknown:
        raise ValueError(f"artifacts has unknown stages {sorted(unknown)}")
    for stage, entry in artifacts.items():
        if not isinstance(entry, dict) or set(entry) != _ARTIFACT_ENTRY_KEYS:
            raise ValueError(
                f"artifacts[{stage!r}] must be a dict with keys {sorted(_ARTIFACT_ENTRY_KEYS)}"
            )
        for key in _ARTIFACT_ENTRY_KEYS:
            if not isinstance(entry[key], str) or not entry[key]:
                raise ValueError(f"artifacts[{stage!r}].{key} must be a nonempty string")


#: Public shape of `Phase4CRunRecord.uturn` — the single source of truth for
#: what `experiments.uturn_stage.uturn_summary_to_record_block` must produce
#: and what `Phase4CRunRecord.__post_init__` accepts. Importing this (via
#: `check_uturn_block`) rather than re-deriving the key set keeps the two
#: layers from silently drifting apart.
UTURN_BLOCK_KEYS: frozenset[str] = frozenset(_UTURN_KEYS)


def check_uturn_block(uturn: Any) -> None:
    """Raise unless `uturn` is `None` or a valid reduced U-turn block.

    The one place this shape is validated; both `Phase4CRunRecord.
    __post_init__` and `uturn_stage.uturn_summary_to_record_block` call this
    rather than each re-implementing the check.
    """
    if uturn is None:
        return
    if not isinstance(uturn, dict) or set(uturn) != _UTURN_KEYS:
        raise ValueError(f"uturn must be None or a dict with keys {sorted(_UTURN_KEYS)}")
    densities, overlap = uturn["mask_densities"], uturn["overlap"]
    if (
        not isinstance(densities, (list, tuple))
        or not isinstance(overlap, (list, tuple))
        or not densities
        or len(densities) != len(overlap)
    ):
        raise ValueError("uturn mask_densities and overlap must be nonempty equal-length sequences")
    for label, values in (("mask_densities", densities), ("overlap", overlap)):
        if not all(_finite(v) for v in values):
            raise ValueError(f"uturn.{label} values must all be finite")
    for key in ("baseline_recovery", "excess_recovery"):
        if not _finite(uturn[key]):
            raise ValueError(f"uturn.{key} must be finite, got {uturn[key]!r}")


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _check_mmd_block(name: str, block: Any) -> None:
    if not isinstance(block, dict) or set(block) != _MMD_BLOCK_KEYS:
        raise ValueError(
            f"mmd[{name!r}] must be a dict with keys {sorted(_MMD_BLOCK_KEYS)}, got "
            f"{sorted(block) if isinstance(block, dict) else type(block).__name__}"
        )
    for key in ("mixture_biased_mmd2", "mixture_unbiased_mmd2_raw"):
        if not _finite(block[key]):
            raise ValueError(f"mmd[{name!r}].{key} must be finite, got {block[key]!r}")
    per_biased, per_unbiased = block["biased_mmd2"], block["unbiased_mmd2_raw"]
    for label, per in (("biased_mmd2", per_biased), ("unbiased_mmd2_raw", per_unbiased)):
        if not isinstance(per, dict) or not per:
            raise ValueError(f"mmd[{name!r}].{label} must be a nonempty {{lambda: value}} dict")
        for lam, value in per.items():
            if not isinstance(lam, str):
                raise ValueError(
                    f"mmd[{name!r}].{label} kernel-scale keys must be strings "
                    f"(JSON round-trip stable), got {lam!r}"
                )
            if not _finite(value):
                raise ValueError(f"mmd[{name!r}].{label}[{lam!r}] must be finite, got {value!r}")
    if set(per_biased) != set(per_unbiased):
        raise ValueError(
            f"mmd[{name!r}] biased and unbiased kernel-scale sets differ: "
            f"{sorted(per_biased)} vs {sorted(per_unbiased)}"
        )


@dataclass(frozen=True)
class Phase4CRunRecord:
    """One completed, self-describing Phase 4C run (see module docstring)."""

    status: str
    experiment_id: str
    pair_id: str
    repeat_id: int
    intervention: str
    condition: str
    dimensions: Dimensions
    seeds: SeedHierarchy
    spec_fingerprint: str
    resolved_config: dict[str, Any]
    resolved_config_sha256: str
    teacher_id: str
    checkpoint_id: str
    checkpoint_file_sha256: str
    sampler: dict[str, Any]
    model: dict[str, Any]
    artifacts: dict[str, dict[str, str]]
    mmd: dict[str, dict[str, Any]]
    nearest_training: dict[str, float]
    pair_correlation_error: dict[str, float]
    optimization_step: int
    examples_seen: int
    train_data_sha256: str
    validation_data_sha256: str | None
    git_commit: str | None
    run_dir: str
    run_manifest_sha256: str
    uturn: dict[str, Any] | None = None
    validation_problems: tuple[str, ...] = field(default_factory=tuple)
    migration: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.status not in COMPLETION_STATUSES:
            raise ValueError(
                f"status must be one of {sorted(COMPLETION_STATUSES)}, got {self.status!r}"
            )
        for name in ("experiment_id", "pair_id", "condition", "intervention"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a nonempty string, got {getattr(self, name)!r}")
        if not isinstance(self.repeat_id, int) or isinstance(self.repeat_id, bool):
            raise TypeError(f"repeat_id must be int, got {type(self.repeat_id).__name__}")
        if self.repeat_id < 0:
            raise ValueError(f"repeat_id must be >= 0, got {self.repeat_id}")
        for name in (
            "spec_fingerprint",
            "resolved_config_sha256",
            "teacher_id",
            "checkpoint_id",
            "checkpoint_file_sha256",
            "train_data_sha256",
            "run_dir",
            "run_manifest_sha256",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a nonempty string")
        if self.git_commit is not None and (
            not isinstance(self.git_commit, str) or not self.git_commit
        ):
            raise ValueError(
                f"git_commit must be None or a nonempty string, got {self.git_commit!r}"
            )
        if not isinstance(self.resolved_config, dict) or not self.resolved_config:
            raise ValueError(
                "resolved_config must be a nonempty dict (the reconstructible RunConfig)"
            )
        _check_artifacts(self.artifacts)
        if not isinstance(self.dimensions, Dimensions):
            raise TypeError("dimensions must be a Dimensions instance")
        if not isinstance(self.seeds, SeedHierarchy):
            raise TypeError("seeds must be a SeedHierarchy instance")
        if not isinstance(self.sampler, dict) or "sampler_name" not in self.sampler:
            raise ValueError("sampler must be an identity dict containing sampler_name")
        tokens = self.sampler.get("tokens_per_step")
        if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 1:
            raise ValueError(f"sampler.tokens_per_step must be an int >= 1, got {tokens!r}")
        if not isinstance(self.model, dict) or not self.model:
            raise ValueError("model must be a nonempty identity dict")
        if set(self.mmd) != set(COMPARISONS):
            raise ValueError(
                f"mmd must carry exactly the comparisons {list(COMPARISONS)}, got "
                f"{sorted(self.mmd)}"
            )
        for name, block in self.mmd.items():
            _check_mmd_block(name, block)
        if set(self.nearest_training) != _NEAREST_KEYS:
            raise ValueError(f"nearest_training must have keys {sorted(_NEAREST_KEYS)}")
        if set(self.pair_correlation_error) != _PAIR_CORR_KEYS:
            raise ValueError(f"pair_correlation_error must have keys {sorted(_PAIR_CORR_KEYS)}")
        for group_name, group in (
            ("nearest_training", self.nearest_training),
            ("pair_correlation_error", self.pair_correlation_error),
        ):
            for key, value in group.items():
                if not _finite(value):
                    raise ValueError(f"{group_name}.{key} must be finite, got {value!r}")
        check_uturn_block(self.uturn)
        for name in ("optimization_step", "examples_seen"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be an int >= 0, got {value!r}")
        if not isinstance(self.validation_problems, tuple) or not all(
            isinstance(p, str) for p in self.validation_problems
        ):
            raise TypeError("validation_problems must be a tuple of strings")
        if self.migration is not None:
            if not isinstance(self.migration, dict) or set(self.migration) != _MIGRATION_KEYS:
                raise ValueError(
                    f"migration must be None or a dict with keys {sorted(_MIGRATION_KEYS)}"
                )
            if self.migration["from_schema"] is not None and not isinstance(
                self.migration["from_schema"], str
            ):
                raise TypeError("migration.from_schema must be None or a string")
            if not isinstance(self.migration["to_schema"], str) or not self.migration["to_schema"]:
                raise ValueError("migration.to_schema must be a nonempty string")
            if (
                not isinstance(self.migration["performed_at"], str)
                or not self.migration["performed_at"]
            ):
                raise ValueError("migration.performed_at must be a nonempty ISO timestamp string")
            if not isinstance(self.migration["source_artifacts_unchanged"], bool):
                raise TypeError("migration.source_artifacts_unchanged must be a bool")

    @property
    def model_config_digest(self) -> str:
        return model_config_digest(self.model)

    @property
    def is_migrated(self) -> bool:
        """True iff this record was constructed after the fact (a missing
        `run_record.json` backfilled on resume) rather than written
        immediately when the run first completed."""
        return self.migration is not None

    @property
    def record_validated(self) -> bool:
        """True iff every stage artifact and cross-check passed at build time."""
        return not self.validation_problems

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RUN_RECORD_SCHEMA_VERSION,
            "status": self.status,
            "experiment_id": self.experiment_id,
            "pair_id": self.pair_id,
            "repeat_id": self.repeat_id,
            "intervention": self.intervention,
            "condition": self.condition,
            "dimensions": self.dimensions.to_dict(),
            "seeds": self.seeds.to_dict(),
            "spec_fingerprint": self.spec_fingerprint,
            "resolved_config": dict(self.resolved_config),
            "resolved_config_sha256": self.resolved_config_sha256,
            "teacher_id": self.teacher_id,
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_file_sha256": self.checkpoint_file_sha256,
            "sampler": dict(self.sampler),
            "model": dict(self.model),
            "model_config_digest": self.model_config_digest,
            "artifacts": {stage: dict(entry) for stage, entry in self.artifacts.items()},
            "mmd": {name: dict(block) for name, block in self.mmd.items()},
            "nearest_training": dict(self.nearest_training),
            "pair_correlation_error": dict(self.pair_correlation_error),
            "uturn": dict(self.uturn) if self.uturn is not None else None,
            "optimization_step": self.optimization_step,
            "examples_seen": self.examples_seen,
            "train_data_sha256": self.train_data_sha256,
            "validation_data_sha256": self.validation_data_sha256,
            "git_commit": self.git_commit,
            "run_dir": self.run_dir,
            "run_manifest_sha256": self.run_manifest_sha256,
            "validation_problems": list(self.validation_problems),
            "migration": dict(self.migration) if self.migration is not None else None,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Phase4CRunRecord:
        if d.get("schema_version") != RUN_RECORD_SCHEMA_VERSION:
            raise ValueError(
                f"run record has schema_version {d.get('schema_version')!r}, expected "
                f"{RUN_RECORD_SCHEMA_VERSION!r}"
            )
        unknown = set(d) - set(_RECORD_KEYS)
        if unknown:
            raise ValueError(f"unknown run-record keys {sorted(unknown)}")
        missing = [k for k in _RECORD_KEYS if k not in d]
        if missing:
            raise ValueError(f"run record missing keys {missing}")
        record = Phase4CRunRecord(
            status=d["status"],
            experiment_id=d["experiment_id"],
            pair_id=d["pair_id"],
            repeat_id=d["repeat_id"],
            intervention=d["intervention"],
            condition=d["condition"],
            dimensions=Dimensions.from_dict(d["dimensions"]),
            seeds=SeedHierarchy.from_dict(d["seeds"]),
            spec_fingerprint=d["spec_fingerprint"],
            resolved_config=d["resolved_config"],
            resolved_config_sha256=d["resolved_config_sha256"],
            teacher_id=d["teacher_id"],
            checkpoint_id=d["checkpoint_id"],
            checkpoint_file_sha256=d["checkpoint_file_sha256"],
            sampler=d["sampler"],
            model=d["model"],
            artifacts=d["artifacts"],
            mmd=d["mmd"],
            nearest_training=d["nearest_training"],
            pair_correlation_error=d["pair_correlation_error"],
            uturn=d["uturn"],
            optimization_step=d["optimization_step"],
            examples_seen=d["examples_seen"],
            train_data_sha256=d["train_data_sha256"],
            validation_data_sha256=d["validation_data_sha256"],
            git_commit=d["git_commit"],
            run_dir=d["run_dir"],
            run_manifest_sha256=d["run_manifest_sha256"],
            validation_problems=tuple(d["validation_problems"]),
            migration=d["migration"],
        )
        if d["model_config_digest"] != record.model_config_digest:
            raise ValueError(
                f"stored model_config_digest {d['model_config_digest']!r} does not match "
                f"the model identity's digest {record.model_config_digest!r} — the record "
                "was tampered with or written by an incompatible version"
            )
        return record


def _read_json(path: Path, description: str) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"cannot build run record: missing {description} at {path}")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"cannot build run record: malformed {description} at {path}: {e}") from e


def backfill_migration_block(*, performed_at: str) -> dict[str, Any]:
    """The standard `migration` block for a `run_record.json` built after
    the fact — a run that completed before a record existed for it, or whose
    record was lost. `from_schema: None` marks "no prior record existed;
    this is not a schema-version upgrade of a previous one." `performed_at`
    (an ISO timestamp) is the caller's responsibility so tests can supply a
    fixed value instead of wall-clock time."""
    return {
        "from_schema": None,
        "to_schema": RUN_RECORD_SCHEMA_VERSION,
        "performed_at": performed_at,
        "source_artifacts_unchanged": True,
    }


def build_run_record(
    run_dir: str | Path,
    *,
    uturn: dict[str, Any] | None = None,
    migration: dict[str, Any] | None = None,
) -> Phase4CRunRecord:
    """Assemble the canonical record from a completed engine run directory.

    Reads only the run's own manifests and summaries (`run_manifest.json`,
    `train/summary.json`, `eval/summary.json`, `eval/manifest.json`); the
    heavyweight provenance checks (checkpoint identity recomputation, file
    hashes, teacher matching) already ran inside `evaluate_run` when the eval
    stage was produced. Stage-artifact validation problems are recorded in
    `validation_problems`, not raised — the analysis layer rejects
    unvalidated records explicitly (spec §9 rule 1). Missing or malformed
    files that make the record itself unbuildable DO raise.
    """
    run_dir = Path(run_dir)
    run_manifest_path = run_dir / "run_manifest.json"
    run_manifest = _read_json(run_manifest_path, "run_manifest.json")
    if run_manifest.get("schema_version") != "maskeddiffusion.experiment_run.v1":
        raise ValueError(
            f"run manifest at {run_manifest_path} has schema_version "
            f"{run_manifest.get('schema_version')!r}; expected the engine's "
            "maskeddiffusion.experiment_run.v1"
        )
    spec = run_manifest.get("spec") or {}
    train_summary = _read_json(run_dir / "train" / "summary.json", "train summary")
    eval_summary = _read_json(run_dir / "eval" / "summary.json", "eval summary")
    eval_manifest = _read_json(run_dir / "eval" / "manifest.json", "eval manifest")

    # Canonical resolved config: every stage writes an identical copy
    # (`config.to_json`); the train stage's is authoritative since it is the
    # one build_state actually ran against.
    resolved_config_path = run_dir / "train" / "resolved_config.json"
    resolved_config = _read_json(resolved_config_path, "train resolved_config.json")
    resolved_config_sha256 = sha256_file(resolved_config_path)

    # Required stages' own manifest.json must exist for the record to link
    # complete provenance — a missing one is not a soft validation problem
    # (like a hash mismatch on a file that does exist) but the same kind of
    # unbuildable-record condition as a missing train/eval summary.json, and
    # raises accordingly. Optional stages (uturn) are simply absent when not
    # run.
    artifacts: dict[str, dict[str, str]] = {}
    for stage in REQUIRED_ARTIFACT_STAGES:
        manifest_path = run_dir / stage / "manifest.json"
        artifacts[stage] = {
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
        }
    for stage in OPTIONAL_ARTIFACT_STAGES:
        manifest_path = run_dir / stage / "manifest.json"
        if manifest_path.exists():
            artifacts[stage] = {
                "manifest_path": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
            }

    # "completed" is the only value COMPLETION_STATUSES accepts today (only
    # completed runs ever reach build_run_record); an inconsistent
    # run_manifest status is flagged below as a validation problem, not
    # smuggled into a different status value.
    status = "completed"
    problems: list[str] = []
    if run_manifest.get("status") != "completed":
        problems.append(f"run_manifest status {run_manifest.get('status')!r} != 'completed'")
    for stage in REQUIRED_ARTIFACT_STAGES:
        for problem in validate_artifact(run_dir / stage):
            problems.append(f"stage {stage}: {problem}")

    mmd = {name: eval_summary[name] for name in COMPARISONS if name in eval_summary}
    missing_cmp = [name for name in COMPARISONS if name not in mmd]
    if missing_cmp:
        raise ValueError(f"eval summary at {run_dir} is missing MMD comparisons {missing_cmp}")
    # `display_sqrt_clipped_mixture` is a presentation convenience of the
    # evaluate CLI, not part of the canonical record.
    mmd = {
        name: {k: v for k, v in block.items() if k in _MMD_BLOCK_KEYS}
        for name, block in mmd.items()
    }

    checkpoint_id = eval_manifest.get("checkpoint_id")
    checkpoint_file_sha256 = eval_manifest.get("checkpoint_file_sha256")
    if not checkpoint_id or not checkpoint_file_sha256:
        raise ValueError(
            f"eval manifest at {run_dir} lacks checkpoint_id/checkpoint_file_sha256 — "
            "regenerate the run with the current package version"
        )

    model_identity_keys = (
        "model",
        "visible_dim",
        "normalization",
        "v_policy",
        "bias_policy",
        "diagonal_policy",
    )
    eval_model = eval_manifest.get("model") or {}
    model_identity = {k: eval_model[k] for k in model_identity_keys if k in eval_model}
    if set(model_identity) != set(model_identity_keys):
        raise ValueError(
            f"eval manifest model config at {run_dir} lacks identity keys "
            f"{sorted(set(model_identity_keys) - set(model_identity))}"
        )

    git_commit = run_manifest.get("environment", {}).get("git_sha")

    return Phase4CRunRecord(
        status=status,
        experiment_id=spec["experiment_id"],
        pair_id=spec["pair_id"],
        repeat_id=spec["repeat_id"],
        intervention=spec["intervention"],
        condition=spec["condition"],
        dimensions=Dimensions.from_dict(spec["dimensions"]),
        seeds=SeedHierarchy.from_dict(spec["seeds"]),
        spec_fingerprint=run_manifest["spec_fingerprint"],
        resolved_config=resolved_config,
        resolved_config_sha256=resolved_config_sha256,
        teacher_id=run_manifest["teacher_id"],
        checkpoint_id=checkpoint_id,
        checkpoint_file_sha256=checkpoint_file_sha256,
        sampler=eval_manifest["sampler"],
        model=model_identity,
        artifacts=artifacts,
        mmd=mmd,
        nearest_training=eval_summary["nearest_training"],
        pair_correlation_error=eval_summary["pair_correlation_error"],
        uturn=uturn,
        optimization_step=int(train_summary["final_step"]),
        examples_seen=int(train_summary["examples_seen"]),
        train_data_sha256=run_manifest["train_data_sha256"],
        validation_data_sha256=run_manifest["validation_data_sha256"],
        git_commit=git_commit,
        run_dir=str(run_dir),
        run_manifest_sha256=sha256_file(run_manifest_path),
        validation_problems=tuple(problems),
        migration=migration,
    )


def write_run_record(record: Phase4CRunRecord, run_dir: str | Path) -> Path:
    """Write `run_record.json` into the run directory.

    Never overwrites an existing record that differs: a disagreeing record
    means the directory content changed after the record was written, which
    is a provenance question for a human, not for silent regeneration.
    """
    path = Path(run_dir) / RUN_RECORD_FILENAME
    payload = record.to_dict()
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            raise ValueError(
                f"existing run record {path} is malformed ({e}); refusing to overwrite — "
                "move it aside manually"
            ) from e
        if existing != payload:
            raise ValueError(
                f"existing run record {path} disagrees with the freshly built record — "
                "refusing to overwrite; move it aside manually"
            )
        return path
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def verify_artifact_hashes(record: Phase4CRunRecord) -> list[str]:
    """Recompute every linked artifact's SHA-256 and compare against the
    record's stored value; return problem strings (empty = all consistent).

    This is what makes the record's hashes worth anything: a
    `run_record.json` that merely stores a hash without anyone ever
    recomputing it provides no tamper detection at all. Checks: the
    resolved-config file (`<run_dir>/train/resolved_config.json`), every
    stage's own `manifest.json` (`record.artifacts`), the checkpoint file
    (`<run_dir>/train/checkpoints/final.pt`), and `run_manifest.json`
    itself. A missing linked file is reported as a problem, not raised —
    callers decide whether that is fatal.
    """
    problems: list[str] = []
    run_dir = Path(record.run_dir)

    def check(label: str, path: Path, expected: str) -> None:
        if not path.exists():
            problems.append(f"{label}: linked file missing at {path}")
            return
        actual = sha256_file(path)
        if actual != expected:
            problems.append(
                f"{label}: sha256 mismatch at {path} (expected {expected}, got {actual})"
            )

    check(
        "resolved_config",
        run_dir / "train" / "resolved_config.json",
        record.resolved_config_sha256,
    )
    for stage, entry in record.artifacts.items():
        check(f"artifacts[{stage!r}]", Path(entry["manifest_path"]), entry["manifest_sha256"])
    check(
        "checkpoint",
        run_dir / "train" / "checkpoints" / "final.pt",
        record.checkpoint_file_sha256,
    )
    check("run_manifest", run_dir / "run_manifest.json", record.run_manifest_sha256)
    return problems


def load_run_record(path: str | Path, *, verify_hashes: bool = True) -> Phase4CRunRecord:
    """Load and fully re-validate one `run_record.json`.

    `verify_hashes=True` (default) additionally recomputes and checks every
    linked artifact's SHA-256 (`verify_artifact_hashes`) and raises if any
    mismatch or missing file is found — a record whose hashes are never
    re-checked provides no tamper detection. Pass `False` only for
    inspecting a record whose linked run directory is known to be
    unavailable (e.g. inspecting a copied-out `run_record.json` alone).
    """
    path = Path(path)
    record = Phase4CRunRecord.from_dict(_read_json(path, RUN_RECORD_FILENAME))
    if verify_hashes:
        problems = verify_artifact_hashes(record)
        if problems:
            raise ValueError(
                f"run record {path} failed artifact hash verification: {'; '.join(problems)}"
            )
    return record
