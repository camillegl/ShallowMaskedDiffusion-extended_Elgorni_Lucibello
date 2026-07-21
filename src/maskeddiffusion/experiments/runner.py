"""Resumable executor for an `ExperimentPlan`.

Per condition the engine runs the full train -> sample -> evaluate pipeline
into `<experiment_root>/<pair_id>/<condition>/{train,samples,eval}/`, each a
standard ADR-003 run artifact (gated by `artifacts.validate_artifact`), plus
a `run_manifest.json` (schema `maskeddiffusion.experiment_run.v1`) recording
the spec, its fingerprint, and content hashes of the exact teacher / train /
validation tensors used. When every arm of a comparison group is complete,
a `pair_manifest.json` (schema `maskeddiffusion.experiment_pair.v1`) records
checks specific to the group's `comparison_type`
(`experiments.interventions` module docstring):

- `"paired_disorder"` (v_trainability, sampler_stochasticity,
  optimization_budget): teacher-id and data-hash equality across arms is
  REQUIRED — a mismatch means CPU determinism was violated and the group is
  refused.
- `"matched_seed_finite_size"` (finite_d only): arms share every seed VALUE
  but a different `latent_dim` means a different-shape teacher, so
  `teacher_id` equality is impossible and disorder-content equality is
  never checked. Instead `teacher_id` DISTINCTNESS is asserted — a
  collision would mean the arms were not actually at different dimensions,
  a real bug. This group is never described as "paired disorder".

Resume discipline (docs/PHASE4C_EXPERIMENT_PROTOCOL.md §4):

- a missing run directory is executed fresh;
- a run directory whose `run_manifest.json` reads `status: "completed"`,
  whose spec and fingerprints match the current plan exactly, and whose
  three stage artifacts all pass `validate_artifact` is SKIPPED — nothing
  is rewritten, so a resume leaves the tree byte-stable;
- a SKIPPED run's `run_record.json`, if already present, is IMMUTABLE: it
  is re-validated (`load_run_record`) and never rebuilt or rewritten — the
  stage-artifact re-validation above already re-proves nothing changed, so
  there is nothing to discover by recomputing it. If `run_record.json` is
  absent (a run that completed before a record existed for it, or whose
  record was lost), it is built exactly once with an explicit `migration`
  block recording that fact — never silently;
- anything else (missing/malformed/partial manifest, spec or fingerprint
  mismatch, failing stage validation, disagreeing resolved config) is
  REFUSED with the exact problems listed. The engine never overwrites any
  existing run output, valid or invalid; recovery is a manual decision to
  move the directory aside.

The evaluation stage reuses `maskeddiffusion.cli.evaluate.evaluate_run` so
the engine inherits the full checkpoint/teacher/sample provenance
verification of the CLI rather than duplicating it.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from ..artifacts import RunArtifact, environment_metadata, sha256_file, validate_artifact
from ..checkpoints import load_checkpoint
from ..cli.evaluate import evaluate_run
from ..models import LinearMaskedScore, LinearScoreConfig
from ..samplers import sample as run_sampler
from ..training import build_state, train
from .pairs import diff_leaves, validate_group
from .plan import ExperimentPlan
from .schema import backfill_migration_block, build_run_record, load_run_record, write_run_record
from .spec import ExperimentSpec, spec_fingerprint
from .uturn_stage import load_uturn_block, run_uturn_stage, uturn_summary_to_record_block

RUN_SCHEMA_VERSION = "maskeddiffusion.experiment_run.v1"
PAIR_SCHEMA_VERSION = "maskeddiffusion.experiment_pair.v1"
STAGES = ("train", "samples", "eval")


def _tensor_sha256(tensor: torch.Tensor) -> str:
    t = tensor.detach().to(torch.float32).cpu().contiguous()
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()


def _experiment_extra(spec: ExperimentSpec) -> dict[str, Any]:
    return {
        "experiment_id": spec.experiment_id,
        "pair_id": spec.pair_id,
        "repeat_id": spec.repeat_id,
        "intervention": spec.intervention,
        "condition": spec.condition,
        "spec_fingerprint": spec_fingerprint(spec),
    }


def _execute_fresh(spec: ExperimentSpec, run_dir: Path, *, device: str, command: str) -> None:
    config = spec.to_run_config()
    extra = _experiment_extra(spec)

    # Build the exact quenched tensors for this arm, hash them, and inject
    # them into the training loop, so the run manifest hashes precisely what
    # was trained on (arms of a strict pair share seeds, hence — CPU
    # determinism — identical tensors; the pair manifest verifies that).
    state, teacher, train_data = build_state(config, device)
    validation_data = None
    if config.training.validation_size > 0:
        validation_data = teacher.sample_batch(
            config.training.validation_size,
            config.seeds.generator("validation_data_seed"),
        ).to(device)
    train_data_sha256 = _tensor_sha256(train_data)
    validation_data_sha256 = (
        _tensor_sha256(validation_data) if validation_data is not None else None
    )

    # -- train stage (mirrors maskeddiffusion-train wiring) ----------------
    train_dir = run_dir / "train"
    artifact = RunArtifact(train_dir)
    config.to_json(train_dir / "resolved_config.json")
    state, teacher, summary = train(
        config,
        device=device,
        state=state,
        teacher=teacher,
        train_data=train_data,
        validation_data=validation_data,
        on_log=artifact.log_metrics,
        checkpoint_dir=train_dir / "checkpoints",
    )
    final_ckpt = train_dir / "checkpoints" / "final.pt"
    artifact.register_file(final_ckpt, "final checkpoint")
    teacher_path = train_dir / "teacher.pt"
    teacher.save(teacher_path)
    artifact.register_file(teacher_path, "quenched teacher state")
    artifact.write_summary(summary)
    artifact.write_manifest(
        command=command,
        device=device,
        teacher_id=teacher.teacher_id,
        seeds=config.seeds.to_dict(),
        sampler=config.sampler.identity(),
        objective={
            "name": "continuous_time_masked_bce",
            "l2reg": config.training.l2reg,
            "min_time": config.training.min_time,
        },
        model=config.model.identity(),
        extra=extra,
    )

    # -- sample stage (mirrors maskeddiffusion-sample wiring) -------------
    samples_dir = run_dir / "samples"
    payload = load_checkpoint(final_ckpt)
    model_cfg_stored = dict(payload["model_config"])
    model_cfg_stored.pop("model", None)
    model = LinearMaskedScore(
        LinearScoreConfig(**model_cfg_stored), torch.Generator().manual_seed(0)
    )
    model.load_state_dict(payload["model_state"])
    model = model.to(device).eval()
    with torch.no_grad():
        result = run_sampler(
            model,
            config.sampler,
            config.n_generate,
            order_generator=config.seeds.generator("sampler_order_seed", device=device),
            token_generator=config.seeds.generator("sampler_token_seed", device=device),
        )
    artifact = RunArtifact(samples_dir)
    config.to_json(samples_dir / "resolved_config.json")
    artifact.save_tensor(
        "samples/samples.pt",
        result.values.cpu(),
        f"terminal samples of {config.sampler.sampler_name}",
    )
    artifact.log_metrics({"event": "sampled", "n_samples": config.n_generate})
    artifact.write_summary({"n_samples": config.n_generate, "sampler": config.sampler.identity()})
    artifact.write_manifest(
        command=command,
        device=device,
        teacher_id=payload["teacher_id"],
        seeds=config.seeds.to_dict(),
        sampler=config.sampler.identity(),
        objective={"name": "n/a (sampling run)"},
        model=payload["model_config"],
        input_paths=[str(final_ckpt)],
        extra={
            "checkpoint_id": payload.get("checkpoint_id"),
            "checkpoint_path": str(final_ckpt),
            "checkpoint_file_sha256": sha256_file(final_ckpt),
            **extra,
        },
    )

    # -- evaluate stage (full CLI provenance verification via evaluate_run)
    eval_dir = run_dir / "eval"
    evaluate_run(
        config=config,
        checkpoint=final_ckpt,
        teacher=teacher_path,
        samples_dir=samples_dir,
        out=eval_dir,
        n_true=spec.evaluation.n_true,
        lambdas=list(spec.evaluation.lambdas),
        requested_device=device,
        command=command,
        extra=extra,
    )

    # -- uturn stage (optional; reuses the already-loaded eval-mode model) -
    uturn_block: dict[str, Any] | None = None
    if spec.uturn is not None:
        uturn_dir = run_dir / "uturn"
        summary = run_uturn_stage(
            model=model,
            spec=spec,
            config=config,
            teacher=teacher,
            train_set=train_data,
            checkpoint_path=final_ckpt,
            checkpoint_id=payload.get("checkpoint_id"),
            run_dir=uturn_dir,
            device=device,
            command=command,
        )
        uturn_block = uturn_summary_to_record_block(summary)

    # -- run manifest (written LAST: its presence marks completion) -------
    run_manifest = {
        "schema_version": RUN_SCHEMA_VERSION,
        "status": "completed",
        "spec": spec.to_dict(),
        "spec_fingerprint": spec_fingerprint(spec),
        "teacher_id": teacher.teacher_id,
        "train_data_sha256": train_data_sha256,
        "validation_data_sha256": validation_data_sha256,
        "stages": {stage: stage for stage in STAGES},
        "completed_at": datetime.now(UTC).isoformat(),
        "environment": environment_metadata(device),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n")

    # -- canonical run record (maskeddiffusion.experiments.schema) --------
    # Built strictly from the manifests written above; the analysis layer
    # consumes only this record, never the run directory itself.
    write_run_record(build_run_record(run_dir, uturn=uturn_block), run_dir)


def _verify_completed_run(spec: ExperimentSpec, run_dir: Path) -> None:
    """Raise unless `run_dir` holds a valid completed run of exactly `spec`."""
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise ValueError(
            f"incomplete or foreign artifact at {run_dir}: missing run_manifest.json "
            "— the engine never overwrites partial output; move the directory aside "
            "manually to rerun this condition"
        )
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(
            f"incomplete or corrupt artifact at {run_dir}: malformed run_manifest.json "
            f"({e}); move the directory aside manually to rerun this condition"
        ) from e
    if manifest.get("schema_version") != RUN_SCHEMA_VERSION:
        raise ValueError(
            f"artifact at {run_dir} has schema_version {manifest.get('schema_version')!r}, "
            f"expected {RUN_SCHEMA_VERSION!r}; move the directory aside manually"
        )
    if manifest.get("status") != "completed":
        raise ValueError(
            f"incomplete artifact at {run_dir}: run_manifest status "
            f"{manifest.get('status')!r} != 'completed'; move the directory aside "
            "manually to rerun this condition"
        )
    recorded_spec = manifest.get("spec")
    if recorded_spec != spec.to_dict():
        differing = sorted(diff_leaves(recorded_spec, spec.to_dict()))
        raise ValueError(
            f"provenance-inconsistent artifact at {run_dir}: the recorded spec differs "
            f"from the current plan at {differing} — refusing to overwrite; move the "
            "directory aside manually to rerun this condition"
        )
    if manifest.get("spec_fingerprint") != spec_fingerprint(spec):
        raise ValueError(
            f"provenance-inconsistent artifact at {run_dir}: recorded spec_fingerprint "
            f"{manifest.get('spec_fingerprint')!r} != current {spec_fingerprint(spec)!r} "
            "— refusing to overwrite"
        )
    expected_config = spec.to_run_config().to_dict()
    stages = STAGES + (("uturn",) if spec.uturn is not None else ())
    for stage in stages:
        stage_dir = run_dir / stage
        stage_problems = validate_artifact(stage_dir)
        if stage_problems:
            raise ValueError(
                f"stage {stage!r} of completed run {run_dir} failed artifact "
                f"validation: {'; '.join(stage_problems)} — refusing to overwrite; "
                "move the directory aside manually to rerun this condition"
            )
        resolved = json.loads((stage_dir / "resolved_config.json").read_text())
        if resolved != expected_config:
            differing = sorted(diff_leaves(resolved, expected_config))
            raise ValueError(
                f"stage {stage!r} of completed run {run_dir} has a resolved_config.json "
                f"that differs from the current plan at {differing} — refusing to "
                "overwrite"
            )
    if spec.uturn is None and (run_dir / "uturn").exists():
        raise ValueError(
            f"completed run {run_dir} has a 'uturn' stage directory but the current "
            "plan's spec carries no [uturn] config — refusing to overwrite; move the "
            "directory aside manually to rerun this condition"
        )


def _substantive(manifest: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in manifest.items() if k not in ("completed_at", "environment")}


def _paired_disorder_checks(arms: dict[str, Any]) -> dict[str, Any]:
    """Checks for a `"paired_disorder"` group: disorder content must be
    identical across arms (same quenched `F`, same train/validation draws)."""
    teacher_id_equal = len({a["teacher_id"] for a in arms.values()}) == 1
    train_data_equal = len({a["train_data_sha256"] for a in arms.values()}) == 1
    validation_data_equal = len({a["validation_data_sha256"] for a in arms.values()}) == 1
    return {
        "teacher_id_equal": teacher_id_equal,
        "train_data_equal": train_data_equal,
        "validation_data_equal": validation_data_equal,
        "passed": teacher_id_equal and train_data_equal and validation_data_equal,
    }


def _matched_seed_finite_size_checks(
    specs: tuple[ExperimentSpec, ...], arms: dict[str, Any]
) -> dict[str, Any]:
    """Checks for a `"matched_seed_finite_size"` group (finite_d): arms
    share every seed VALUE and the aspect/sample ratios but are, by
    construction, DIFFERENT-shape teachers — never a same-disorder pairing.

    Positively asserts what `docs/PHASE4C_EXPERIMENT_PROTOCOL.md`'s
    `matched_seed_finite_size` rules require, rather than merely tolerating
    a mismatch the way a `"paired_disorder"` group's equality check does:
    same repeat index, same seed hierarchy values, same ratios, distinct
    `(D, N, M)`, and — the check that actually matters — DISTINCT
    `teacher_id` across every arm. A collision here is a real bug (the arms
    were not actually at different dimensions) and fails the run.
    """
    repeat_ids = {s.repeat_id for s in specs}
    seed_dicts = {tuple(sorted(s.seeds.to_dict().items())) for s in specs}
    ratio_pairs = {(s.dimensions.aspect_ratio, s.dimensions.sample_ratio) for s in specs}
    dim_triples = {
        (s.dimensions.latent_dim, s.dimensions.visible_dim, s.dimensions.train_size) for s in specs
    }
    teacher_ids = [a["teacher_id"] for a in arms.values()]
    same_repeat_index = len(repeat_ids) == 1
    same_seed_hierarchy_values = len(seed_dicts) == 1
    same_ratios = len(ratio_pairs) == 1
    distinct_dimensions = len(dim_triples) == len(specs)
    distinct_teacher_id = len(set(teacher_ids)) == len(teacher_ids)
    return {
        "same_repeat_index": same_repeat_index,
        "same_seed_hierarchy_values": same_seed_hierarchy_values,
        "same_ratios": same_ratios,
        "distinct_dimensions": distinct_dimensions,
        "distinct_teacher_id": distinct_teacher_id,
        "passed": (
            same_repeat_index
            and same_seed_hierarchy_values
            and same_ratios
            and distinct_dimensions
            and distinct_teacher_id
        ),
        "note": (
            "matched_seed_finite_size is NOT a same-disorder comparison: arms share "
            "seed values but have different-shape teachers by construction — never "
            "interpret this group as paired_disorder"
        ),
    }


def _write_pair_manifest(
    plan: ExperimentPlan,
    pair_id: str,
    specs: tuple[ExperimentSpec, ...],
    output_root: str | Path,
    device: str,
) -> Path:
    arms: dict[str, Any] = {}
    for spec in specs:
        run_dir = plan.run_dir(output_root, spec)
        run_manifest = json.loads((run_dir / "run_manifest.json").read_text())
        arms[spec.condition] = {
            "spec_fingerprint": run_manifest["spec_fingerprint"],
            "teacher_id": run_manifest["teacher_id"],
            "train_data_sha256": run_manifest["train_data_sha256"],
            "validation_data_sha256": run_manifest["validation_data_sha256"],
            "run_dir": str(run_dir),
        }
    if plan.comparison_type == "paired_disorder":
        comparison_checks = _paired_disorder_checks(arms)
    elif plan.comparison_type == "matched_seed_finite_size":
        comparison_checks = _matched_seed_finite_size_checks(specs, arms)
    else:
        # COMPARISON_TYPES is exhaustive and validated at Intervention construction.
        raise ValueError(f"unknown comparison_type {plan.comparison_type!r}")
    problems = validate_group(list(specs))
    manifest = {
        "schema_version": PAIR_SCHEMA_VERSION,
        "experiment_id": plan.experiment_id,
        "pair_id": pair_id,
        "repeat_id": specs[0].repeat_id,
        "intervention": plan.intervention,
        "comparison_type": plan.comparison_type,
        "conditions": [spec.condition for spec in specs],
        "arms": arms,
        "pair_validation": {"problems": problems},
        "comparison_checks": comparison_checks,
        "shared_seed_streams": specs[0].seeds.to_dict(),
        "completed_at": datetime.now(UTC).isoformat(),
        "environment": environment_metadata(device),
    }
    path = plan.experiment_root(output_root) / pair_id / "pair_manifest.json"
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            raise ValueError(
                f"existing pair manifest {path} is malformed ({e}); refusing to "
                "overwrite — move it aside manually"
            ) from e
        if existing.get("schema_version") != PAIR_SCHEMA_VERSION or _substantive(
            existing
        ) != _substantive(manifest):
            raise ValueError(
                f"existing pair manifest {path} is inconsistent with the current "
                "completed runs — refusing to overwrite; move it aside manually"
            )
        return path
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    if problems:
        raise ValueError(
            f"comparison group {pair_id!r} failed pair validation after execution: "
            f"{'; '.join(problems)}"
        )
    if not comparison_checks["passed"]:
        raise ValueError(
            f"comparison group {pair_id!r} (comparison_type={plan.comparison_type!r}) "
            f"failed its comparison checks (see comparison_checks in {path}) — do not "
            "interpret these runs"
        )
    return path


def execute_plan(
    plan: ExperimentPlan,
    output_root: str | Path,
    *,
    device: str,
    command: str,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write the pre-run manifest, then run or skip every condition resumably."""
    plan.write_plan_manifest(output_root, device=device, config_path=config_path)
    completed: list[str] = []
    skipped: list[str] = []
    for pair_id, specs in plan.groups().items():
        for spec in specs:
            run_dir = plan.run_dir(output_root, spec)
            if run_dir.exists():
                _verify_completed_run(spec, run_dir)
                # A completed run's record.json is immutable once written:
                # if present, only re-validate it — never rebuild or
                # rewrite (`_verify_completed_run` above already re-proved
                # every stage artifact valid, so there is nothing left to
                # discover by rebuilding). If absent — a run that completed
                # before a record existed for it, or whose record was lost —
                # build it once, explicitly marked as a migration
                # (docs/PHASE4C_EXPERIMENT_PROTOCOL.md §5), never silently.
                record_path = run_dir / "run_record.json"
                if record_path.exists():
                    load_run_record(record_path)
                else:
                    uturn_block = load_uturn_block(run_dir) if spec.uturn is not None else None
                    migration = backfill_migration_block(performed_at=datetime.now(UTC).isoformat())
                    write_run_record(
                        build_run_record(run_dir, uturn=uturn_block, migration=migration),
                        run_dir,
                    )
                skipped.append(str(run_dir))
            else:
                _execute_fresh(spec, run_dir, device=device, command=command)
                completed.append(str(run_dir))
        _write_pair_manifest(plan, pair_id, specs, output_root, device)
    return {
        "experiment_id": plan.experiment_id,
        "intervention": plan.intervention,
        "n_runs": len(plan.specs),
        "n_completed": len(completed),
        "n_skipped": len(skipped),
        "completed": completed,
        "skipped": skipped,
    }
