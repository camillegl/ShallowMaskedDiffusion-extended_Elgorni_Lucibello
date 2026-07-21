"""verify_artifact_hashes / load_run_record(verify_hashes=True): real linked
files, real tamper scenarios. This is what makes a `Phase4CRunRecord`'s
hashes worth anything — a stored hash nobody ever recomputes detects
nothing.
"""

from __future__ import annotations

import dataclasses
import hashlib

import pytest
from test_run_record_schema import _artifacts_block, make_record

from maskeddiffusion.experiments.schema import (
    load_run_record,
    verify_artifact_hashes,
    write_run_record,
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _build_consistent_run(tmp_path):
    """A run directory whose every linked file matches the record's stored
    hashes exactly — the fully-consistent baseline every test tampers
    with."""
    run_dir = tmp_path / "run"
    (run_dir / "train" / "checkpoints").mkdir(parents=True)

    resolved_config_bytes = b'{"dimensions": {"latent_dim": 8}}'
    (run_dir / "train" / "resolved_config.json").write_bytes(resolved_config_bytes)

    checkpoint_bytes = b"fake checkpoint content"
    (run_dir / "train" / "checkpoints" / "final.pt").write_bytes(checkpoint_bytes)

    artifacts: dict[str, dict[str, str]] = {}
    for stage in ("train", "samples", "eval"):
        stage_dir = run_dir / stage
        stage_dir.mkdir(exist_ok=True)
        manifest_path = stage_dir / "manifest.json"
        manifest_bytes = f'{{"stage": "{stage}"}}'.encode()
        manifest_path.write_bytes(manifest_bytes)
        artifacts[stage] = {
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256_bytes(manifest_bytes),
        }

    record = make_record(
        run_dir=str(run_dir),
        resolved_config_sha256=_sha256_bytes(resolved_config_bytes),
        checkpoint_file_sha256=_sha256_bytes(checkpoint_bytes),
        artifacts=artifacts,
    )
    # run_manifest.json's own hash is checked last; write it to match.
    run_manifest_bytes = b'{"schema_version": "maskeddiffusion.experiment_run.v1"}'
    (run_dir / "run_manifest.json").write_bytes(run_manifest_bytes)
    record = dataclasses.replace(record, run_manifest_sha256=_sha256_bytes(run_manifest_bytes))
    write_run_record(record, run_dir)
    return run_dir, record


def test_consistent_run_verifies_clean(tmp_path):
    run_dir, record = _build_consistent_run(tmp_path)
    assert verify_artifact_hashes(record) == []
    loaded = load_run_record(run_dir / "run_record.json")
    assert loaded == record


def test_tampered_resolved_config_file_detected(tmp_path):
    run_dir, record = _build_consistent_run(tmp_path)
    (run_dir / "train" / "resolved_config.json").write_bytes(b"{tampered}")
    problems = verify_artifact_hashes(record)
    assert any("resolved_config" in p and "mismatch" in p for p in problems)
    with pytest.raises(ValueError, match="failed artifact hash verification"):
        load_run_record(run_dir / "run_record.json")


def test_tampered_checkpoint_file_detected(tmp_path):
    run_dir, record = _build_consistent_run(tmp_path)
    (run_dir / "train" / "checkpoints" / "final.pt").write_bytes(b"tampered checkpoint")
    problems = verify_artifact_hashes(record)
    assert any("checkpoint" in p and "mismatch" in p for p in problems)
    with pytest.raises(ValueError, match="failed artifact hash verification"):
        load_run_record(run_dir / "run_record.json")


@pytest.mark.parametrize("stage", ["train", "samples", "eval"])
def test_tampered_stage_manifest_detected(tmp_path, stage):
    run_dir, record = _build_consistent_run(tmp_path)
    manifest_path = run_dir / stage / "manifest.json"
    manifest_path.write_bytes(b'{"stage": "tampered"}')
    problems = verify_artifact_hashes(record)
    assert any(f"artifacts[{stage!r}]" in p and "mismatch" in p for p in problems)


def test_tampered_uturn_manifest_detected(tmp_path):
    run_dir, record = _build_consistent_run(tmp_path)
    uturn_dir = run_dir / "uturn"
    uturn_dir.mkdir()
    manifest_bytes = b'{"stage": "uturn"}'
    manifest_path = uturn_dir / "manifest.json"
    manifest_path.write_bytes(manifest_bytes)
    with_uturn_artifacts = {
        **record.artifacts,
        "uturn": {
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256_bytes(manifest_bytes),
        },
    }
    record = dataclasses.replace(record, artifacts=with_uturn_artifacts)
    assert verify_artifact_hashes(record) == []

    manifest_path.write_bytes(b'{"stage": "tampered"}')
    problems = verify_artifact_hashes(record)
    assert any("artifacts['uturn']" in p and "mismatch" in p for p in problems)


def test_tampered_run_manifest_detected(tmp_path):
    run_dir, record = _build_consistent_run(tmp_path)
    (run_dir / "run_manifest.json").write_bytes(b"{tampered}")
    problems = verify_artifact_hashes(record)
    assert any("run_manifest" in p and "mismatch" in p for p in problems)


def test_missing_linked_file_detected(tmp_path):
    run_dir, record = _build_consistent_run(tmp_path)
    (run_dir / "train" / "checkpoints" / "final.pt").unlink()
    problems = verify_artifact_hashes(record)
    assert any("checkpoint" in p and "missing" in p for p in problems)


def test_verify_hashes_false_skips_verification(tmp_path):
    run_dir, record = _build_consistent_run(tmp_path)
    (run_dir / "train" / "resolved_config.json").write_bytes(b"{tampered}")
    # Structural load still succeeds; hash mismatch is simply not checked.
    loaded = load_run_record(run_dir / "run_record.json", verify_hashes=False)
    assert loaded == record


def test_artifacts_block_fixture_is_not_used_for_hash_checks_here():
    # Sanity: the shared fixture from test_run_record_schema still produces
    # a structurally valid (if not hash-verifiable) artifacts block.
    assert set(_artifacts_block()) == {"train", "samples", "eval"}
