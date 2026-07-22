"""Phase4CRunRecord: validation, round-trip, tamper detection, file I/O."""

from __future__ import annotations

import dataclasses
import json

import pytest

from maskeddiffusion.dimensions import Dimensions
from maskeddiffusion.experiments.schema import (
    RUN_RECORD_SCHEMA_VERSION,
    Phase4CRunRecord,
    load_run_record,
    model_config_digest,
    write_run_record,
)
from maskeddiffusion.randomness import SeedHierarchy


def _mmd_block(center: float) -> dict:
    return {
        "mixture_biased_mmd2": center,
        "mixture_unbiased_mmd2_raw": center - 1e-4,
        "biased_mmd2": {"4.0": center * 0.8, "8.0": center * 1.2},
        "unbiased_mmd2_raw": {"4.0": center * 0.7, "8.0": center * 1.1},
    }


def _artifacts_block() -> dict[str, dict[str, str]]:
    return {
        "train": {"manifest_path": "/tmp/run/train/manifest.json", "manifest_sha256": "1" * 64},
        "samples": {
            "manifest_path": "/tmp/run/samples/manifest.json",
            "manifest_sha256": "2" * 64,
        },
        "eval": {"manifest_path": "/tmp/run/eval/manifest.json", "manifest_sha256": "3" * 64},
    }


def make_record(**overrides) -> Phase4CRunRecord:
    defaults = {
        "status": "completed",
        "experiment_id": "exp-a",
        "pair_id": "exp-a-r000",
        "repeat_id": 0,
        "intervention": "v_trainability",
        "condition": "frozen_zero_v",
        "dimensions": Dimensions.resolve(latent_dim=8, aspect_ratio=2.0, sample_ratio=3.0),
        "seeds": SeedHierarchy.from_base(7),
        "spec_fingerprint": "expspec-0011223344556677",
        "resolved_config": {"dimensions": {"latent_dim": 8}, "n_generate": 4},
        "resolved_config_sha256": "d" * 64,
        "teacher_id": "hmt-abc",
        "checkpoint_id": "ckpt-abc",
        "checkpoint_file_sha256": "f" * 64,
        "sampler": {"sampler_name": "sequential_random_stochastic", "tokens_per_step": 1},
        "model": {
            "model": "linear_masked_score",
            "visible_dim": 16,
            "normalization": "explicit_sqrt_n",
            "v_policy": "frozen_zero",
            "bias_policy": "none",
            "diagonal_policy": "zero",
        },
        "artifacts": _artifacts_block(),
        "mmd": {
            "model_vs_true": _mmd_block(2e-2),
            "true_vs_true": _mmd_block(1e-3),
            "train_vs_true": _mmd_block(4e-2),
            "model_vs_train": _mmd_block(1.5e-2),
        },
        "nearest_training": {
            "model_mean_nearest_overlap": 0.62,
            "true_mean_nearest_overlap": 0.57,
            "excess": 0.05,
        },
        "pair_correlation_error": {"max_abs_error": 0.11, "rms_error": 0.03, "mean_error": 0.001},
        "optimization_step": 4,
        "examples_seen": 32,
        "train_data_sha256": "a" * 64,
        "validation_data_sha256": "b" * 64,
        "git_commit": "e" * 40,
        "run_dir": "/tmp/exp-a/exp-a-r000/frozen_zero_v",
        "run_manifest_sha256": "c" * 64,
    }
    defaults.update(overrides)
    return Phase4CRunRecord(**defaults)


def test_round_trip_preserves_record():
    record = make_record()
    payload = record.to_dict()
    assert payload["schema_version"] == RUN_RECORD_SCHEMA_VERSION
    restored = Phase4CRunRecord.from_dict(json.loads(json.dumps(payload)))
    assert restored == record
    assert restored.record_validated is True


def test_validation_problems_flow_through():
    record = make_record(validation_problems=("stage train: hash mismatch",))
    assert record.record_validated is False
    restored = Phase4CRunRecord.from_dict(record.to_dict())
    assert restored.validation_problems == ("stage train: hash mismatch",)


def test_mmd_must_have_exactly_four_comparisons():
    record = make_record()
    incomplete = {k: v for k, v in record.mmd.items() if k != "model_vs_train"}
    with pytest.raises(ValueError, match="comparisons"):
        make_record(mmd=incomplete)
    with pytest.raises(ValueError, match="comparisons"):
        make_record(mmd={**record.mmd, "extra_cmp": _mmd_block(1e-3)})


def test_mmd_blocks_validated():
    bad = _mmd_block(1e-2)
    bad["mixture_biased_mmd2"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        make_record(mmd={**make_record().mmd, "model_vs_true": bad})
    mismatched = _mmd_block(1e-2)
    mismatched["unbiased_mmd2_raw"] = {"4.0": 1e-3}
    with pytest.raises(ValueError, match="kernel-scale sets differ"):
        make_record(mmd={**make_record().mmd, "model_vs_true": mismatched})


def test_negative_unbiased_mmd2_is_legitimate():
    block = _mmd_block(1e-3)
    block["mixture_unbiased_mmd2_raw"] = -2e-4
    block["unbiased_mmd2_raw"] = {"4.0": -1e-4, "8.0": -3e-4}
    record = make_record(mmd={**make_record().mmd, "true_vs_true": block})
    assert record.mmd["true_vs_true"]["mixture_unbiased_mmd2_raw"] == -2e-4


def test_uturn_validation():
    good = {
        "mask_densities": [0.2, 0.5],
        "overlap": [0.9, 0.7],
        "baseline_recovery": 0.6,
        "excess_recovery": 0.15,
    }
    assert make_record(uturn=good).uturn == good
    with pytest.raises(ValueError, match="equal-length"):
        make_record(uturn={**good, "overlap": [0.9]})
    with pytest.raises(ValueError, match="keys"):
        make_record(uturn={**good, "stray": 1})


def test_sampler_identity_required():
    with pytest.raises(ValueError, match="sampler_name"):
        make_record(sampler={"tokens_per_step": 1})
    with pytest.raises(ValueError, match="tokens_per_step"):
        make_record(sampler={"sampler_name": "one_shot_stochastic", "tokens_per_step": 0})


def test_status_must_be_a_known_completion_status():
    with pytest.raises(ValueError, match="status must be one of"):
        make_record(status="partial")
    with pytest.raises(ValueError, match="status must be one of"):
        make_record(status="")


def test_resolved_config_must_be_nonempty_dict():
    with pytest.raises(ValueError, match="resolved_config"):
        make_record(resolved_config={})
    with pytest.raises(ValueError, match="resolved_config"):
        make_record(resolved_config=None)


def test_resolved_config_sha256_required():
    with pytest.raises(ValueError, match="resolved_config_sha256"):
        make_record(resolved_config_sha256="")


def test_git_commit_optional_but_not_blank():
    assert make_record(git_commit=None).git_commit is None
    with pytest.raises(ValueError, match="git_commit"):
        make_record(git_commit="")


def test_artifacts_requires_train_samples_eval():
    for stage in ("train", "samples", "eval"):
        incomplete = _artifacts_block()
        del incomplete[stage]
        with pytest.raises(ValueError, match="missing required stages"):
            make_record(artifacts=incomplete)


def test_artifacts_rejects_unknown_stage():
    bogus = {**_artifacts_block(), "bogus_stage": {"manifest_path": "x", "manifest_sha256": "y"}}
    with pytest.raises(ValueError, match="unknown stages"):
        make_record(artifacts=bogus)


def test_artifacts_entry_requires_path_and_sha256():
    bad = _artifacts_block()
    bad["train"] = {"manifest_path": "/tmp/x"}
    with pytest.raises(ValueError, match="must be a dict with keys"):
        make_record(artifacts=bad)
    bad2 = _artifacts_block()
    bad2["train"] = {"manifest_path": "", "manifest_sha256": "1" * 64}
    with pytest.raises(ValueError, match="nonempty string"):
        make_record(artifacts=bad2)


def test_artifacts_uturn_optional():
    with_uturn = _artifacts_block()
    with_uturn["uturn"] = {
        "manifest_path": "/tmp/run/uturn/manifest.json",
        "manifest_sha256": "9" * 64,
    }
    record = make_record(artifacts=with_uturn)
    assert set(record.artifacts) == {"train", "samples", "eval", "uturn"}


def test_from_dict_detects_digest_tampering():
    payload = make_record().to_dict()
    payload["model_config_digest"] = "modelcfg-deadbeefdeadbeef"
    with pytest.raises(ValueError, match="tampered"):
        Phase4CRunRecord.from_dict(payload)


def test_from_dict_rejects_wrong_schema_and_keys():
    payload = make_record().to_dict()
    with pytest.raises(ValueError, match="schema_version"):
        Phase4CRunRecord.from_dict({**payload, "schema_version": "v0"})
    with pytest.raises(ValueError, match="unknown"):
        Phase4CRunRecord.from_dict({**payload, "stray": 1})
    missing = dict(payload)
    del missing["mmd"]
    with pytest.raises(ValueError, match="missing"):
        Phase4CRunRecord.from_dict(missing)


def test_model_config_digest_is_content_sensitive():
    record = make_record()
    other = dict(record.model, v_policy="trainable")
    assert model_config_digest(record.model) != model_config_digest(other)
    assert record.model_config_digest.startswith("modelcfg-")


def test_write_and_load_run_record(tmp_path):
    record = make_record(run_dir=str(tmp_path))
    path = write_run_record(record, tmp_path)
    # This fixture's run_dir has no real linked files (train/resolved_config
    # .json, artifact manifests, checkpoint) — hash re-verification is
    # covered with real files in test_run_record_hash_verification.py.
    assert load_run_record(path, verify_hashes=False) == record
    # Idempotent for an identical record...
    assert write_run_record(record, tmp_path) == path
    # ...but refuses a disagreeing one.
    other = dataclasses.replace(record, examples_seen=999)
    with pytest.raises(ValueError, match="disagrees"):
        write_run_record(other, tmp_path)


def test_write_refuses_malformed_existing(tmp_path):
    (tmp_path / "run_record.json").write_text("{not json")
    with pytest.raises(ValueError, match="malformed"):
        write_run_record(make_record(run_dir=str(tmp_path)), tmp_path)
