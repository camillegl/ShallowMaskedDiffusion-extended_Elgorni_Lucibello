import json

import torch

from maskeddiffusion.artifacts import RunArtifact, validate_artifact


def make_valid_artifact(root) -> RunArtifact:
    art = RunArtifact(root)
    art.log_metrics({"step": 1, "train_loss": 0.5})
    art.save_tensor("samples/x.pt", torch.ones(2, 3), "test tensor")
    art.write_summary({"final": 1})
    art.write_manifest(
        command="test",
        device="cpu",
        teacher_id="hmt-test",
        seeds={
            s: 1
            for s in (
                "teacher_seed",
                "train_data_seed",
                "validation_data_seed",
                "evaluation_data_seed",
                "model_seed",
                "mask_seed",
                "dataloader_seed",
                "sampler_order_seed",
                "sampler_token_seed",
                "metric_seed",
            )
        },
        sampler={"sampler_name": "sequential_random_stochastic", "tokens_per_step": 1},
        objective={"name": "continuous_time_masked_bce"},
        model={"model": "linear_masked_score", "visible_dim": 3},
    )
    (root / "resolved_config.json").write_text(
        json.dumps(
            {
                "dimensions": {
                    "latent_dim": 1,
                    "aspect_ratio": 3.0,
                    "sample_ratio": 1.0,
                    "visible_dim": 3,
                    "train_size": 1,
                },
            }
        )
    )
    return art


def test_valid_artifact_accepted(tmp_path):
    make_valid_artifact(tmp_path)
    assert validate_artifact(tmp_path) == []


def test_missing_required_file_rejected(tmp_path):
    make_valid_artifact(tmp_path)
    (tmp_path / "summary.json").unlink()
    problems = validate_artifact(tmp_path)
    assert any("summary.json" in p for p in problems)


def test_missing_seed_stream_rejected(tmp_path):
    make_valid_artifact(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    del manifest["seeds"]["mask_seed"]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    problems = validate_artifact(tmp_path)
    assert any("mask_seed" in p for p in problems)


def test_missing_sampler_rejected(tmp_path):
    make_valid_artifact(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["sampler"] = {}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    problems = validate_artifact(tmp_path)
    assert any("sampler" in p for p in problems)


def test_inconsistent_dimensions_rejected(tmp_path):
    make_valid_artifact(tmp_path)
    (tmp_path / "resolved_config.json").write_text(
        json.dumps(
            {
                "dimensions": {
                    "latent_dim": 1,
                    "aspect_ratio": 3.0,
                    "sample_ratio": 1.0,
                    "visible_dim": 99,
                    "train_size": 1,
                },
            }
        )
    )
    problems = validate_artifact(tmp_path)
    assert any("visible_dim" in p for p in problems)


def test_hash_mismatch_rejected(tmp_path):
    make_valid_artifact(tmp_path)
    torch.save(torch.zeros(2, 3), tmp_path / "samples" / "x.pt")
    problems = validate_artifact(tmp_path)
    assert any("hash mismatch" in p for p in problems)


def test_malformed_manifest_rejected(tmp_path):
    make_valid_artifact(tmp_path)
    (tmp_path / "manifest.json").write_text("{not json")
    problems = validate_artifact(tmp_path)
    assert any("malformed manifest" in p for p in problems)
