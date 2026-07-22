"""maskeddiffusion-uturn CLI end-to-end: tiny CPU train -> U-turn curve ->
artifact. Integration check only; nothing here is scientifically
interpretable."""

import json
from pathlib import Path

import pytest
import torch

from maskeddiffusion.artifacts import validate_artifact
from maskeddiffusion.checkpoints import load_checkpoint
from maskeddiffusion.cli.train import main as train_main
from maskeddiffusion.cli.uturn import main as uturn_main

CONFIG = "configs/smoke/smoke.toml"
GREEDY_CONFIG = "configs/smoke/smoke_greedy.toml"


def _train(tmp_path, name="run", config=CONFIG):
    run = tmp_path / name
    assert train_main(["--config", config, "--output", str(run), "--device", "cpu"]) == 0
    return run, run / "checkpoints" / "final.pt", run / "teacher.pt"


def _uturn(tmp_path, ckpt, teacher_path, name="uturn", config=CONFIG, extra_args=None):
    out = tmp_path / name
    argv = [
        "--config",
        config,
        "--output",
        str(out),
        "--checkpoint",
        str(ckpt),
        "--teacher",
        str(teacher_path),
        "--t-values",
        "0.0",
        "0.5",
        "--n-examples",
        "4",
        "--device",
        "cpu",
    ]
    if extra_args:
        argv += extra_args
    assert uturn_main(argv) == 0
    return out


def test_uturn_cli_end_to_end(tmp_path):
    run, ckpt, teacher_path = _train(tmp_path)
    out = _uturn(tmp_path, ckpt, teacher_path)

    assert validate_artifact(out) == []
    summary = json.loads((out / "summary.json").read_text())
    assert summary["experiment"] == "uturn_reconstruction"
    assert summary["sources"] == ["train", "fresh"]
    assert summary["t_values"] == [0.0, 0.5]
    assert len(summary["points"]) == 2 * 2  # sources x t values
    for point in summary["points"]:
        assert point["no_recovery_baseline"] == 1.0 - point["t_value"]
        assert point["excess_recovery_mean"] == pytest.approx(
            point["q_u_mean"] - point["no_recovery_baseline"]
        )
        if point["t_value"] == 0.0:
            # t=0 -> exact reconstruction, both sources
            assert point["q_u_mean"] == 1.0
            assert point["hamming_error_mean"] == 0.0
    assert len(summary["train_fresh_comparison"]) == 2

    payload = load_checkpoint(ckpt)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["sampler"]["sampler_name"] == "sequential_random_stochastic"
    assert manifest["teacher_id"] == payload["teacher_id"]
    assert manifest["checkpoint_id"] == payload["checkpoint_id"]
    assert manifest["objective"]["name"] == "uturn_reconstruction"
    assert manifest["checkpoint_train_size"] == 48

    per_example_lines = (out / "results" / "per_example.jsonl").read_text().strip().split("\n")
    assert len(per_example_lines) == 2 * 4 * 2  # sources x examples x t values
    cell = json.loads(per_example_lines[0])
    for key in (
        "source",
        "example_index",
        "t_value",
        "q_u",
        "excess_over_baseline",
        "hamming_error",
        "nearest_train_overlap_recon",
    ):
        assert key in cell

    tensors = torch.load(out / "results" / "uturn_tensors.pt", weights_only=False)
    assert tensors["masks"].shape == (2, 4, 32)  # (t, example, N)
    assert tensors["clean"]["train"].shape == (4, 32)
    assert tensors["reconstructions"]["fresh"].shape == (2, 4, 32)
    # paired masks: observed entries of every reconstruction equal the clean row
    mask = tensors["masks"]
    for source in ("train", "fresh"):
        recon = tensors["reconstructions"][source]
        clean = tensors["clean"][source].unsqueeze(0).expand_as(recon)
        assert torch.equal(recon[~mask], clean[~mask])
    assert run  # silence unused


def test_uturn_cli_deterministic_replay(tmp_path):
    _run, ckpt, teacher_path = _train(tmp_path)
    out_a = _uturn(tmp_path, ckpt, teacher_path, name="uturn-a")
    out_b = _uturn(tmp_path, ckpt, teacher_path, name="uturn-b")
    assert json.loads((out_a / "summary.json").read_text()) == json.loads(
        (out_b / "summary.json").read_text()
    )
    assert (out_a / "results" / "per_example.jsonl").read_text() == (
        out_b / "results" / "per_example.jsonl"
    ).read_text()
    tensors_a = torch.load(out_a / "results" / "uturn_tensors.pt", weights_only=False)
    tensors_b = torch.load(out_b / "results" / "uturn_tensors.pt", weights_only=False)
    for source in ("train", "fresh"):
        assert torch.equal(
            tensors_a["reconstructions"][source], tensors_b["reconstructions"][source]
        )


def test_uturn_cli_greedy_sampler_against_stochastic_checkpoint(tmp_path):
    """The sampler identity is a property of the U-turn run's --config, not
    of how the checkpoint was trained: a greedy U-turn run against a
    stochastically-trained checkpoint is valid and recorded honestly."""
    _run, ckpt, teacher_path = _train(tmp_path)
    out = _uturn(tmp_path, ckpt, teacher_path, config=GREEDY_CONFIG)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["sampler"]["sampler_name"] == "sequential_random_greedy"
    assert manifest["sampler"]["token_selection"] == "threshold_at_half"
    assert validate_artifact(out) == []


def test_uturn_cli_supports_trainable_v_checkpoint(tmp_path):
    """Both checkpoint families work: train a v_policy=trainable checkpoint,
    then U-turn it (the model config is restored from the checkpoint)."""
    config_path = tmp_path / "trainable_v.toml"
    text = Path(CONFIG).read_text().replace('v_policy = "frozen_zero"', 'v_policy = "trainable"')
    assert 'v_policy = "trainable"' in text
    config_path.write_text(text)
    _run, ckpt, teacher_path = _train(tmp_path, name="run-v", config=str(config_path))
    out = _uturn(tmp_path, ckpt, teacher_path, config=str(config_path))
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["model"]["v_policy"] == "trainable"
    assert validate_artifact(out) == []


def test_uturn_cli_rejects_teacher_mismatch(tmp_path):
    _run_a, ckpt_a, _teacher_a = _train(tmp_path, name="run-a")
    alt_config = tmp_path / "alt.toml"
    text = Path(CONFIG).read_text().replace("base_seed = 12345", "base_seed = 99999")
    assert "base_seed = 99999" in text
    alt_config.write_text(text)
    _run_b, _ckpt_b, teacher_b = _train(tmp_path, name="run-b", config=str(alt_config))
    with pytest.raises(ValueError, match="teacher_id"):
        uturn_main(
            [
                "--config",
                CONFIG,
                "--output",
                str(tmp_path / "uturn-mismatch"),
                "--checkpoint",
                str(ckpt_a),
                "--teacher",
                str(teacher_b),
                "--t-values",
                "0.5",
                "--device",
                "cpu",
            ]
        )


def test_uturn_cli_rejects_out_of_range_t(tmp_path):
    _run, ckpt, teacher_path = _train(tmp_path)
    with pytest.raises(ValueError, match="0 <= t <= 1"):
        uturn_main(
            [
                "--config",
                CONFIG,
                "--output",
                str(tmp_path / "uturn-bad-t"),
                "--checkpoint",
                str(ckpt),
                "--teacher",
                str(teacher_path),
                "--t-values",
                "1.5",
                "--device",
                "cpu",
            ]
        )


def test_uturn_cli_rejects_more_examples_than_train_size(tmp_path):
    _run, ckpt, teacher_path = _train(tmp_path)
    with pytest.raises(ValueError, match="exceeds"):
        uturn_main(
            [
                "--config",
                CONFIG,
                "--output",
                str(tmp_path / "uturn-too-many"),
                "--checkpoint",
                str(ckpt),
                "--teacher",
                str(teacher_path),
                "--t-values",
                "0.5",
                "--n-examples",
                "49",  # smoke train_size is 48
                "--device",
                "cpu",
            ]
        )


def test_uturn_cli_dry_run_writes_nothing(tmp_path, capsys):
    out = tmp_path / "dry"
    assert (
        uturn_main(
            [
                "--config",
                CONFIG,
                "--output",
                str(out),
                "--checkpoint",
                "nonexistent.pt",
                "--teacher",
                "nonexistent.pt",
                "--t-values",
                "0.0",
                "0.5",
                "--n-examples",
                "4",
                "--dry-run",
            ]
        )
        == 0
    )
    assert not out.exists()
    plan = json.loads(capsys.readouterr().out)
    assert plan["experiment"] == "uturn_reconstruction"
    assert plan["t_values"] == [0.0, 0.5]
