"""End-to-end tiny flow on CPU: teacher -> data -> training -> checkpoint ->
sampling -> metrics -> artifact -> reload -> validation. Integration check
only; nothing here is scientifically interpretable."""

import json
from pathlib import Path

import pytest
import torch

from maskeddiffusion.artifacts import validate_artifact
from maskeddiffusion.checkpoints import load_checkpoint
from maskeddiffusion.cli.evaluate import main as evaluate_main
from maskeddiffusion.cli.sample import main as sample_main
from maskeddiffusion.cli.train import main as train_main
from maskeddiffusion.teacher import HiddenManifoldTeacher

CONFIG = "configs/smoke/smoke.toml"


def _run_train_and_sample(tmp_path, name, config=CONFIG):
    run = tmp_path / f"{name}-run"
    samples_dir = tmp_path / f"{name}-samples"
    assert train_main(["--config", config, "--output", str(run), "--device", "cpu"]) == 0
    ckpt = run / "checkpoints" / "final.pt"
    assert (
        sample_main(
            [
                "--config",
                config,
                "--output",
                str(samples_dir),
                "--checkpoint",
                str(ckpt),
                "--n-samples",
                "4",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    return run, ckpt, samples_dir


def test_full_smoke_flow(tmp_path):
    run = tmp_path / "run"
    samples_dir = tmp_path / "samples"
    eval_dir = tmp_path / "eval"

    assert train_main(["--config", CONFIG, "--output", str(run), "--device", "cpu"]) == 0
    ckpt = run / "checkpoints" / "final.pt"
    assert ckpt.exists()
    assert (run / "teacher.pt").exists()
    assert validate_artifact(run) == []

    # reload checkpoint and teacher; ids consistent
    payload = load_checkpoint(ckpt)
    teacher = HiddenManifoldTeacher.load(run / "teacher.pt")
    assert payload["teacher_id"] == teacher.teacher_id
    assert payload["step"] == 60

    assert (
        sample_main(
            [
                "--config",
                CONFIG,
                "--output",
                str(samples_dir),
                "--checkpoint",
                str(ckpt),
                "--n-samples",
                "6",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    samples = torch.load(samples_dir / "samples" / "samples.pt", weights_only=False)
    assert samples.shape == (6, teacher.dims.visible_dim)
    assert ((samples == 1.0) | (samples == -1.0)).all()
    assert validate_artifact(samples_dir) == []

    assert (
        evaluate_main(
            [
                "--config",
                CONFIG,
                "--output",
                str(eval_dir),
                "--checkpoint",
                str(ckpt),
                "--teacher",
                str(run / "teacher.pt"),
                "--samples",
                str(samples_dir),
                "--n-true",
                "50",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    summary = json.loads((eval_dir / "summary.json").read_text())
    for comparison in ("model_vs_true", "true_vs_true", "train_vs_true", "model_vs_train"):
        assert "mixture_biased_mmd2" in summary[comparison]
    manifest = json.loads((eval_dir / "manifest.json").read_text())
    assert manifest["checkpoint_id"] == payload["checkpoint_id"]
    assert manifest["requested_device"] == "cpu"
    assert manifest["actual_device"] == "cpu"


def test_dry_run_writes_nothing(tmp_path, capsys):
    out = tmp_path / "dry"
    assert (
        train_main(["--config", CONFIG, "--output", str(out), "--device", "cpu", "--dry-run"]) == 0
    )
    assert not out.exists()
    plan = json.loads(capsys.readouterr().out)
    assert plan["resolved_dimensions"]["visible_dim"] == 32
    assert plan["resolved_dimensions"]["train_size"] == 48


def test_evaluate_rejects_bare_tensor_path_for_samples(tmp_path):
    """--samples must be a sample-run artifact directory, not the raw
    samples.pt file — the CLI needs the directory's manifest.json to verify
    provenance (teacher_id, checkpoint_id, sampler identity)."""
    _run, ckpt, samples_dir = _run_train_and_sample(tmp_path, "a")
    teacher_path = tmp_path / "a-run" / "teacher.pt"
    with pytest.raises(ValueError, match="manifest.json"):
        evaluate_main(
            [
                "--config",
                CONFIG,
                "--output",
                str(tmp_path / "eval"),
                "--checkpoint",
                str(ckpt),
                "--teacher",
                str(teacher_path),
                "--samples",
                str(samples_dir / "samples" / "samples.pt"),  # bare tensor, not a dir
                "--device",
                "cpu",
            ]
        )


def test_evaluate_rejects_samples_from_different_teacher(tmp_path):
    """A --samples artifact generated under a different teacher (here, a
    different base_seed) must be rejected, not silently scored against the
    wrong finite-F law."""
    _run_a, ckpt_a, samples_a = _run_train_and_sample(tmp_path, "a")

    alt_config_path = tmp_path / "alt.toml"
    alt_config_path.write_text(_smoke_toml_with_seed(99999))
    _run_b, ckpt_b, _samples_b = _run_train_and_sample(tmp_path, "b", str(alt_config_path))

    with pytest.raises(ValueError, match="teacher_id"):
        evaluate_main(
            [
                "--config",
                str(alt_config_path),
                "--output",
                str(tmp_path / "eval-mismatch"),
                "--checkpoint",
                str(ckpt_b),
                "--teacher",
                str(_run_b / "teacher.pt"),
                "--samples",
                str(samples_a),  # samples from run "a"'s teacher, not run "b"'s
                "--device",
                "cpu",
            ]
        )


def _smoke_toml_with_seed(base_seed: int) -> str:
    text = Path(CONFIG).read_text().replace("base_seed = 12345", f"base_seed = {base_seed}")
    assert f"base_seed = {base_seed}" in text, "smoke.toml's base_seed line format changed"
    return text


def test_evaluate_rejects_samples_from_stale_checkpoint_id(tmp_path):
    """Same teacher, but the sample artifact's recorded checkpoint_id no
    longer matches --checkpoint's own content hash (e.g. the checkpoint
    file was retrained/overwritten after the samples were generated) — must
    be rejected, not silently scored as if nothing changed."""
    _run, ckpt, samples_dir = _run_train_and_sample(tmp_path, "a")

    manifest_path = samples_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["checkpoint_id"]  # sanity: the field is actually populated
    manifest["checkpoint_id"] = "ckpt-0000000000000000"  # tamper: stale/wrong hash
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="checkpoint_id"):
        evaluate_main(
            [
                "--config",
                CONFIG,
                "--output",
                str(tmp_path / "eval-stale"),
                "--checkpoint",
                str(ckpt),
                "--teacher",
                str(_run / "teacher.pt"),
                "--samples",
                str(samples_dir),
                "--device",
                "cpu",
            ]
        )
