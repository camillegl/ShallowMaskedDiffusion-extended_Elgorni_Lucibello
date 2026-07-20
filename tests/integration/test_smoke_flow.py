"""End-to-end tiny flow on CPU: teacher -> data -> training -> checkpoint ->
sampling -> metrics -> artifact -> reload -> validation. Integration check
only; nothing here is scientifically interpretable."""

import json

import torch

from maskeddiffusion.artifacts import validate_artifact
from maskeddiffusion.checkpoints import load_checkpoint
from maskeddiffusion.cli.evaluate import main as evaluate_main
from maskeddiffusion.cli.sample import main as sample_main
from maskeddiffusion.cli.train import main as train_main
from maskeddiffusion.teacher import HiddenManifoldTeacher

CONFIG = "configs/smoke/smoke.toml"


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
                str(samples_dir / "samples" / "samples.pt"),
                "--n-true",
                "50",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    summary = json.loads((eval_dir / "summary.json").read_text())
    for comparison in ("model_vs_true", "true_vs_true", "train_vs_true"):
        assert "mixture_biased_mmd2" in summary[comparison]


def test_dry_run_writes_nothing(tmp_path, capsys):
    out = tmp_path / "dry"
    assert (
        train_main(["--config", CONFIG, "--output", str(out), "--device", "cpu", "--dry-run"]) == 0
    )
    assert not out.exists()
    plan = json.loads(capsys.readouterr().out)
    assert plan["resolved_dimensions"]["visible_dim"] == 32
    assert plan["resolved_dimensions"]["train_size"] == 48
