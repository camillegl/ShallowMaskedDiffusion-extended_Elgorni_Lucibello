"""End-to-end reproducibility of `maskeddiffusion-sample`: same config
(fixed sampler_order_seed/sampler_token_seed) and checkpoint reproduce a
bit-identical samples.pt, and the resolved config / manifest both record the
two sampler seeds distinctly. Integration check only, not a scientific claim
of statistical independence between the two RNG streams."""

from __future__ import annotations

import json

import torch

from maskeddiffusion.cli.sample import main as sample_main
from maskeddiffusion.cli.train import main as train_main

CONFIG = "configs/smoke/smoke.toml"


def _train_once(tmp_path):
    run = tmp_path / "run"
    assert train_main(["--config", CONFIG, "--output", str(run), "--device", "cpu"]) == 0
    return run / "checkpoints" / "final.pt"


def test_sample_cli_deterministic_replay(tmp_path):
    ckpt = _train_once(tmp_path)

    out_a = tmp_path / "samples_a"
    out_b = tmp_path / "samples_b"
    for out in (out_a, out_b):
        assert (
            sample_main(
                [
                    "--config",
                    CONFIG,
                    "--output",
                    str(out),
                    "--checkpoint",
                    str(ckpt),
                    "--n-samples",
                    "8",
                    "--device",
                    "cpu",
                ]
            )
            == 0
        )

    samples_a = torch.load(out_a / "samples" / "samples.pt", weights_only=False)
    samples_b = torch.load(out_b / "samples" / "samples.pt", weights_only=False)
    assert torch.equal(samples_a, samples_b)


def test_sample_cli_records_distinct_sampler_seeds_in_config_and_manifest(tmp_path):
    ckpt = _train_once(tmp_path)
    out = tmp_path / "samples"
    assert (
        sample_main(
            [
                "--config",
                CONFIG,
                "--output",
                str(out),
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

    resolved = json.loads((out / "resolved_config.json").read_text())
    order_seed = resolved["seeds"]["sampler_order_seed"]
    token_seed = resolved["seeds"]["sampler_token_seed"]
    assert order_seed != token_seed

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["seeds"]["sampler_order_seed"] == order_seed
    assert manifest["seeds"]["sampler_token_seed"] == token_seed
