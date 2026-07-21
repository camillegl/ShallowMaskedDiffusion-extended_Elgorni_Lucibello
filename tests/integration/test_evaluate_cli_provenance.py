"""Provenance checks for `maskeddiffusion-evaluate`:

- the training set scored against is reconstructed from the *checkpoint's*
  own recorded config (train_size, train_data_seed), never from whatever
  --config happens to be passed to the evaluate invocation;
- the --samples artifact is validated (manifest-recorded hashes checked)
  before its tensor is loaded and scored.
"""

from __future__ import annotations

import json
from pathlib import Path

from maskeddiffusion.cli.evaluate import main as evaluate_main
from maskeddiffusion.cli.sample import main as sample_main
from maskeddiffusion.cli.train import main as train_main

CONFIG = "configs/smoke/smoke.toml"

# Same visible_dim (32) as smoke.toml, but a different train_size (sample_ratio
# 3.0 -> train_size 24, vs. smoke.toml's 48) and a different base_seed -- so a
# different train_data_seed too. If evaluate ever reconstructed the training
# set from *this* config instead of the checkpoint's own, results would differ
# from an evaluate run against the checkpoint's original config.
MISMATCHED_CONFIG = """
n_generate = 0

[dimensions]
latent_dim = 8
aspect_ratio = 4.0
sample_ratio = 3.0

[seeds]
base_seed = 999

[model]
normalization = "explicit_sqrt_n"
v_policy = "frozen_zero"
bias_policy = "none"
diagonal_policy = "zero"

[training]
max_steps = 1
batch_size = 4

[sampler]
sampler_name = "sequential_random_stochastic"
"""


def _train_and_sample(tmp_path):
    run = tmp_path / "run"
    assert train_main(["--config", CONFIG, "--output", str(run), "--device", "cpu"]) == 0
    ckpt = run / "checkpoints" / "final.pt"
    samples = tmp_path / "samples"
    assert (
        sample_main(
            [
                "--config",
                CONFIG,
                "--output",
                str(samples),
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
    return ckpt, samples


def test_evaluate_reconstructs_train_set_from_checkpoint_not_cli_config(tmp_path):
    ckpt, samples = _train_and_sample(tmp_path)

    mismatched = tmp_path / "mismatched.toml"
    mismatched.write_text(MISMATCHED_CONFIG)

    out_same = tmp_path / "eval_same"
    out_mismatched = tmp_path / "eval_mismatched"
    for cfg, out in ((CONFIG, out_same), (str(mismatched), out_mismatched)):
        assert (
            evaluate_main(
                [
                    "--config",
                    cfg,
                    "--output",
                    str(out),
                    "--checkpoint",
                    str(ckpt),
                    "--teacher",
                    str(tmp_path / "run" / "teacher.pt"),
                    "--samples",
                    str(samples),
                    "--n-true",
                    "16",
                    "--device",
                    "cpu",
                ]
            )
            == 0
        )

    summary_same = json.loads((out_same / "summary.json").read_text())
    summary_mismatched = json.loads((out_mismatched / "summary.json").read_text())
    # train_vs_true is computed purely from the reconstructed training set and
    # a true_b draw off the --config's own metric_seed/evaluation_data_seed,
    # which differ between the two invocations here -- so this alone isn't
    # proof. What we actually assert: model_vs_train, which is computed from
    # the (config-independent) model samples against the reconstructed train
    # set, must be identical across both --config invocations, since the
    # reconstructed train set must come from the checkpoint, not --config.
    assert (
        summary_same["model_vs_train"]["mixture_biased_mmd2"]
        == summary_mismatched["model_vs_train"]["mixture_biased_mmd2"]
    )


def test_evaluate_manifest_records_checkpoint_derived_train_provenance(tmp_path):
    """The evaluate manifest's `seeds` field records this invocation's own
    --config seed hierarchy, which is distinct from the checkpoint-derived
    train_size/train_data_seed actually used to reconstruct the training set
    (see test_evaluate_reconstructs_train_set_from_checkpoint_not_cli_config).
    Both must be readable directly from the manifest without cross-referencing
    the checkpoint file by hand, and must genuinely differ when a mismatched
    --config is passed."""
    import tomllib

    ckpt, samples = _train_and_sample(tmp_path)
    mismatched = tmp_path / "mismatched.toml"
    mismatched.write_text(MISMATCHED_CONFIG)

    out = tmp_path / "eval_out"
    assert (
        evaluate_main(
            [
                "--config",
                str(mismatched),
                "--output",
                str(out),
                "--checkpoint",
                str(ckpt),
                "--teacher",
                str(tmp_path / "run" / "teacher.pt"),
                "--samples",
                str(samples),
                "--n-true",
                "16",
                "--device",
                "cpu",
            ]
        )
        == 0
    )

    manifest = json.loads((out / "manifest.json").read_text())
    resolved = tomllib.loads(Path(CONFIG).read_text())
    expected_train_size = round(
        resolved["dimensions"]["sample_ratio"] * resolved["dimensions"]["latent_dim"]
    )
    assert manifest["checkpoint_train_size"] == expected_train_size
    # MISMATCHED_CONFIG's own train_size (sample_ratio 3.0 * latent_dim 8 = 24)
    # must not leak into the recorded checkpoint provenance
    assert manifest["checkpoint_train_size"] != 24
    # MISMATCHED_CONFIG uses base_seed 999, genuinely different from smoke.toml's
    # 12345 -- so the checkpoint-derived seed and this invocation's own --config
    # seed must differ, proving the manifest doesn't just echo --config seeds.
    assert manifest["checkpoint_train_data_seed"] != manifest["seeds"]["train_data_seed"]


def test_evaluate_rejects_checkpoint_tampered_outside_semantic_hash(tmp_path):
    """checkpoint_id (checkpoints.py's semantic hash) covers model_state,
    model_config, teacher_id, step, and examples_seen -- not the whole
    checkpoint file (e.g. optimizer_state). A checkpoint mutated in a field
    the semantic hash doesn't cover would pass the checkpoint_id check, but
    must still be caught by the file-level checkpoint_file_sha256 comparison
    recorded in the sample manifest."""
    import torch

    import pytest

    ckpt, samples = _train_and_sample(tmp_path)

    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    for state in payload["optimizer_state"]["state"].values():
        for key, val in state.items():
            if isinstance(val, torch.Tensor):
                state[key] = val + 1.0
    torch.save(payload, ckpt)  # checkpoint_id (unaffected) left as-is

    with pytest.raises(ValueError, match="sha256"):
        evaluate_main(
            [
                "--config",
                CONFIG,
                "--output",
                str(tmp_path / "eval_out"),
                "--checkpoint",
                str(ckpt),
                "--teacher",
                str(tmp_path / "run" / "teacher.pt"),
                "--samples",
                str(samples),
                "--n-true",
                "16",
                "--device",
                "cpu",
            ]
        )


def test_evaluate_rejects_samples_artifact_with_tampered_tensor(tmp_path):
    ckpt, samples = _train_and_sample(tmp_path)

    import torch

    tensor_path = samples / "samples" / "samples.pt"
    tensor = torch.load(tensor_path, map_location="cpu", weights_only=False)
    torch.save(tensor + 1.0, tensor_path)  # tamper without updating manifest hash

    import pytest

    with pytest.raises(ValueError, match="artifact validation"):
        evaluate_main(
            [
                "--config",
                CONFIG,
                "--output",
                str(tmp_path / "eval_out"),
                "--checkpoint",
                str(ckpt),
                "--teacher",
                str(tmp_path / "run" / "teacher.pt"),
                "--samples",
                str(samples),
                "--n-true",
                "16",
                "--device",
                "cpu",
            ]
        )
