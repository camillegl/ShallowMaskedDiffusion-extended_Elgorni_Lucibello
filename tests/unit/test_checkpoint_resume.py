"""Exact CPU resume: an uninterrupted 2-segment run must equal
checkpoint-at-segment-1 + resumed segment-2, in model and optimizer state."""

import dataclasses

import torch

from maskeddiffusion.config import RunConfig, TrainingConfig
from maskeddiffusion.dimensions import Dimensions
from maskeddiffusion.models import LinearScoreConfig
from maskeddiffusion.randomness import SeedHierarchy
from maskeddiffusion.training import build_state, train


def make_config(max_steps: int) -> RunConfig:
    dims = Dimensions.resolve(latent_dim=4, aspect_ratio=2.0, sample_ratio=3.0)
    return RunConfig(
        dimensions=dims,
        seeds=SeedHierarchy.from_base(77),
        model=LinearScoreConfig(visible_dim=dims.visible_dim),
        training=TrainingConfig(
            max_steps=max_steps, batch_size=4, learning_rate=1e-2, l2reg=0.01, log_every=1
        ),
    )


def test_exact_cpu_resume(tmp_path):
    # uninterrupted run: 6 steps
    full_cfg = make_config(6)
    state_full, _, _ = train(full_cfg, device="cpu")

    # interrupted run: 3 steps + checkpoint
    part_cfg = make_config(3)
    ckpt_dir = tmp_path / "ckpts"
    state_a, teacher_a, _ = train(part_cfg, device="cpu", checkpoint_dir=ckpt_dir)
    assert (ckpt_dir / "final.pt").exists()

    # resume to 6 steps from the checkpoint, fresh process state
    resume_cfg = dataclasses.replace(
        part_cfg, training=dataclasses.replace(part_cfg.training, max_steps=6)
    )
    fresh_state, fresh_teacher, fresh_data = build_state(resume_cfg, "cpu")
    state_b, _, _ = train(
        resume_cfg,
        device="cpu",
        state=fresh_state,
        teacher=fresh_teacher,
        train_data=fresh_data,
        resume_from=ckpt_dir / "final.pt",
    )

    assert state_b.step == state_full.step == 6
    assert state_b.examples_seen == state_full.examples_seen
    for k, v in state_full.model.state_dict().items():
        torch.testing.assert_close(state_b.model.state_dict()[k], v, atol=0, rtol=0)
    # optimizer state identical
    opt_full = state_full.optimizer.state_dict()
    opt_b = state_b.optimizer.state_dict()
    assert opt_full["param_groups"] == opt_b["param_groups"]
    for pid, st in opt_full["state"].items():
        for key, val in st.items():
            if isinstance(val, torch.Tensor):
                torch.testing.assert_close(opt_b["state"][pid][key], val, atol=0, rtol=0)
            else:
                assert opt_b["state"][pid][key] == val


def test_resume_rejects_wrong_teacher(tmp_path):
    cfg = make_config(2)
    ckpt_dir = tmp_path / "c"
    train(cfg, device="cpu", checkpoint_dir=ckpt_dir)
    other = dataclasses.replace(cfg, seeds=SeedHierarchy.from_base(1234))
    import pytest

    with pytest.raises(ValueError, match="teacher_id"):
        train(other, device="cpu", resume_from=ckpt_dir / "final.pt")


def test_load_checkpoint_rejects_tampered_weights(tmp_path):
    """A checkpoint whose weights were modified after saving, with the stored
    checkpoint_id left untouched, must be rejected on load rather than
    silently trusted (checkpoint_id is a content hash, recomputed on load)."""
    import pytest

    from maskeddiffusion.checkpoints import load_checkpoint

    cfg = make_config(2)
    ckpt_dir = tmp_path / "c"
    train(cfg, device="cpu", checkpoint_dir=ckpt_dir)
    path = ckpt_dir / "final.pt"

    payload = torch.load(path, map_location="cpu", weights_only=False)
    tampered_key = next(iter(payload["model_state"]))
    payload["model_state"][tampered_key] = payload["model_state"][tampered_key] + 1.0
    torch.save(payload, path)  # checkpoint_id left as-is: simulates tampering

    with pytest.raises(ValueError, match="checkpoint_id"):
        load_checkpoint(path)


def test_load_checkpoint_accepts_untampered_roundtrip(tmp_path):
    from maskeddiffusion.checkpoints import load_checkpoint

    cfg = make_config(2)
    ckpt_dir = tmp_path / "c"
    train(cfg, device="cpu", checkpoint_dir=ckpt_dir)
    load_checkpoint(ckpt_dir / "final.pt")  # must not raise


def test_load_checkpoint_rejects_tampered_model_config(tmp_path):
    """checkpoint_id must cover model_config, not just model_state: flipping
    a config field (e.g. normalization) changes how the unchanged weight
    bytes are interpreted by the model, so it must invalidate the hash too."""
    import pytest

    from maskeddiffusion.checkpoints import load_checkpoint

    cfg = make_config(2)
    ckpt_dir = tmp_path / "c"
    train(cfg, device="cpu", checkpoint_dir=ckpt_dir)
    path = ckpt_dir / "final.pt"

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["model_config"]["normalization"] == "explicit_sqrt_n"
    payload["model_config"] = {**payload["model_config"], "normalization": "none"}
    torch.save(payload, path)  # weights and checkpoint_id untouched: only config changed

    with pytest.raises(ValueError, match="checkpoint_id"):
        load_checkpoint(path)


def test_load_checkpoint_rejects_tampering_with_any_checkpoint_byte(tmp_path):
    """Any field checkpoint_identity hashes over (model_state, model_config,
    teacher_id, step, examples_seen) must be tamper-evident, not just
    model_state weights."""
    import pytest

    from maskeddiffusion.checkpoints import load_checkpoint

    cfg = make_config(2)
    ckpt_dir = tmp_path / "c"
    train(cfg, device="cpu", checkpoint_dir=ckpt_dir)
    path = ckpt_dir / "final.pt"
    original = torch.load(path, map_location="cpu", weights_only=False)

    for mutate in (
        lambda p: p.__setitem__("teacher_id", "not-" + p["teacher_id"]),
        lambda p: p.__setitem__("step", p["step"] + 1),
        lambda p: p.__setitem__("examples_seen", p["examples_seen"] + 1),
    ):
        payload = {**original, "model_state": dict(original["model_state"])}
        mutate(payload)
        torch.save(payload, path)  # checkpoint_id left as the original value
        with pytest.raises(ValueError, match="checkpoint_id"):
            load_checkpoint(path)
