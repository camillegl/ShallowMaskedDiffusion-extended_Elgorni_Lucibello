"""Regression: the new package reproduces the pinned legacy behavior.

Fixtures in tests/fixtures/original_architecture_v1/ were generated from the
legacy modules (see generate_fixtures.py and manifest.json there). The new
model runs in its named legacy-compat mode (normalization="legacy_init_only",
free diagonal) — discrepancy D3/D4 documented, not silently reproduced.
"""

from pathlib import Path

import pytest
import torch

from maskeddiffusion.masking import MaskedBatch, mask_from_uniforms
from maskeddiffusion.models import LinearMaskedScore, LinearScoreConfig
from maskeddiffusion.objectives import (
    continuous_time_masked_bce_from_batch,
    l2_regularization,
)

FIXTURES = Path(__file__).parents[1] / "fixtures" / "original_architecture_v1"


def load(name: str) -> dict:
    return torch.load(FIXTURES / name, map_location="cpu", weights_only=False)


def compat_model(W: torch.Tensor, V: torch.Tensor, b: torch.Tensor | None) -> LinearMaskedScore:
    n = W.shape[0]
    cfg = LinearScoreConfig(
        visible_dim=n,
        normalization="legacy_init_only",  # no runtime 1/sqrt(N) (D3)
        v_policy="trainable",
        bias_policy="trainable" if b is not None else "none",
        diagonal_policy="free",  # legacy trains the full matrix
    )
    model = LinearMaskedScore(cfg, torch.Generator().manual_seed(0))
    with torch.no_grad():
        model.W.copy_(W)
        model.V.copy_(V)
        if b is not None:
            assert model.b is not None
            model.b.copy_(b)
    return model


def test_linear_model_fixture():
    fx = load("linear_model.pt")
    model = compat_model(fx["W"], fx["V"], fx["b"])
    with torch.no_grad():
        logits = model(fx["x0"], fx["mask"])
    torch.testing.assert_close(logits, fx["logits"], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(torch.sigmoid(logits), fx["probabilities"], atol=1e-6, rtol=1e-6)


def test_masking_fixture():
    fx = load("masking.pt")
    batch = mask_from_uniforms(fx["x0"], fx["t"], fx["uniforms"])
    assert torch.equal(batch.is_masked, fx["mask"])
    # legacy in-band encoding: masked entries zeroed
    xt = fx["x0"].clone()
    xt[batch.is_masked] = 0.0
    assert torch.equal(xt, fx["xt"])


def test_objective_fixture():
    fx = load("objective.pt")
    model = compat_model(fx["W"], fx["V"], None)
    batch = MaskedBatch(
        values=fx["x0"],
        is_masked=fx["mask"],
        mask_probability=fx["t"],
        clean_targets=fx["x0"],
    )
    result = continuous_time_masked_bce_from_batch(model, batch)
    torch.testing.assert_close(result.data_loss, fx["data_loss"], atol=1e-6, rtol=1e-6)
    # legacy l2: 0.5*l2reg/(L*alpha)*||all params||^2; here all params trainable,
    # and train_size == round(L*alpha_legacy)
    L = fx["W"].shape[0]
    train_size = 4  # L=8, alpha_legacy=0.5 (fixture manifest)
    reg = l2_regularization(model, l2reg=0.1, train_size=train_size)
    torch.testing.assert_close(reg, fx["l2loss"], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(result.data_loss + reg, fx["total_loss"], atol=1e-6, rtol=1e-6)
    assert L == 8


def test_stochastic_sampler_fixture_trajectory():
    """The pinned trajectory is structurally valid and the new compat model
    reproduces the legacy score at every intermediate state."""
    fx = load("sampler_stochastic.pt")
    model = compat_model(fx["W"], fx["V"], None)
    states = fx["states"]
    for step in range(states.shape[0] - 1):
        before, after = states[step], states[step + 1]
        changed = before != after
        # exactly one token committed per step across the batch rows that had masks
        for row in range(before.shape[0]):
            row_changed = changed[row].nonzero().flatten()
            if (before[row] == 0).any():
                assert row_changed.numel() <= 1
            for j in row_changed.tolist():
                assert before[row, j] == 0.0, "revised a committed token"
                assert after[row, j] in (-1.0, 1.0)
        # new model agrees with the legacy score on this state
        is_masked = before == 0.0
        with torch.no_grad():
            new_logits = model(before, is_masked)
        legacy_logits = before @ fx["W"].t() + is_masked.to(before.dtype) @ fx["V"].t()
        torch.testing.assert_close(new_logits, legacy_logits, atol=1e-6, rtol=1e-6)
    final = fx["final"]
    assert ((final == 1.0) | (final == -1.0)).all()


def test_greedy_reconstruction_fixture_exact_replay():
    """Greedy decoding is deterministic given the order: exact replay with the
    new compat model must reproduce logits, intermediates, and final state."""
    fx = load("greedy_reconstruction.pt")
    model = compat_model(fx["W"], fx["V"], None)
    B = fx["x0"].shape[0]
    T0 = int(fx["T0"].item())
    xt = fx["corrupted"].clone()
    order = fx["decoding_order"]
    step = 0
    with torch.no_grad():
        for T in range(T0, 0, -1):
            is_masked = xt == 0.0
            logits = model(xt, is_masked)
            to_unmask = order[:, T - 1]
            lg = logits[torch.arange(B), to_unmask]
            torch.testing.assert_close(lg, fx["step_logits"][step], atol=1e-6, rtol=1e-6)
            xt[torch.arange(B), to_unmask] = (torch.sigmoid(lg) >= 0.5).float() * 2 - 1
            step += 1
            torch.testing.assert_close(xt, fx["intermediates"][step])
    assert torch.equal(xt, fx["final"])
    overlap = (xt * fx["x0"]).mean()
    torch.testing.assert_close(overlap, fx["final_overlap"], atol=1e-6, rtol=1e-6)


def test_fixture_manifest_complete():
    import json

    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    names = {e["fixture"] for e in manifest["fixtures"]}
    assert names == {
        "linear_model",
        "masking",
        "objective",
        "sampler_stochastic",
        "greedy_reconstruction",
    }
    for entry in manifest["fixtures"]:
        for key in (
            "source_commit",
            "source_module",
            "source_symbol",
            "environment",
            "seed",
            "dimensions",
            "tolerance",
            "discrepancies",
            "preserves",
        ):
            assert key in entry, f"{entry['fixture']} missing {key}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
