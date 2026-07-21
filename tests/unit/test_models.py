import math

import pytest
import torch

from maskeddiffusion.models import LinearMaskedScore, LinearScoreConfig


def make(n=8, **kwargs) -> LinearMaskedScore:
    cfg = LinearScoreConfig(visible_dim=n, **kwargs)
    return LinearMaskedScore(cfg, torch.Generator().manual_seed(0))


def batch(n=8, b=3, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.where(torch.rand((b, n), generator=g) < 0.5, 1.0, -1.0)
    m = torch.rand((b, n), generator=g) < 0.4
    return x, m


def test_explicit_runtime_normalization():
    n = 16
    model = make(n, normalization="explicit_sqrt_n", diagonal_policy="free")
    legacy = make(n, normalization="legacy_init_only", diagonal_policy="free")
    with torch.no_grad():
        legacy.W.copy_(model.W)
        legacy.V.copy_(model.V)
    x, m = batch(n)
    with torch.no_grad():
        h_new = model(x, m)
        h_legacy = legacy(x, m)
    torch.testing.assert_close(h_new * math.sqrt(n), h_legacy)


def test_masked_values_do_not_leak():
    model = make()
    x, m = batch()
    x2 = x.clone()
    x2[m] = -x2[m]  # flip values under the mask; output must be identical
    with torch.no_grad():
        torch.testing.assert_close(model(x, m), model(x2, m))


def test_diagonal_zero_policy():
    model = make(diagonal_policy="zero")
    assert model.effective_W().diagonal().abs().max().item() == 0.0
    # even after a gradient step the effective diagonal stays zero
    x, m = batch()
    out = model(x, m).sum()
    out.backward()
    with torch.no_grad():
        model.W += 1.0
    assert model.effective_W().diagonal().abs().max().item() == 0.0


def test_v_frozen_and_trainable_modes():
    frozen = make(v_policy="frozen_zero")
    assert not frozen.V.requires_grad
    assert frozen.V.abs().max().item() == 0.0
    trainable = make(v_policy="trainable")
    assert trainable.V.requires_grad


def test_bias_modes():
    assert make(bias_policy="none").b is None
    m1 = make(bias_policy="trainable")
    assert m1.b is not None and m1.b.requires_grad
    m2 = make(bias_policy="frozen_zero")
    assert m2.b is not None and not m2.b.requires_grad


def test_regularized_parameters_exclude_frozen():
    model = make(v_policy="frozen_zero", bias_policy="frozen_zero")
    regs = model.regularized_parameters()
    assert regs == [model.W]
    model2 = make(v_policy="trainable", bias_policy="trainable")
    assert set(map(id, model2.regularized_parameters())) == {
        id(model2.W),
        id(model2.V),
        id(model2.b),
    }


def test_probabilities_are_sigmoid_of_logits():
    model = make()
    x, m = batch()
    with torch.no_grad():
        torch.testing.assert_close(model.probabilities(x, m), torch.sigmoid(model(x, m)))


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("normalization", "explicit_sqrt_N"),  # capitalization typo
        ("v_policy", "trainble"),  # typo
        ("bias_policy", "frozen"),  # near-miss of "frozen_zero"
        ("diagonal_policy", "diag"),
    ],
)
def test_rejects_unrecognized_config_strings(field, bad_value):
    """An unrecognized string must raise, not silently fall through to
    another valid scientific configuration via a broad `else` branch."""
    with pytest.raises(ValueError, match=field):
        LinearScoreConfig(visible_dim=8, **{field: bad_value})


def test_rejects_nonpositive_visible_dim():
    with pytest.raises(ValueError):
        LinearScoreConfig(visible_dim=0)
    with pytest.raises(ValueError):
        LinearScoreConfig(visible_dim=-4)


def test_rejects_non_int_visible_dim():
    with pytest.raises(TypeError):
        LinearScoreConfig(visible_dim=8.0)
