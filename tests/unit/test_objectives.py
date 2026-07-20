import itertools

import pytest
import torch
import torch.nn.functional as F

from maskeddiffusion.masking import MaskedBatch
from maskeddiffusion.models import LinearMaskedScore, LinearScoreConfig
from maskeddiffusion.objectives import (
    continuous_time_masked_bce,
    continuous_time_masked_bce_from_batch,
    exact_visible_count_bce,
    fixed_mask_probability_bce,
    l2_regularization,
)

N, B = 4, 3


def make_model(**kwargs) -> LinearMaskedScore:
    cfg = LinearScoreConfig(visible_dim=N, **kwargs)
    return LinearMaskedScore(cfg, torch.Generator().manual_seed(0))


def spins(b=B, n=N, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.where(torch.rand((b, n), generator=g) < 0.5, 1.0, -1.0)


def manual_loss(model, x, mask, t):
    """Direct reimplementation of the contract loss for verification."""
    logits = model(x, mask)
    y = (x + 1) / 2
    losses = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    w = (1.0 / t).unsqueeze(1).expand_as(x)
    return (losses * w * mask.float()).sum() / (x.shape[1] * x.shape[0])


def test_one_over_t_weighting_exact():
    model = make_model()
    x = spins()
    t = torch.tensor([0.25, 0.5, 0.9])
    u = torch.rand((B, N), generator=torch.Generator().manual_seed(2))
    mask = u < t.unsqueeze(1)
    batch = MaskedBatch(values=x, is_masked=mask, mask_probability=t, clean_targets=x)
    result = continuous_time_masked_bce_from_batch(model, batch)
    torch.testing.assert_close(result.data_loss, manual_loss(model, x, mask, t))


def test_exact_vs_monte_carlo_enumeration():
    """For tiny N and fixed t, the Bernoulli-mask expectation can be enumerated
    exactly; Monte Carlo over many draws must converge to it."""
    model = make_model()
    x = spins(b=1)
    t_val = 0.5
    t = torch.tensor([t_val])
    exact = torch.zeros(())
    for bits in itertools.product([False, True], repeat=N):
        mask = torch.tensor([bits])
        prob = (t_val ** sum(bits)) * ((1 - t_val) ** (N - sum(bits)))
        exact = exact + prob * manual_loss(model, x, mask, t)
    g = torch.Generator().manual_seed(3)
    draws = 40_000
    mc = torch.zeros(())
    for _ in range(draws):
        u = torch.rand((1, N), generator=g)
        mask = u < t_val
        mc = mc + manual_loss(model, x, mask, t)
    mc = mc / draws
    assert mc.item() == pytest.approx(exact.item(), rel=0.05)


def test_fixed_mask_probability_estimator():
    model = make_model()
    x = spins(b=500)
    res = fixed_mask_probability_bce(
        model, x, mask_probability=0.5, generator=torch.Generator().manual_seed(4)
    )
    assert res.diagnostics["mask_probability"] == 0.5
    assert 0.4 < res.diagnostics["masked_fraction"] < 0.6
    with pytest.raises(ValueError):
        fixed_mask_probability_bce(model, x, 0.0, torch.Generator().manual_seed(0))


def test_exact_visible_count_estimator():
    model = make_model()
    x = spins(b=10)
    res = exact_visible_count_bce(
        model, x, visible_count=1, generator=torch.Generator().manual_seed(5)
    )
    assert res.masked_token_count == 10 * (N - 1)


def test_regularization_separately():
    model = make_model(v_policy="trainable")
    reg = l2_regularization(model, l2reg=0.2, train_size=10)
    expected = 0.5 * 0.2 / 10 * sum((p**2).sum() for p in [model.W, model.V])
    torch.testing.assert_close(reg, expected)
    assert l2_regularization(model, 0.0, 10).item() == 0.0


def test_regularization_excludes_frozen():
    frozen = make_model(v_policy="frozen_zero")
    trainable = make_model(v_policy="trainable")
    with torch.no_grad():
        trainable.W.copy_(frozen.W)
        trainable.V.copy_(torch.ones(N, N))  # nonzero V
    reg_frozen = l2_regularization(frozen, 1.0, 1)
    reg_trainable = l2_regularization(trainable, 1.0, 1)
    assert reg_trainable.item() > reg_frozen.item()
    # frozen V contributes nothing even if it were nonzero
    expected_frozen = 0.5 * (frozen.W**2).sum()
    torch.testing.assert_close(reg_frozen, expected_frozen)


def test_min_time_changes_time_mixture_only_when_set():
    model = make_model()
    x = spins(b=200)
    res0 = continuous_time_masked_bce(model, x, torch.Generator().manual_seed(6), min_time=0.0)
    res_shift = continuous_time_masked_bce(model, x, torch.Generator().manual_seed(6), min_time=0.5)
    assert res_shift.diagnostics["mean_time"] > res0.diagnostics["mean_time"]
    assert res_shift.diagnostics["mean_time"] >= 0.5


def test_result_shapes_and_totals():
    model = make_model()
    x = spins()
    res = continuous_time_masked_bce(
        model, x, torch.Generator().manual_seed(7), l2reg=0.1, train_size=5
    )
    assert res.per_example.shape == (B,)
    torch.testing.assert_close(res.total, res.data_loss + res.regularization)
