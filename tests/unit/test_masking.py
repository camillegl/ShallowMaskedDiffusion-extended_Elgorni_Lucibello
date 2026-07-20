import pytest
import torch

from maskeddiffusion.masking import (
    bernoulli_mask,
    continuous_time_mask,
    exact_visible_count_mask,
    mask_from_uniforms,
)


def spins(b: int, n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.where(torch.rand((b, n), generator=g) < 0.5, 1.0, -1.0)


def test_bernoulli_masking_rate():
    x = spins(2000, 32)
    t = torch.full((2000,), 0.3)
    batch = bernoulli_mask(x, t, torch.Generator().manual_seed(1))
    rate = batch.is_masked.float().mean().item()
    assert rate == pytest.approx(0.3, abs=0.01)


def test_exact_visible_count():
    x = spins(50, 16)
    batch = exact_visible_count_mask(x, visible_count=5, generator=torch.Generator().manual_seed(2))
    assert (batch.visible_count() == 5).all()
    assert (batch.masked_count() == 11).all()


def test_exact_count_bounds():
    x = spins(3, 8)
    g = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError):
        exact_visible_count_mask(x, visible_count=9, generator=g)
    with pytest.raises(ValueError):
        exact_visible_count_mask(x, visible_count=-1, generator=g)


def test_explicit_uniform_replay():
    x = spins(4, 8)
    t = torch.tensor([0.2, 0.5, 0.8, 1.0])
    u = torch.rand((4, 8), generator=torch.Generator().manual_seed(3))
    a = mask_from_uniforms(x, t, u)
    b = mask_from_uniforms(x, t, u)
    assert torch.equal(a.is_masked, b.is_masked)
    assert torch.equal(a.is_masked, u < t.unsqueeze(1))


def test_clean_values_never_mutated():
    x = spins(4, 8)
    x_copy = x.clone()
    batch = continuous_time_mask(x, torch.Generator().manual_seed(4))
    assert torch.equal(batch.values, x_copy)  # values stay clean spins
    assert batch.values.abs().min().item() == 1.0  # no in-band 0 sentinel
    assert batch.is_masked.dtype == torch.bool


def test_continuous_time_deterministic_under_generator():
    x = spins(6, 10)
    a = continuous_time_mask(x, torch.Generator().manual_seed(5))
    b = continuous_time_mask(x, torch.Generator().manual_seed(5))
    assert torch.equal(a.is_masked, b.is_masked)
    assert torch.equal(a.mask_probability, b.mask_probability)
