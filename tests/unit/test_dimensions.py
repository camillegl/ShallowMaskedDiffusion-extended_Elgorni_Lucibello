import math

import pytest

from maskeddiffusion.dimensions import Dimensions


def test_resolve_basic():
    d = Dimensions.resolve(latent_dim=100, aspect_ratio=5.0, sample_ratio=8.0)
    assert d.visible_dim == 500
    assert d.train_size == 800
    assert d.visible_load == pytest.approx(1.6)


def test_rounding_rule():
    d = Dimensions.resolve(latent_dim=10, aspect_ratio=2.55, sample_ratio=1.24)
    assert d.visible_dim == round(2.55 * 10) == 26
    assert d.train_size == round(1.24 * 10) == 12
    assert d.visible_load == pytest.approx(12 / 26)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"latent_dim": 0, "aspect_ratio": 1.0, "sample_ratio": 1.0},
        {"latent_dim": -5, "aspect_ratio": 1.0, "sample_ratio": 1.0},
        {"latent_dim": 10, "aspect_ratio": 0.0, "sample_ratio": 1.0},
        {"latent_dim": 10, "aspect_ratio": -1.0, "sample_ratio": 1.0},
        {"latent_dim": 10, "aspect_ratio": 1.0, "sample_ratio": 0.0},
        {"latent_dim": 10, "aspect_ratio": math.inf, "sample_ratio": 1.0},
        {"latent_dim": 10, "aspect_ratio": 1.0, "sample_ratio": math.nan},
        {"latent_dim": 10, "aspect_ratio": 0.001, "sample_ratio": 1.0},  # visible_dim 0
        {"latent_dim": 10, "aspect_ratio": 1.0, "sample_ratio": 0.001},  # train_size 0
    ],
)
def test_invalid_values(kwargs):
    with pytest.raises((ValueError, TypeError)):
        Dimensions.resolve(**kwargs)


def test_non_int_latent_dim_rejected():
    with pytest.raises(TypeError):
        Dimensions.resolve(latent_dim=10.0, aspect_ratio=1.0, sample_ratio=1.0)  # type: ignore[arg-type]


def test_serialization_roundtrip():
    d = Dimensions.resolve(latent_dim=32, aspect_ratio=3.0, sample_ratio=2.5)
    d2 = Dimensions.from_dict(d.to_dict())
    assert d == d2


def test_from_dict_rejects_inconsistent_stored_values():
    d = Dimensions.resolve(latent_dim=32, aspect_ratio=3.0, sample_ratio=2.5)
    corrupted = {**d.to_dict(), "visible_dim": d.visible_dim + 1}
    with pytest.raises(ValueError):
        Dimensions.from_dict(corrupted)


def test_visible_load_consistency_enforced():
    with pytest.raises(ValueError):
        Dimensions(
            latent_dim=10,
            aspect_ratio=1.0,
            sample_ratio=1.0,
            visible_dim=10,
            train_size=10,
            visible_load=2.0,
        )
