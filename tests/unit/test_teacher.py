import math

import pytest
import torch

from maskeddiffusion.dimensions import Dimensions
from maskeddiffusion.teacher import HiddenManifoldTeacher, sign_pm1

DIMS = Dimensions.resolve(latent_dim=16, aspect_ratio=2.0, sample_ratio=3.0)


def make_teacher(seed: int = 7) -> HiddenManifoldTeacher:
    return HiddenManifoldTeacher.sample(DIMS, torch.Generator().manual_seed(seed))


def test_f_shape_and_scaling():
    t = make_teacher()
    assert t.F.shape == (DIMS.visible_dim, DIMS.latent_dim)
    # F_ia ~ N(0, 1/D): column variance ~ 1/D over many entries
    big_dims = Dimensions.resolve(latent_dim=400, aspect_ratio=2.0, sample_ratio=1.0)
    big = HiddenManifoldTeacher.sample(big_dims, torch.Generator().manual_seed(0))
    assert big.F.var().item() == pytest.approx(1.0 / 400, rel=0.1)


def test_sign_zero_is_plus_one():
    x = sign_pm1(torch.tensor([-2.0, -0.0, 0.0, 3.0]))
    assert torch.equal(x, torch.tensor([-1.0, 1.0, 1.0, 1.0]))


def test_outputs_strictly_pm1():
    t = make_teacher()
    x = t.sample_batch(50, torch.Generator().manual_seed(1))
    assert ((x == 1.0) | (x == -1.0)).all()
    assert not (x == 0.0).any()


def test_fixed_f_reuse_and_independent_latents():
    t = make_teacher()
    f_before = t.F.clone()
    a = t.sample_batch(10, torch.Generator().manual_seed(1))
    b = t.sample_batch(10, torch.Generator().manual_seed(2))
    assert torch.equal(t.F, f_before)  # F unchanged within the repeat
    assert not torch.equal(a, b)  # independent latent draws differ
    # same generator seed -> identical samples (explicit randomness)
    a2 = t.sample_batch(10, torch.Generator().manual_seed(1))
    assert torch.equal(a, a2)


def test_serialization_and_stable_id():
    t = make_teacher()
    tid = t.teacher_id
    t2 = HiddenManifoldTeacher.from_state_dict(t.state_dict())
    assert torch.equal(t.F, t2.F)
    assert t2.teacher_id == tid
    # different F -> different id
    other = make_teacher(seed=8)
    assert other.teacher_id != tid


def test_save_load_roundtrip(tmp_path):
    t = make_teacher()
    t.save(tmp_path / "teacher.pt")
    t2 = HiddenManifoldTeacher.load(tmp_path / "teacher.pt")
    assert t2.teacher_id == t.teacher_id


def test_from_state_dict_rejects_tampered_f():
    t = make_teacher()
    state = t.state_dict()
    state["F"] = state["F"] + 0.1
    with pytest.raises(ValueError):
        HiddenManifoldTeacher.from_state_dict(state)


def test_correlation_matrix_matches_arcsine_empirically():
    dims = Dimensions.resolve(latent_dim=8, aspect_ratio=1.5, sample_ratio=1.0)
    t = HiddenManifoldTeacher.sample(dims, torch.Generator().manual_seed(3))
    analytic = t.correlation_matrix()
    assert torch.allclose(analytic.diagonal(), torch.ones(dims.visible_dim), atol=1e-6)
    samples = t.sample_batch(200_000, torch.Generator().manual_seed(4))
    empirical = samples.t() @ samples / samples.shape[0]
    off = ~torch.eye(dims.visible_dim, dtype=torch.bool)
    max_err = (empirical - analytic)[off].abs().max().item()
    assert max_err < 0.01, f"empirical vs arcsine max error {max_err}"


def test_arcsine_clipping_is_safe():
    # duplicate rows in F give cos=1 exactly; arcsin must not produce nan
    dims = Dimensions.resolve(latent_dim=4, aspect_ratio=1.0, sample_ratio=1.0)
    F = torch.randn(4, 4)
    F[1] = F[0]
    t = HiddenManifoldTeacher(dims, F)
    c = t.correlation_matrix()
    assert torch.isfinite(c).all()
    # float32 cosine of duplicated rows lands near-but-below 1; asin is finite
    assert c[0, 1].item() == pytest.approx(2 / math.pi * math.asin(1.0), abs=2e-3)
