import torch

from maskeddiffusion.dimensions import Dimensions
from maskeddiffusion.randomness import STREAMS, SeedHierarchy
from maskeddiffusion.teacher import HiddenManifoldTeacher


def test_from_base_distinct_streams():
    h = SeedHierarchy.from_base(1000)
    seeds = [getattr(h, s) for s in STREAMS]
    assert len(set(seeds)) == len(seeds)


def test_serialization_roundtrip():
    h = SeedHierarchy.from_base(7)
    h2 = SeedHierarchy.from_dict(h.to_dict())
    assert h == h2


def test_changing_one_stream_does_not_alter_others():
    dims = Dimensions.resolve(latent_dim=8, aspect_ratio=2.0, sample_ratio=2.0)
    h1 = SeedHierarchy.from_base(1)
    h2 = SeedHierarchy(**{**h1.to_dict(), "metric_seed": 999_999})
    t1 = HiddenManifoldTeacher.sample(dims, h1.generator("teacher_seed"))
    t2 = HiddenManifoldTeacher.sample(dims, h2.generator("teacher_seed"))
    assert torch.equal(t1.F, t2.F)  # teacher stream untouched by metric change
    d1 = t1.sample_batch(5, h1.generator("train_data_seed"))
    d2 = t2.sample_batch(5, h2.generator("train_data_seed"))
    assert torch.equal(d1, d2)
    m1 = torch.rand(4, generator=h1.generator("metric_seed"))
    m2 = torch.rand(4, generator=h2.generator("metric_seed"))
    assert not torch.equal(m1, m2)  # the changed stream does change


def test_generator_state_serialization_preserves_sequence():
    h = SeedHierarchy.from_base(3)
    g = h.generator("mask_seed")
    _ = torch.rand(10, generator=g)
    state = g.get_state()
    a = torch.rand(5, generator=g)
    g2 = torch.Generator()
    g2.set_state(state)
    b = torch.rand(5, generator=g2)
    assert torch.equal(a, b)


def test_teacher_fixed_within_repeat():
    dims = Dimensions.resolve(latent_dim=8, aspect_ratio=2.0, sample_ratio=2.0)
    h = SeedHierarchy.from_base(5)
    teacher = HiddenManifoldTeacher.sample(dims, h.generator("teacher_seed"))
    f = teacher.F.clone()
    teacher.sample_batch(10, h.generator("train_data_seed"))
    teacher.sample_batch(10, h.generator("validation_data_seed"))
    teacher.sample_batch(10, h.generator("evaluation_data_seed"))
    assert torch.equal(teacher.F, f)
