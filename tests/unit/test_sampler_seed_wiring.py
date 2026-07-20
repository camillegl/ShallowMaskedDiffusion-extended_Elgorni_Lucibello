"""Sampler seed wiring: sampler_order_seed and sampler_token_seed are
deterministic, separately derived RNG streams (not claimed statistically
independent) that drive distinct roles in `samplers.sample`.

Traces config -> SeedHierarchy -> torch.Generator -> samplers.sample; the CLI
and artifact-manifest hop is covered by
tests/integration/test_sample_cli_reproducibility.py.
"""

from __future__ import annotations

import torch

from maskeddiffusion.config import load_config
from maskeddiffusion.models import LinearMaskedScore, LinearScoreConfig
from maskeddiffusion.randomness import SeedHierarchy
from maskeddiffusion.samplers import SamplerConfig, sample

N = 6
CONFIG = "configs/smoke/smoke.toml"


def make_model(seed: int = 0) -> LinearMaskedScore:
    cfg = LinearScoreConfig(visible_dim=N, v_policy="trainable")
    model = LinearMaskedScore(cfg, torch.Generator().manual_seed(seed))
    with torch.no_grad():
        model.V.copy_(torch.randn(N, N, generator=torch.Generator().manual_seed(seed + 1)))
    return model


def test_seed_hierarchy_derives_distinct_order_and_token_seeds():
    seeds = SeedHierarchy.from_base(0)
    assert seeds.sampler_order_seed != seeds.sampler_token_seed
    order_gen = seeds.generator("sampler_order_seed")
    token_gen = seeds.generator("sampler_token_seed")
    # distinct Generator objects seeded from distinct integers
    assert order_gen is not token_gen
    assert not torch.equal(order_gen.get_state(), token_gen.get_state())


def test_resolved_config_and_manifest_record_both_seed_values(tmp_path):
    config = load_config(CONFIG)
    seeds_dict = config.seeds.to_dict()
    assert "sampler_order_seed" in seeds_dict
    assert "sampler_token_seed" in seeds_dict
    assert seeds_dict["sampler_order_seed"] != seeds_dict["sampler_token_seed"]

    out = tmp_path / "resolved_config.json"
    config.to_json(out)
    import json

    written = json.loads(out.read_text())
    assert written["seeds"]["sampler_order_seed"] == seeds_dict["sampler_order_seed"]
    assert written["seeds"]["sampler_token_seed"] == seeds_dict["sampler_token_seed"]


def test_changing_token_seed_leaves_coordinate_order_unchanged():
    """With order_generator fixed, the sequence of resolved coordinates
    (trajectory_masks) must be bit-identical regardless of the token seed:
    coordinate selection only ever consumes order_generator draws."""
    model = make_model()
    cfg = SamplerConfig("sequential_random_stochastic")
    order_seed = 7

    a = sample(
        model,
        cfg,
        batch_size=4,
        order_generator=torch.Generator().manual_seed(order_seed),
        token_generator=torch.Generator().manual_seed(1),
        record_trajectory=True,
    )
    b = sample(
        model,
        cfg,
        batch_size=4,
        order_generator=torch.Generator().manual_seed(order_seed),
        token_generator=torch.Generator().manual_seed(2),
        record_trajectory=True,
    )

    assert len(a.trajectory_masks) == len(b.trajectory_masks)
    for mask_a, mask_b in zip(a.trajectory_masks, b.trajectory_masks, strict=True):
        assert torch.equal(mask_a, mask_b)
    # but the decoded values differ because the token stream differs
    assert not torch.equal(a.values, b.values)


def test_changing_order_seed_leaves_token_generator_stream_unchanged():
    """With token_generator seed fixed, the *stream itself* is unaffected by
    order_generator's seed: same number of steps, same tensor shape drawn
    each step for the sequential sampler regardless of which coordinates get
    selected, so the token generator's final internal state must match bit
    for bit across two different order seeds."""
    model = make_model()
    cfg = SamplerConfig("sequential_random_stochastic")
    token_seed = 55

    token_gen_a = torch.Generator().manual_seed(token_seed)
    sample(
        model,
        cfg,
        batch_size=4,
        order_generator=torch.Generator().manual_seed(11),
        token_generator=token_gen_a,
    )

    token_gen_b = torch.Generator().manual_seed(token_seed)
    sample(
        model,
        cfg,
        batch_size=4,
        order_generator=torch.Generator().manual_seed(999),
        token_generator=token_gen_b,
    )

    assert torch.equal(token_gen_a.get_state(), token_gen_b.get_state())


def test_changing_order_seed_changes_resolution_order():
    model = make_model()
    cfg = SamplerConfig("sequential_random_stochastic")
    token_seed = 3

    a = sample(
        model,
        cfg,
        batch_size=4,
        order_generator=torch.Generator().manual_seed(1),
        token_generator=torch.Generator().manual_seed(token_seed),
        record_trajectory=True,
    )
    b = sample(
        model,
        cfg,
        batch_size=4,
        order_generator=torch.Generator().manual_seed(2),
        token_generator=torch.Generator().manual_seed(token_seed),
        record_trajectory=True,
    )
    assert any(
        not torch.equal(mask_a, mask_b)
        for mask_a, mask_b in zip(a.trajectory_masks, b.trajectory_masks, strict=True)
    )


def test_full_trajectory_deterministic_replay_both_seeds_fixed():
    model = make_model()
    cfg = SamplerConfig("sequential_random_stochastic")
    a = sample(
        model,
        cfg,
        batch_size=4,
        order_generator=torch.Generator().manual_seed(42),
        token_generator=torch.Generator().manual_seed(1042),
        record_trajectory=True,
    )
    b = sample(
        model,
        cfg,
        batch_size=4,
        order_generator=torch.Generator().manual_seed(42),
        token_generator=torch.Generator().manual_seed(1042),
        record_trajectory=True,
    )
    assert torch.equal(a.values, b.values)
    for mask_a, mask_b in zip(a.trajectory_masks, b.trajectory_masks, strict=True):
        assert torch.equal(mask_a, mask_b)
