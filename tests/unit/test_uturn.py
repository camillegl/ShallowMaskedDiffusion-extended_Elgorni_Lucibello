"""U-turn / reconstruction experiment (src/maskeddiffusion/uturn.py).

Protocol-level tests: t=0 exactness, observed-coordinate freezing, terminal
binarity, seed determinism and pairing, train/fresh stream independence, the
zero-information 1-t baseline, and the claim-discipline rule that no result
is labelled memorization without the fresh-source comparison.
"""

import json
import math

import pytest
import torch

from maskeddiffusion.dimensions import Dimensions
from maskeddiffusion.models import LinearMaskedScore, LinearScoreConfig
from maskeddiffusion.randomness import SeedHierarchy
from maskeddiffusion.samplers import SamplerConfig, sample
from maskeddiffusion.teacher import HiddenManifoldTeacher
from maskeddiffusion.uturn import (
    UTurnConfig,
    run_uturn,
    summarize_uturn,
    uturn_cell_seed,
)

ALL_SAMPLERS = [
    "sequential_random_stochastic",
    "sequential_random_greedy",
    "sequential_confidence_greedy",
    "parallel_random_stochastic",
    "parallel_random_greedy",
    "one_shot_stochastic",
    "one_shot_greedy",
]


def _tokens_per_step_for(name: str) -> int:
    # tokens_per_step only affects parallel_* samplers (SamplerConfig rejects
    # a non-1 value for the others as misleading provenance).
    return 2 if "parallel" in name else 1


def make_setup(latent_dim=4, aspect_ratio=4.0, sample_ratio=8.0, base_seed=7, v_policy="trainable"):
    dims = Dimensions.resolve(
        latent_dim=latent_dim, aspect_ratio=aspect_ratio, sample_ratio=sample_ratio
    )
    seeds = SeedHierarchy.from_base(base_seed)
    teacher = HiddenManifoldTeacher.sample(dims, seeds.generator("teacher_seed"))
    train_set = teacher.sample_batch(dims.train_size, seeds.generator("train_data_seed"))
    model = LinearMaskedScore(
        LinearScoreConfig(visible_dim=dims.visible_dim, v_policy=v_policy),
        seeds.generator("model_seed"),
    )
    return dims, seeds, teacher, train_set, model


def make_zero_model(visible_dim: int) -> LinearMaskedScore:
    """Zero-information model: W = V = 0, no bias — every logit is identically
    0, so stochastic decoding is a fair coin independent of the clean data."""
    model = LinearMaskedScore(
        LinearScoreConfig(visible_dim=visible_dim, v_policy="frozen_zero"),
        torch.Generator().manual_seed(0),
    )
    with torch.no_grad():
        model.W.zero_()
    return model


@pytest.mark.parametrize("sampler_name", ALL_SAMPLERS)
def test_t_zero_gives_exact_reconstruction(sampler_name):
    """At t=0 nothing is masked: the sampler must return the clean example
    exactly, for every sampler and both sources (q_U = 1, Hamming = 0)."""
    _dims, seeds, teacher, train_set, model = make_setup()
    cfg = UTurnConfig(t_values=(0.0,), n_examples=3)
    sampler = SamplerConfig(sampler_name, tokens_per_step=_tokens_per_step_for(sampler_name))
    res = run_uturn(model, sampler, teacher, train_set, cfg, seeds)
    for cell in res.cells:
        assert cell.n_masked == 0
        assert cell.q_u == 1.0
        assert cell.excess_over_baseline == 0.0
        assert cell.hamming_error == 0.0
        assert cell.hamming_error_masked is None
    for source in res.sources:
        assert torch.equal(res.reconstructions[source][0], res.clean[source])


@pytest.mark.parametrize("sampler_name", ALL_SAMPLERS)
def test_observed_coordinates_never_change(sampler_name):
    """Protocol step 4: originally observed coordinates remain fixed. This is
    the sampler's absorbing-state guarantee exercised through the U-turn
    driver, for every sampler identity."""
    _dims, seeds, teacher, train_set, model = make_setup()
    cfg = UTurnConfig(t_values=(0.5,), n_examples=4)
    sampler = SamplerConfig(sampler_name, tokens_per_step=_tokens_per_step_for(sampler_name))
    res = run_uturn(model, sampler, teacher, train_set, cfg, seeds)
    mask = res.masks[0]  # (B, N), shared across sources (paired seeds)
    assert mask.any() and (~mask).any()  # the test is non-vacuous
    for source in res.sources:
        recon = res.reconstructions[source][0]
        clean = res.clean[source]
        assert torch.equal(recon[~mask], clean[~mask]), (
            f"{sampler_name}/{source}: an observed coordinate was revised"
        )


@pytest.mark.parametrize("sampler_name", ALL_SAMPLERS)
def test_terminal_states_binary_and_fully_unmasked(sampler_name):
    """Terminal reconstructions take only ±1. Since masked entries start at 0
    (the only non-terminal value) and only resolved entries become ±1, an
    all-±1 terminal state is exactly a fully unmasked one."""
    _dims, seeds, teacher, train_set, model = make_setup()
    cfg = UTurnConfig(t_values=(0.25, 0.75), n_examples=3)
    sampler = SamplerConfig(sampler_name, tokens_per_step=_tokens_per_step_for(sampler_name))
    res = run_uturn(model, sampler, teacher, train_set, cfg, seeds)
    for source in res.sources:
        recon = res.reconstructions[source]
        assert ((recon == 1.0) | (recon == -1.0)).all()


def test_identical_seeds_reproduce_identical_outputs():
    """Same seed hierarchy -> bitwise-identical masks, reconstructions, cell
    records, and summaries."""
    _dims, seeds, teacher, train_set, model = make_setup()
    cfg = UTurnConfig(t_values=(0.25, 0.5, 0.75), n_examples=4)
    sampler = SamplerConfig("sequential_random_stochastic")
    a = run_uturn(model, sampler, teacher, train_set, cfg, seeds)
    b = run_uturn(model, sampler, teacher, train_set, cfg, seeds)
    assert torch.equal(a.masks, b.masks)
    assert a.cells == b.cells
    for source in a.sources:
        assert torch.equal(a.clean[source], b.clean[source])
        assert torch.equal(a.reconstructions[source], b.reconstructions[source])
    assert summarize_uturn(a) == summarize_uturn(b)


def test_masks_are_paired_across_source_selections():
    """The mask at (example_index, t) must not depend on which sources are
    run: a train-only and a fresh-only run with the same seeds produce the
    identical mask tensor (paired-seed design)."""
    _dims, seeds, teacher, train_set, model = make_setup()
    sampler = SamplerConfig("sequential_random_stochastic")
    cfg_both = UTurnConfig(t_values=(0.3, 0.6), n_examples=4, sources=("train", "fresh"))
    cfg_train = UTurnConfig(t_values=(0.3, 0.6), n_examples=4, sources=("train",))
    cfg_fresh = UTurnConfig(t_values=(0.3, 0.6), n_examples=4, sources=("fresh",))
    both = run_uturn(model, sampler, teacher, train_set, cfg_both, seeds)
    train_only = run_uturn(model, sampler, teacher, train_set, cfg_train, seeds)
    fresh_only = run_uturn(model, sampler, teacher, train_set, cfg_fresh, seeds)
    assert torch.equal(both.masks, train_only.masks)
    assert torch.equal(both.masks, fresh_only.masks)
    # and the per-source results of the joint run match the single-source runs
    assert torch.equal(both.reconstructions["train"], train_only.reconstructions["train"])
    assert torch.equal(both.reconstructions["fresh"], fresh_only.reconstructions["fresh"])


def test_uturn_cell_seed_derivation_is_deterministic_and_key_sensitive():
    a = uturn_cell_seed("mask", 123, 4, 0.5)
    assert a == uturn_cell_seed("mask", 123, 4, 0.5)
    assert 0 <= a < 2**63
    assert a != uturn_cell_seed("mask", 123, 5, 0.5)  # example index matters
    assert a != uturn_cell_seed("mask", 123, 4, 0.6)  # t value matters
    assert a != uturn_cell_seed("order", 123, 4, 0.5)  # purpose matters
    assert a != uturn_cell_seed("mask", 124, 4, 0.5)  # stream seed matters


def test_train_and_fresh_streams_are_independent():
    """The fresh stream must not depend on the training set: swapping in a
    different training set (same seed hierarchy otherwise) leaves every
    fresh-source draw and reconstruction bitwise unchanged, while the
    train-source cells follow the new training set. Also, fresh draws are
    not bitwise copies of any training row."""
    _dims, seeds, teacher, train_set_a, model = make_setup()
    train_set_b = teacher.sample_batch(train_set_a.shape[0], torch.Generator().manual_seed(987654))
    cfg = UTurnConfig(t_values=(0.5,), n_examples=4)
    sampler = SamplerConfig("sequential_random_stochastic")
    run_a = run_uturn(model, sampler, teacher, train_set_a, cfg, seeds)
    run_b = run_uturn(model, sampler, teacher, train_set_b, cfg, seeds)

    assert torch.equal(run_a.clean["fresh"], run_b.clean["fresh"])
    assert torch.equal(run_a.reconstructions["fresh"], run_b.reconstructions["fresh"])
    q_a = [c.q_u for c in run_a.cells if c.source == "fresh"]
    q_b = [c.q_u for c in run_b.cells if c.source == "fresh"]
    assert q_a == q_b

    assert not torch.equal(run_a.clean["train"], run_b.clean["train"])
    # fresh draws are independent of the train stream, not copies of it
    for fresh_row in run_a.clean["fresh"]:
        for train_row in torch.cat([train_set_a, train_set_b], dim=0):
            assert not torch.equal(fresh_row, train_row)


@pytest.mark.parametrize("v_policy", ["frozen_zero", "trainable"])
def test_run_supports_frozen_zero_and_trainable_v_models(v_policy):
    """Both checkpoint families (frozen_zero and trainable V) run end to end
    through the library driver."""
    _dims, seeds, teacher, train_set, model = make_setup(v_policy=v_policy)
    cfg = UTurnConfig(t_values=(0.5,), n_examples=2)
    res = run_uturn(
        model, SamplerConfig("sequential_random_greedy"), teacher, train_set, cfg, seeds
    )
    assert len(res.cells) == 2 * 2 * 1
    assert model.config.v_policy == v_policy


@pytest.mark.parametrize("t_value", [0.25, 0.5, 0.75])
def test_zero_information_model_follows_no_recovery_baseline(t_value):
    """A zero-information model (all logits identically 0) cannot recover a
    masked coordinate better than chance, so q_U(t) must follow the
    no-recovery baseline 1 - t up to finite-sample noise.

    Explicit tolerance derivation. Per example, q_U - (1 - t) decomposes as
    (n_observed/N - (1 - t)) + (1/N) * sum_{i masked} s_i with s_i iid fair
    ±1 (stochastic decoding at zero logits is independent of the clean data):
        Var = t(1-t)/N + t/N = t(2-t)/N  <=  2t/N.
    Over B independent examples the standard error is <= sqrt(2t/(N B)).
    The test uses N = 64, B = 64 and a 4-sigma tolerance
    4*sqrt(2t/(N*B)) (deterministic under the fixed test seeds)."""
    dims, seeds, teacher, train_set, _model = make_setup(
        latent_dim=16, aspect_ratio=4.0, sample_ratio=8.0
    )
    assert dims.visible_dim == 64
    zero_model = make_zero_model(dims.visible_dim)
    n_examples = 64
    cfg = UTurnConfig(t_values=(t_value,), n_examples=n_examples, sources=("fresh",))
    res = run_uturn(
        zero_model,
        SamplerConfig("sequential_random_stochastic"),
        teacher,
        train_set,
        cfg,
        seeds,
    )
    point = summarize_uturn(res)["points"][0]
    tol = 4.0 * math.sqrt(2.0 * t_value / (dims.visible_dim * n_examples))
    assert abs(point["q_u_mean"] - point["no_recovery_baseline"]) <= tol
    assert abs(point["excess_recovery_mean"]) <= tol


def test_no_result_labelled_memorization_without_fresh_comparison():
    """Claim discipline: a train-source-only summary must contain no
    memorization-labelled field anywhere (and no train/fresh comparison
    block); the comparison appears only when both sources were run."""
    _dims, seeds, teacher, train_set, model = make_setup()
    sampler = SamplerConfig("sequential_random_stochastic")

    train_only = run_uturn(
        model,
        sampler,
        teacher,
        train_set,
        UTurnConfig(t_values=(0.5,), n_examples=3, sources=("train",)),
        seeds,
    )
    summary_train_only = summarize_uturn(train_only)
    assert "train_fresh_comparison" not in summary_train_only
    assert "memoriz" not in json.dumps(summary_train_only).lower()

    both = run_uturn(
        model,
        sampler,
        teacher,
        train_set,
        UTurnConfig(t_values=(0.25, 0.5), n_examples=3),
        seeds,
    )
    summary_both = summarize_uturn(both)
    comparison = summary_both["train_fresh_comparison"]
    assert len(comparison) == 2
    for entry in comparison:
        assert "excess_q_u_train_minus_fresh" in entry
        assert "excess_nearest_train_overlap_train_minus_fresh" in entry


def test_uturn_config_validation():
    with pytest.raises(ValueError, match="non-empty"):
        UTurnConfig(t_values=(), n_examples=1)
    with pytest.raises(ValueError, match="0 <= t <= 1"):
        UTurnConfig(t_values=(-0.1,), n_examples=1)
    with pytest.raises(ValueError, match="0 <= t <= 1"):
        UTurnConfig(t_values=(1.5,), n_examples=1)
    with pytest.raises(ValueError, match="0 <= t <= 1"):
        UTurnConfig(t_values=(float("nan"),), n_examples=1)
    with pytest.raises(TypeError):
        UTurnConfig(t_values=(True,), n_examples=1)
    with pytest.raises(ValueError, match="duplicate"):
        UTurnConfig(t_values=(0.5, 0.5), n_examples=1)
    with pytest.raises(ValueError, match=">= 1"):
        UTurnConfig(t_values=(0.5,), n_examples=0)
    with pytest.raises(TypeError):
        UTurnConfig(t_values=(0.5,), n_examples=2.5)
    with pytest.raises(ValueError, match="source"):
        UTurnConfig(t_values=(0.5,), n_examples=1, sources=("test",))
    with pytest.raises(ValueError, match="non-empty"):
        UTurnConfig(t_values=(0.5,), n_examples=1, sources=())
    with pytest.raises(ValueError, match="duplicate"):
        UTurnConfig(t_values=(0.5,), n_examples=1, sources=("train", "train"))


def test_run_rejects_more_examples_than_train_rows():
    """Train-source examples come from the actual training set; asking for
    more than train_size must fail loudly, not silently cycle."""
    dims, seeds, teacher, train_set, model = make_setup()
    cfg = UTurnConfig(t_values=(0.5,), n_examples=dims.train_size + 1, sources=("train",))
    with pytest.raises(ValueError, match="exceeds"):
        run_uturn(
            model,
            SamplerConfig("sequential_random_stochastic"),
            teacher,
            train_set,
            cfg,
            seeds,
        )


def test_run_rejects_teacher_model_dimension_mismatch():
    _dims, seeds, teacher, train_set, model = make_setup()
    other_dims = Dimensions.resolve(latent_dim=3, aspect_ratio=5.0, sample_ratio=4.0)
    other_teacher = HiddenManifoldTeacher.sample(other_dims, torch.Generator().manual_seed(1))
    cfg = UTurnConfig(t_values=(0.5,), n_examples=1)
    with pytest.raises(ValueError, match="visible_dim"):
        run_uturn(
            model,
            SamplerConfig("sequential_random_stochastic"),
            other_teacher,
            train_set,
            cfg,
            seeds,
        )


def test_sampler_interface_is_used_not_reimplemented(monkeypatch):
    """The U-turn driver must go through samplers.sample's initial_values /
    initial_mask interface: a monkeypatched sampler that records its inputs
    sees the partially observed state (masked entries blanked, mask passed
    through) and is called once per (source, example, t) cell."""
    _dims, seeds, teacher, train_set, model = make_setup()
    calls = []

    def recording_sample(model, config, batch_size, **kwargs):
        calls.append(kwargs)
        return sample(model, config, batch_size, **kwargs)

    import maskeddiffusion.uturn as uturn_module

    monkeypatch.setattr(uturn_module, "sample", recording_sample)
    cfg = UTurnConfig(t_values=(0.5,), n_examples=2, sources=("train", "fresh"))
    run_uturn(
        model,
        SamplerConfig("sequential_random_stochastic"),
        teacher,
        train_set,
        cfg,
        seeds,
    )

    assert len(calls) == 2 * 2 * 1
    for kwargs in calls:
        iv = kwargs["initial_values"]
        im = kwargs["initial_mask"]
        assert iv is not None and im is not None
        assert (iv[im] == 0.0).all()  # masked entries are never passed in-band
