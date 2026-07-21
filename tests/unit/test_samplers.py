import pytest
import torch

from maskeddiffusion.models import LinearMaskedScore, LinearScoreConfig
from maskeddiffusion.samplers import SamplerConfig, sample

N = 6


def make_model(seed=0) -> LinearMaskedScore:
    cfg = LinearScoreConfig(visible_dim=N, v_policy="trainable")
    model = LinearMaskedScore(cfg, torch.Generator().manual_seed(seed))
    with torch.no_grad():
        model.V.copy_(torch.randn(N, N, generator=torch.Generator().manual_seed(seed + 1)))
    return model


def gens(seed=0):
    return {
        "order_generator": torch.Generator().manual_seed(seed),
        "token_generator": torch.Generator().manual_seed(seed + 1000),
    }


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
    # tokens_per_step only affects parallel_* samplers; SamplerConfig now
    # rejects a non-1 value for every other sampler as misleading provenance.
    return 2 if "parallel" in name else 1


@pytest.mark.parametrize("name", ALL_SAMPLERS)
def test_termination_and_pm1(name):
    model = make_model()
    cfg = SamplerConfig(name, tokens_per_step=_tokens_per_step_for(name))
    res = sample(model, cfg, batch_size=3, **gens())
    assert ((res.values == 1.0) | (res.values == -1.0)).all()


@pytest.mark.parametrize("name", ALL_SAMPLERS)
def test_no_revision_and_only_masked_change(name):
    model = make_model()
    cfg = SamplerConfig(name, tokens_per_step=_tokens_per_step_for(name))
    res = sample(model, cfg, batch_size=3, record_trajectory=True, **gens())
    for step in range(len(res.trajectory) - 1):
        before, after = res.trajectory[step], res.trajectory[step + 1]
        m_before = res.trajectory_masks[step]
        changed = before != after
        assert (changed <= m_before).all(), "changed a committed/observed token"
    # masks are monotone decreasing
    for step in range(len(res.trajectory_masks) - 1):
        assert (res.trajectory_masks[step + 1] <= res.trajectory_masks[step]).all()


def test_tokens_per_step_respected():
    model = make_model()
    cfg = SamplerConfig("parallel_random_stochastic", tokens_per_step=2)
    res = sample(model, cfg, batch_size=2, record_trajectory=True, **gens())
    for step in range(len(res.trajectory_masks) - 1):
        resolved = res.trajectory_masks[step].sum(dim=1) - res.trajectory_masks[step + 1].sum(dim=1)
        assert (resolved <= 2).all()
        assert (resolved >= 1).all()


def test_sequential_resolves_one_per_step():
    model = make_model()
    cfg = SamplerConfig("sequential_random_stochastic")
    res = sample(model, cfg, batch_size=2, record_trajectory=True, **gens())
    assert len(res.trajectory) == N + 1  # one token per row per step


def test_one_shot_single_step():
    model = make_model()
    cfg = SamplerConfig("one_shot_greedy")
    res = sample(model, cfg, batch_size=2, record_trajectory=True, **gens())
    assert len(res.trajectory) == 2


def test_deterministic_replay():
    model = make_model()
    cfg = SamplerConfig("sequential_random_stochastic")
    a = sample(model, cfg, batch_size=4, **gens(seed=42))
    b = sample(model, cfg, batch_size=4, **gens(seed=42))
    assert torch.equal(a.values, b.values)


def test_stochastic_vs_greedy_distinction():
    model = make_model()
    greedy_cfg = SamplerConfig("one_shot_greedy")
    a = sample(model, greedy_cfg, batch_size=2, **gens(seed=1))
    b = sample(model, greedy_cfg, batch_size=2, **gens(seed=99))
    assert torch.equal(a.values, b.values)  # greedy from full mask is deterministic
    stoch_cfg = SamplerConfig("one_shot_stochastic")
    outs = [sample(model, stoch_cfg, batch_size=8, **gens(seed=s)).values for s in (1, 2)]
    assert not torch.equal(outs[0], outs[1])  # stochastic varies with token seed


def test_stochastic_requires_token_generator():
    model = make_model()
    cfg = SamplerConfig("sequential_random_stochastic")
    with pytest.raises(ValueError):
        sample(
            model,
            cfg,
            batch_size=1,
            order_generator=torch.Generator().manual_seed(0),
            token_generator=None,
        )


def test_reconstruction_keeps_observed_tokens():
    model = make_model()
    cfg = SamplerConfig("sequential_random_greedy")
    x = torch.where(torch.rand(2, N, generator=torch.Generator().manual_seed(5)) < 0.5, 1.0, -1.0)
    mask = torch.zeros(2, N, dtype=torch.bool)
    mask[:, :3] = True
    res = sample(model, cfg, batch_size=2, initial_values=x, initial_mask=mask, **gens())
    assert torch.equal(res.values[:, 3:], x[:, 3:])


def test_confidence_greedy_picks_max_abs_logit():
    model = make_model()
    cfg = SamplerConfig("sequential_confidence_greedy")
    res = sample(model, cfg, batch_size=1, record_trajectory=True, **gens())
    # first resolved coordinate is the argmax |logit| of the fully masked state
    with torch.no_grad():
        logits0 = model(res.trajectory[0], res.trajectory_masks[0])
    first_changed = (res.trajectory[0] != res.trajectory[1]).nonzero()[0, 1]
    assert first_changed == logits0.abs().argmax(dim=1)[0]


def test_identity_records_selection_rules():
    ident = SamplerConfig("sequential_confidence_greedy").identity()
    assert ident["coordinate_selection"] == "max_abs_logit"
    assert ident["token_selection"] == "threshold_at_half"
    assert ident["revision_policy"] == "never"
    ident2 = SamplerConfig("parallel_random_stochastic", tokens_per_step=4).identity()
    assert ident2["token_selection"] == "bernoulli_sigmoid"
    assert ident2["tokens_per_step"] == 4


def test_rejects_unrecognized_sampler_name():
    """An unrecognized sampler_name must raise at config construction, not
    silently fall into the `elif name in _PARALLEL_NAMES` dispatch branch
    in `sample()` (it previously used a bare `else: # parallel_*`)."""
    with pytest.raises(ValueError, match="sampler_name"):
        SamplerConfig("sequential_random_stochstic")  # typo


def test_rejects_bad_tokens_per_step():
    with pytest.raises(ValueError):
        SamplerConfig("parallel_random_stochastic", tokens_per_step=0)
    with pytest.raises(TypeError):
        SamplerConfig("parallel_random_stochastic", tokens_per_step=1.5)


def test_rejects_bad_temperature():
    with pytest.raises(ValueError):
        SamplerConfig("sequential_random_stochastic", temperature=0.0)
    with pytest.raises(ValueError):
        SamplerConfig("sequential_random_stochastic", temperature=-1.0)
    with pytest.raises(ValueError):
        SamplerConfig("sequential_random_stochastic", temperature=float("nan"))


def test_nonunit_temperature_is_accepted_not_silently_rejected():
    """Non-unit temperature is a deliberate new sampler identity (see module
    docstring), not a malformed config; it must not be rejected."""
    cfg = SamplerConfig("sequential_random_stochastic", temperature=0.5)
    assert cfg.temperature == 0.5


@pytest.mark.parametrize(
    "name",
    [
        "sequential_random_stochastic",
        "sequential_random_greedy",
        "sequential_confidence_greedy",
        "one_shot_stochastic",
        "one_shot_greedy",
    ],
)
def test_rejects_ineffective_tokens_per_step(name):
    """tokens_per_step only affects parallel_* samplers; a non-1 value for
    any other sampler was previously silently ignored and recorded verbatim
    in the run manifest, misrepresenting provenance."""
    with pytest.raises(ValueError, match="tokens_per_step"):
        SamplerConfig(name, tokens_per_step=2)


@pytest.mark.parametrize(
    "name", ["sequential_random_greedy", "sequential_confidence_greedy", "one_shot_greedy"]
)
def test_rejects_ineffective_temperature_for_greedy(name):
    """Greedy decoding (`logits >= 0`) never reads temperature; a non-unit
    value was previously silently ignored and recorded verbatim in the run
    manifest, misrepresenting provenance."""
    with pytest.raises(ValueError, match="temperature"):
        SamplerConfig(name, temperature=0.5)


def test_parallel_sampler_accepts_tokens_per_step():
    cfg = SamplerConfig("parallel_random_stochastic", tokens_per_step=3)
    assert cfg.tokens_per_step == 3


def test_stochastic_sampler_accepts_nonunit_temperature():
    cfg = SamplerConfig("sequential_random_stochastic", temperature=0.5)
    assert cfg.temperature == 0.5


@pytest.mark.skipif(
    not (torch.cuda.is_available() or torch.backends.mps.is_available()),
    reason="requires a non-cpu device to construct a genuinely mismatched generator",
)
def test_generator_device_mismatch_raises_clear_error():
    """torch.rand(..., generator=g, device=d) requires g.device == d; sample()
    must raise a clear, actionable error before doing any tensor op, instead
    of torch's own less-clear RuntimeError from deep inside the sampling loop."""
    model = make_model()  # cpu
    other_device = "cuda" if torch.cuda.is_available() else "mps"
    mismatched_generator = torch.Generator(device=other_device).manual_seed(0)
    cfg = SamplerConfig("sequential_random_stochastic")
    with pytest.raises(ValueError, match="device"):
        sample(
            model,
            cfg,
            batch_size=1,
            order_generator=mismatched_generator,
            token_generator=torch.Generator().manual_seed(0),
        )
