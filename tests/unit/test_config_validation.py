"""Runtime validation of TrainingConfig scientific parameters
(docs/RESEARCH_SPEC.md; see also test_notation_enforcement.py for the
alpha/gamma naming contract, and test_models.py / test_samplers.py for
LinearScoreConfig / SamplerConfig validation)."""

import pytest

from maskeddiffusion.config import TrainingConfig


def test_default_config_is_valid():
    TrainingConfig()  # must not raise


@pytest.mark.parametrize("bad", [0, -1, 1.5])
def test_rejects_bad_max_steps(bad):
    with pytest.raises((ValueError, TypeError)):
        TrainingConfig(max_steps=bad)


@pytest.mark.parametrize("bad", [0, -1, 1.5])
def test_rejects_bad_batch_size(bad):
    with pytest.raises((ValueError, TypeError)):
        TrainingConfig(batch_size=bad)


@pytest.mark.parametrize("bad", [0.0, -1e-3, float("nan"), float("inf")])
def test_rejects_bad_learning_rate(bad):
    with pytest.raises(ValueError):
        TrainingConfig(learning_rate=bad)


@pytest.mark.parametrize("bad", [-1e-6, float("nan"), float("inf")])
def test_rejects_bad_l2reg(bad):
    with pytest.raises(ValueError):
        TrainingConfig(l2reg=bad)


@pytest.mark.parametrize("bad", [-0.01, 1.0, 1.5, float("nan"), float("inf")])
def test_rejects_min_time_outside_zero_one(bad):
    """min_time must satisfy 0 <= min_time < 1 (docs/RESEARCH_SPEC.md:
    t ~ U(min_time, 1); min_time=1 would make every sequence fully masked
    with probability 1, degenerate for the objective)."""
    with pytest.raises(ValueError):
        TrainingConfig(min_time=bad)


def test_min_time_zero_is_valid():
    TrainingConfig(min_time=0.0)  # must not raise: the documented default


@pytest.mark.parametrize("bad", [-1, 1.5])
def test_rejects_bad_validation_size(bad):
    with pytest.raises((ValueError, TypeError)):
        TrainingConfig(validation_size=bad)


@pytest.mark.parametrize("bad", [0, -1, 1.5])
def test_rejects_bad_validation_every(bad):
    with pytest.raises((ValueError, TypeError)):
        TrainingConfig(validation_every=bad)


@pytest.mark.parametrize("bad", [-1, 1.5])
def test_rejects_bad_checkpoint_every(bad):
    with pytest.raises((ValueError, TypeError)):
        TrainingConfig(checkpoint_every=bad)


@pytest.mark.parametrize("bad", [0, -1, 1.5])
def test_rejects_bad_log_every(bad):
    with pytest.raises((ValueError, TypeError)):
        TrainingConfig(log_every=bad)
