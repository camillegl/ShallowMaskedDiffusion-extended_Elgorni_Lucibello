"""ExperimentSpec: serialization round-trip, fingerprints, validation."""

from __future__ import annotations

import pytest

from maskeddiffusion.dimensions import Dimensions
from maskeddiffusion.experiments.spec import (
    EvaluationConfig,
    ExperimentSpec,
    spec_fingerprint,
    spec_identities,
)
from maskeddiffusion.models import LinearScoreConfig
from maskeddiffusion.randomness import SeedHierarchy
from maskeddiffusion.samplers import SamplerConfig
from maskeddiffusion.uturn import UTurnConfig


def make_spec(**overrides):
    dims = Dimensions.resolve(latent_dim=8, aspect_ratio=2.0, sample_ratio=3.0)
    defaults = {
        "experiment_id": "exp-a",
        "pair_id": "exp-a-r000",
        "repeat_id": 0,
        "intervention": "v_trainability",
        "condition": "frozen_zero_v",
        "dimensions": dims,
        "seeds": SeedHierarchy.from_base(7),
        "model": LinearScoreConfig(visible_dim=dims.visible_dim),
        "sampler": SamplerConfig("sequential_random_stochastic"),
        "evaluation": EvaluationConfig(n_true=8, lambdas=(4.0, 8.0)),
        "n_generate": 4,
    }
    defaults.update(overrides)
    return ExperimentSpec(**defaults)


def test_round_trip_preserves_spec_and_fingerprint():
    spec = make_spec()
    restored = ExperimentSpec.from_dict(spec.to_dict())
    assert restored == spec
    assert spec_fingerprint(restored) == spec_fingerprint(spec)


def test_fingerprint_changes_with_any_config_change():
    base = make_spec()
    changed = [
        make_spec(condition="trainable_v"),
        make_spec(repeat_id=1, seeds=SeedHierarchy.from_base(8)),
        make_spec(n_generate=5),
        make_spec(evaluation=EvaluationConfig(n_true=9, lambdas=(4.0, 8.0))),
    ]
    fingerprints = {spec_fingerprint(s) for s in [base, *changed]}
    assert len(fingerprints) == len(changed) + 1


def test_from_dict_rejects_unknown_and_missing_keys():
    good = make_spec().to_dict()
    with pytest.raises(ValueError, match="unknown"):
        ExperimentSpec.from_dict({**good, "extra": 1})
    bad = dict(good)
    del bad["evaluation"]
    with pytest.raises(ValueError, match="missing"):
        ExperimentSpec.from_dict(bad)


@pytest.mark.parametrize(
    "field,value",
    [
        ("experiment_id", "has space"),
        ("pair_id", "/leading"),
        ("condition", ".dot"),
        ("condition", ""),
    ],
)
def test_path_safe_slugs_enforced(field, value):
    with pytest.raises(ValueError, match="path-safe slug"):
        make_spec(**{field: value})


def test_n_generate_and_repeat_id_validation():
    with pytest.raises(ValueError, match="n_generate"):
        make_spec(n_generate=0)
    with pytest.raises(ValueError, match="repeat_id"):
        make_spec(repeat_id=-1)


def test_evaluation_config_validation():
    with pytest.raises(ValueError, match="n_true"):
        EvaluationConfig(n_true=1)
    with pytest.raises(ValueError, match="kernel scale"):
        EvaluationConfig(lambdas=(4.0, -1.0))
    with pytest.raises(ValueError, match="nonempty"):
        EvaluationConfig(lambdas=())


def test_identities_are_derived_not_serialized():
    spec = make_spec()
    ids = spec_identities(spec)
    assert set(ids) == {"model", "optimizer", "sampler"}
    assert ids["model"]["v_policy"] == spec.model.v_policy
    assert ids["sampler"]["sampler_name"] == spec.sampler.sampler_name
    assert "identities" not in spec.to_dict()


def test_to_run_config_matches_spec_fields():
    spec = make_spec()
    config = spec.to_run_config()
    assert config.dimensions == spec.dimensions
    assert config.seeds == spec.seeds
    assert config.n_generate == spec.n_generate


def test_uturn_defaults_to_none_and_round_trips():
    spec = make_spec()
    assert spec.uturn is None
    assert spec.to_dict()["uturn"] is None
    assert ExperimentSpec.from_dict(spec.to_dict()).uturn is None

    with_uturn = make_spec(
        uturn=UTurnConfig(t_values=(0.2, 0.5, 0.8), n_examples=4, sources=("train", "fresh"))
    )
    restored = ExperimentSpec.from_dict(with_uturn.to_dict())
    assert restored.uturn == with_uturn.uturn
    assert restored == with_uturn


def test_uturn_changes_fingerprint():
    base = make_spec()
    with_uturn = make_spec(uturn=UTurnConfig(t_values=(0.5,), n_examples=2))
    assert spec_fingerprint(base) != spec_fingerprint(with_uturn)
