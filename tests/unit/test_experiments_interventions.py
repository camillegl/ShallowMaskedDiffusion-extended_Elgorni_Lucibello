"""Intervention registry: arm generation and parameter validation."""

from __future__ import annotations

import pytest

from maskeddiffusion.experiments.interventions import (
    INTERVENTIONS,
    SAMPLER_FAMILIES,
    get_intervention,
)


def test_registry_names_and_comparison_types():
    assert set(INTERVENTIONS) == {
        "v_trainability",
        "sampler_stochasticity",
        "optimization_budget",
        "finite_d",
    }
    assert INTERVENTIONS["finite_d"].comparison_type == "matched_seed_finite_size"
    for name in ("v_trainability", "sampler_stochasticity", "optimization_budget"):
        assert INTERVENTIONS[name].comparison_type == "paired_disorder"


def test_unknown_intervention_rejected():
    with pytest.raises(ValueError, match="unknown intervention"):
        get_intervention("nope")


def test_v_trainability_arms():
    arms = get_intervention("v_trainability").build_arms({})
    assert [a.condition for a in arms] == ["frozen_zero_v", "trainable_v"]
    assert arms[0].overrides == {"model": {"v_policy": "frozen_zero"}}
    assert arms[1].overrides == {"model": {"v_policy": "trainable"}}
    with pytest.raises(ValueError, match="unknown parameters"):
        get_intervention("v_trainability").build_arms({"stray": 1})


@pytest.mark.parametrize("family", sorted(SAMPLER_FAMILIES))
def test_sampler_stochasticity_arms(family):
    arms = get_intervention("sampler_stochasticity").build_arms({"sampler_family": family})
    stochastic, greedy = SAMPLER_FAMILIES[family]
    assert arms[0].overrides == {"sampler": {"sampler_name": stochastic}}
    assert arms[1].overrides == {"sampler": {"sampler_name": greedy}}


def test_sampler_stochasticity_requires_known_family():
    with pytest.raises(ValueError, match="sampler_family"):
        get_intervention("sampler_stochasticity").build_arms({})
    with pytest.raises(ValueError, match="sampler_family"):
        get_intervention("sampler_stochasticity").build_arms({"sampler_family": "bogus"})


def test_optimization_budget_arms():
    arms = get_intervention("optimization_budget").build_arms(
        {"budgets": {"short": 60, "long": 600}}
    )
    assert {a.condition for a in arms} == {"short", "long"}
    assert {a.overrides["training"]["max_steps"] for a in arms} == {60, 600}
    with pytest.raises(ValueError, match=">= 2"):
        get_intervention("optimization_budget").build_arms({"budgets": {"only": 60}})
    with pytest.raises(ValueError, match="distinct"):
        get_intervention("optimization_budget").build_arms({"budgets": {"a": 60, "b": 60}})
    with pytest.raises(ValueError, match="max_steps"):
        get_intervention("optimization_budget").build_arms({"budgets": {"a": 60, "b": 0}})


def test_finite_d_arms():
    arms = get_intervention("finite_d").build_arms({"latent_dims": [8, 16]})
    assert [a.condition for a in arms] == ["d8", "d16"]
    assert arms[1].overrides == {"dimensions": {"latent_dim": 16}}
    with pytest.raises(ValueError, match="distinct"):
        get_intervention("finite_d").build_arms({"latent_dims": [8, 8]})
    with pytest.raises(ValueError, match="latent_dims"):
        get_intervention("finite_d").build_arms({"latent_dims": [8]})
