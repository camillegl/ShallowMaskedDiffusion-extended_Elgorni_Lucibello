"""Experiment-plan loading: deterministic expansion, pairing, config hygiene."""

from __future__ import annotations

from pathlib import Path

import pytest

from maskeddiffusion.experiments.pairs import validate_pair
from maskeddiffusion.experiments.plan import (
    REPEAT_SEED_STRIDE,
    load_experiment_config,
)

SMOKE_DIR = Path(__file__).resolve().parents[2] / "configs" / "experiments" / "smoke_d8"
PILOT_DIR = Path(__file__).resolve().parents[2] / "configs" / "experiments" / "pilot"
CAMPAIGN_V1_DIR = Path(__file__).resolve().parents[2] / "configs" / "experiments" / "campaign_v1"

BASE_TOML = """
n_generate = 4

[experiment]
experiment_id = "unit-plan"
repeats = 2
base_seed = 123

[intervention]
name = "v_trainability"

[dimensions]
latent_dim = 8
aspect_ratio = 2.0
sample_ratio = 3.0

[training]
max_steps = 4
batch_size = 8

[evaluation]
n_true = 8
"""


def write_config(tmp_path: Path, text: str = BASE_TOML) -> Path:
    path = tmp_path / "plan.toml"
    path.write_text(text)
    return path


def test_expansion_is_deterministic_and_paired(tmp_path):
    path = write_config(tmp_path)
    plan_a = load_experiment_config(path)
    plan_b = load_experiment_config(path)
    assert [s.to_dict() for s in plan_a.specs] == [s.to_dict() for s in plan_b.specs]
    assert plan_a.repeats == 2
    assert len(plan_a.specs) == 4  # 2 repeats x 2 conditions
    groups = plan_a.groups()
    assert sorted(groups) == ["unit-plan-r000", "unit-plan-r001"]
    for specs in groups.values():
        a, b = specs
        assert validate_pair(a, b) == []
        assert a.seeds == b.seeds  # full hierarchy shared verbatim


def test_repeat_seed_stride_separates_repeats(tmp_path):
    plan = load_experiment_config(write_config(tmp_path))
    r0 = plan.groups()["unit-plan-r000"][0].seeds
    r1 = plan.groups()["unit-plan-r001"][0].seeds
    assert r1.base_seed - r0.base_seed == REPEAT_SEED_STRIDE
    assert r1.teacher_seed != r0.teacher_seed


def test_intervened_field_forbidden_in_base_tables(tmp_path):
    hijacked = BASE_TOML.replace("[training]", '[model]\nv_policy = "trainable"\n\n[training]')
    with pytest.raises(ValueError, match="v_policy"):
        load_experiment_config(write_config(tmp_path, hijacked))


def test_explicit_seeds_table_rejected(tmp_path):
    with pytest.raises(ValueError, match="no \\[seeds\\] table"):
        load_experiment_config(write_config(tmp_path, BASE_TOML + "\n[seeds]\nbase_seed = 1\n"))


def test_n_generate_must_be_explicit(tmp_path):
    with pytest.raises(ValueError, match="n_generate"):
        load_experiment_config(write_config(tmp_path, BASE_TOML.replace("n_generate = 4\n", "")))


def test_unknown_top_level_key_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown top-level"):
        load_experiment_config(write_config(tmp_path, BASE_TOML + "\n[stray]\nx = 1\n"))


def test_dry_run_dict_reports_exact_counts(tmp_path):
    plan = load_experiment_config(write_config(tmp_path))
    dry = plan.dry_run_dict(tmp_path / "out")
    assert dry["n_runs"] == 4
    assert dry["n_pairs"] == 2
    assert dry["n_conditions"] == 2
    assert dry["projected"]["model_samples_total"] == 16
    assert dry["projected"]["evaluation_true_samples_per_run"] == 16
    assert all("spec_fingerprint" in run for run in dry["runs"])


@pytest.mark.parametrize("name", sorted(p.name for p in SMOKE_DIR.glob("*.toml")))
def test_committed_smoke_configs_load_and_validate(name):
    plan = load_experiment_config(SMOKE_DIR / name)
    assert plan.specs
    for specs in plan.groups().values():
        assert len(specs) >= 2


@pytest.mark.parametrize("name", sorted(p.name for p in PILOT_DIR.glob("*.toml")))
def test_committed_pilot_configs_load_and_validate(name):
    plan = load_experiment_config(PILOT_DIR / name)
    assert plan.specs
    for specs in plan.groups().values():
        assert len(specs) >= 2


@pytest.mark.parametrize("name", sorted(p.name for p in CAMPAIGN_V1_DIR.glob("*.toml")))
def test_committed_campaign_v1_configs_load_and_validate(name):
    """Load-and-validate only — campaign_v1 remains a plan, never executed
    by this suite (docs/PHASE4C_EXPERIMENT_PROTOCOL.md §7)."""
    plan = load_experiment_config(CAMPAIGN_V1_DIR / name)
    assert plan.specs
    for specs in plan.groups().values():
        assert len(specs) >= 2


def test_pilot_and_campaign_v1_dirs_are_nonempty():
    assert list(PILOT_DIR.glob("*.toml"))
    assert len(list(CAMPAIGN_V1_DIR.glob("*.toml"))) == 14


def test_uturn_table_is_optional_and_parsed(tmp_path):
    plan_without = load_experiment_config(write_config(tmp_path))
    assert all(s.uturn is None for s in plan_without.specs)

    toml = (
        BASE_TOML
        + "\n[uturn]\nt_values = [0.2, 0.5, 0.8]\nn_examples = 4\n"
        + 'sources = ["train", "fresh"]\n'
    )
    plan_with = load_experiment_config(write_config(tmp_path, toml))
    for spec in plan_with.specs:
        assert spec.uturn is not None
        assert spec.uturn.t_values == (0.2, 0.5, 0.8)
        assert spec.uturn.n_examples == 4
        assert spec.uturn.sources == ("train", "fresh")
    # Paired arms share the uturn config exactly, like every other shared field.
    a, b = plan_with.groups()["unit-plan-r000"]
    assert a.uturn == b.uturn


def test_uturn_sources_default_to_both(tmp_path):
    toml = BASE_TOML + "\n[uturn]\nt_values = [0.5]\nn_examples = 2\n"
    plan = load_experiment_config(write_config(tmp_path, toml))
    assert plan.specs[0].uturn.sources == ("train", "fresh")


def test_uturn_requires_t_values_and_n_examples(tmp_path):
    with pytest.raises(ValueError, match="t_values"):
        load_experiment_config(write_config(tmp_path, BASE_TOML + "\n[uturn]\nn_examples = 2\n"))
    with pytest.raises(ValueError, match="n_examples"):
        load_experiment_config(write_config(tmp_path, BASE_TOML + "\n[uturn]\nt_values = [0.5]\n"))


def test_finite_d_arms_share_seed_values_not_dimensions(tmp_path):
    toml = BASE_TOML.replace(
        'name = "v_trainability"', 'name = "finite_d"\nlatent_dims = [8, 16]'
    ).replace("latent_dim = 8\n", "")
    plan = load_experiment_config(write_config(tmp_path, toml))
    a, b = plan.groups()["unit-plan-r000"]
    assert a.seeds == b.seeds
    assert a.dimensions.latent_dim != b.dimensions.latent_dim
    assert a.dimensions.aspect_ratio == b.dimensions.aspect_ratio
    assert plan.comparison_type == "matched_seed_finite_size"
