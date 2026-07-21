"""Experiment-plan loading and the deterministic pre-run manifest.

An experiment TOML file defines ONE intervention over a shared base
configuration; `load_experiment_config` expands it deterministically into
one `ExperimentSpec` per (repeat, condition):

- repeat seeds derive as `SeedHierarchy.from_base(base_seed +
  REPEAT_SEED_STRIDE * repeat_id)` — the stride is safely above the
  per-stream offsets (<= 1153) used by `SeedHierarchy.from_base`, so repeat
  streams never collide;
- `pair_id = f"{experiment_id}-r{repeat_id:03d}"` identifies the comparison
  group of one repeat; every arm of the group shares the full seed
  hierarchy verbatim, enforced by `experiments.pairs.validate_group` at load
  time (plans fail fast) — but shared seed values only make arms
  content-identical disorder when the intervention's `comparison_type` is
  `"paired_disorder"`; `finite_d` is `"matched_seed_finite_size"`
  (`experiments.interventions` module docstring) and must never be treated
  as the same kind of comparison;
- the intervened field must come from the intervention ALONE: e.g.
  `[model].v_policy` is forbidden under `v_trainability`,
  `[training].max_steps` under `optimization_budget`,
  `[sampler].sampler_name` under `sampler_stochasticity`, and
  `[dimensions].latent_dim` under `finite_d`. Exactly one source of truth
  per field — no silent precedence.

`ExperimentPlan.write_plan_manifest` writes the pre-run manifest
(`experiment_manifest.json`) BEFORE any run starts. Its `plan` section is a
deterministic function of the TOML file; the `provenance` section carries
timestamp/git/environment. On resume the existing manifest's `plan` section
must match the current plan exactly, otherwise execution refuses rather
than mixing runs from different plans in one directory.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..artifacts import environment_metadata, sha256_file
from ..config import TrainingConfig, _reject_forbidden
from ..dimensions import Dimensions
from ..models import LinearScoreConfig
from ..randomness import SeedHierarchy
from ..samplers import SamplerConfig
from ..uturn import UTURN_SOURCES, UTurnConfig
from .interventions import get_intervention
from .pairs import validate_group
from .spec import EvaluationConfig, ExperimentSpec, spec_fingerprint, spec_identities

PLAN_SCHEMA_VERSION = "maskeddiffusion.experiment_plan.v1"

# Per-repeat base-seed stride (see module docstring).
REPEAT_SEED_STRIDE = 10_000

_KNOWN_TOP_LEVEL = {
    "experiment",
    "intervention",
    "dimensions",
    "model",
    "training",
    "sampler",
    "evaluation",
    "n_generate",
    "uturn",
}


def _require_key(table: dict[str, Any], key: str, where: str) -> Any:
    if key not in table:
        raise ValueError(f"experiment config must set [{where}].{key}")
    return table[key]


@dataclass(frozen=True)
class ExperimentPlan:
    """The fully expanded, deterministic plan: one spec per (repeat, condition)."""

    experiment_id: str
    intervention: str
    comparison_type: str
    repeats: int
    base_seed: int
    specs: tuple[ExperimentSpec, ...]
    source_config_sha256: str | None = None

    @property
    def conditions(self) -> tuple[str, ...]:
        seen: list[str] = []
        for spec in self.specs:
            if spec.condition not in seen:
                seen.append(spec.condition)
        return tuple(seen)

    def groups(self) -> dict[str, tuple[ExperimentSpec, ...]]:
        """pair_id -> the group's specs, in plan order."""
        groups: dict[str, list[ExperimentSpec]] = {}
        for spec in self.specs:
            groups.setdefault(spec.pair_id, []).append(spec)
        return {pair_id: tuple(specs) for pair_id, specs in groups.items()}

    def experiment_root(self, output_root: str | Path) -> Path:
        return Path(output_root) / self.experiment_id

    def run_dir(self, output_root: str | Path, spec: ExperimentSpec) -> Path:
        return self.experiment_root(output_root) / spec.pair_id / spec.condition

    def dry_run_dict(self, output_root: str | Path) -> dict[str, Any]:
        """Exact run count, dimensions, seeds, projected sample counts, paths."""
        runs = []
        for spec in self.specs:
            run_dir = self.run_dir(output_root, spec)
            runs.append(
                {
                    "pair_id": spec.pair_id,
                    "repeat_id": spec.repeat_id,
                    "condition": spec.condition,
                    "dimensions": spec.dimensions.to_dict(),
                    "seeds": spec.seeds.to_dict(),
                    "identities": spec_identities(spec),
                    "training": asdict(spec.training),
                    "evaluation": spec.evaluation.to_dict(),
                    "n_generate": spec.n_generate,
                    "uturn": (
                        {
                            "t_values": list(spec.uturn.t_values),
                            "n_examples": spec.uturn.n_examples,
                            "sources": list(spec.uturn.sources),
                        }
                        if spec.uturn is not None
                        else None
                    ),
                    "spec_fingerprint": spec_fingerprint(spec),
                    "paths": {
                        "run_dir": str(run_dir),
                        "run_manifest": str(run_dir / "run_manifest.json"),
                        "train": str(run_dir / "train"),
                        "samples": str(run_dir / "samples"),
                        "eval": str(run_dir / "eval"),
                        **({"uturn": str(run_dir / "uturn")} if spec.uturn is not None else {}),
                    },
                }
            )
        n_generate_values = {spec.n_generate for spec in self.specs}
        n_true_values = {spec.evaluation.n_true for spec in self.specs}
        root = self.experiment_root(output_root)
        return {
            "experiment_id": self.experiment_id,
            "intervention": self.intervention,
            "comparison_type": self.comparison_type,
            "repeats": self.repeats,
            "n_pairs": len(self.groups()),
            "n_conditions": len(self.conditions),
            "n_runs": len(self.specs),
            "permitted_diff_fields": sorted(
                get_intervention(self.intervention).permitted_diff_fields
            ),
            "runs": runs,
            "projected": {
                "model_samples_per_run": (
                    n_generate_values.pop() if len(n_generate_values) == 1 else "varies"
                ),
                "model_samples_total": sum(spec.n_generate for spec in self.specs),
                "evaluation_true_samples_per_run": (
                    2 * n_true_values.pop() if len(n_true_values) == 1 else "varies"
                ),
            },
            "paths": {
                "experiment_root": str(root),
                "experiment_manifest": str(root / "experiment_manifest.json"),
            },
        }

    def plan_manifest_dict(
        self, output_root: str | Path, *, device: str, config_path: str | Path | None
    ) -> dict[str, Any]:
        specs_section = []
        for spec in self.specs:
            specs_section.append(
                {
                    **spec.to_dict(),
                    "identities": spec_identities(spec),
                    "spec_fingerprint": spec_fingerprint(spec),
                    "run_dir": str(self.run_dir(output_root, spec)),
                }
            )
        return {
            "schema_version": PLAN_SCHEMA_VERSION,
            "plan": {
                "experiment_id": self.experiment_id,
                "intervention": self.intervention,
                "comparison_type": self.comparison_type,
                "repeats": self.repeats,
                "base_seed": self.base_seed,
                "n_runs": len(self.specs),
                "permitted_diff_fields": sorted(
                    get_intervention(self.intervention).permitted_diff_fields
                ),
                "specs": specs_section,
                "projected": self.dry_run_dict(output_root)["projected"],
            },
            "provenance": {
                "timestamp": datetime.now(UTC).isoformat(),
                "config_path": str(config_path) if config_path is not None else None,
                "config_sha256": (sha256_file(config_path) if config_path is not None else None),
                **environment_metadata(device),
            },
        }

    def write_plan_manifest(
        self, output_root: str | Path, *, device: str, config_path: str | Path | None
    ) -> Path:
        """Write the pre-run manifest; on resume verify it matches this plan.

        Never overwrites an existing manifest whose `plan` section matches
        (keeps the original provenance/timestamp, so a resume run leaves the
        experiment directory byte-stable); refuses to proceed if it disagrees.
        """
        root = self.experiment_root(output_root)
        root.mkdir(parents=True, exist_ok=True)
        path = root / "experiment_manifest.json"
        manifest = self.plan_manifest_dict(output_root, device=device, config_path=config_path)
        if path.exists():
            try:
                existing = json.loads(path.read_text())
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"existing experiment manifest {path} is malformed ({e}); "
                    "refusing to proceed — move the directory aside manually"
                ) from e
            if (
                existing.get("schema_version") != PLAN_SCHEMA_VERSION
                or existing.get("plan") != manifest["plan"]
            ):
                raise ValueError(
                    f"existing experiment manifest {path} describes a different plan "
                    "than the current config — refusing to mix runs from different "
                    "plans in one directory; use a fresh --output or move the old "
                    "directory aside manually"
                )
            return path
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        return path


def load_experiment_config(path: str | Path) -> ExperimentPlan:
    """Load and expand an experiment TOML file into an `ExperimentPlan`."""
    path = Path(path)
    raw = tomllib.loads(path.read_text())
    _reject_forbidden(raw, "root")
    if "seeds" in raw:
        raise ValueError(
            "experiment configs take no [seeds] table: every stream derives "
            "deterministically from [experiment].base_seed "
            "(base_seed + REPEAT_SEED_STRIDE * repeat_id per repeat)"
        )
    unknown_top = set(raw) - _KNOWN_TOP_LEVEL
    if unknown_top:
        raise ValueError(
            f"unknown top-level experiment config keys {sorted(unknown_top)}; "
            f"known: {sorted(_KNOWN_TOP_LEVEL)}"
        )

    exp_raw = raw.get("experiment")
    if exp_raw is None:
        raise ValueError("experiment config must contain an [experiment] table")
    experiment_id = _require_key(exp_raw, "experiment_id", "experiment")
    base_seed = _require_key(exp_raw, "base_seed", "experiment")
    if not isinstance(base_seed, int) or isinstance(base_seed, bool):
        raise TypeError(f"base_seed must be int, got {type(base_seed).__name__}")
    repeats = exp_raw.get("repeats", 1)
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        raise ValueError(f"repeats must be an int >= 1, got {repeats!r}")

    int_raw = raw.get("intervention")
    if int_raw is None:
        raise ValueError("experiment config must contain an [intervention] table")
    name = _require_key(int_raw, "name", "intervention")
    intervention = get_intervention(name)
    params = {k: v for k, v in int_raw.items() if k != "name"}
    arms = intervention.build_arms(params)

    if "n_generate" not in raw:
        raise ValueError(
            "experiment config must set n_generate explicitly — an experiment "
            "arm's generated-sample count is never defaulted"
        )
    n_generate = raw["n_generate"]

    dims_raw = raw.get("dimensions")
    if dims_raw is None:
        raise ValueError("experiment config must contain a [dimensions] table")
    aspect_ratio = _require_key(dims_raw, "aspect_ratio", "dimensions")
    sample_ratio = _require_key(dims_raw, "sample_ratio", "dimensions")
    base_latent_dim = dims_raw.get("latent_dim")
    if intervention.name == "finite_d":
        if base_latent_dim is not None:
            raise ValueError(
                "[dimensions].latent_dim must come from the finite_d intervention's "
                "latent_dims list, not the base [dimensions] table — remove it"
            )
    elif base_latent_dim is None:
        raise ValueError("experiment config must set [dimensions].latent_dim")

    model_raw = dict(raw.get("model", {}))
    if "visible_dim" in model_raw:
        raise ValueError("[model].visible_dim is derived from dimensions — remove it")
    if intervention.name == "v_trainability" and "v_policy" in model_raw:
        raise ValueError(
            "[model].v_policy must come from the v_trainability intervention, not "
            "the base [model] table — remove it"
        )

    training_raw = dict(raw.get("training", {}))
    if intervention.name == "optimization_budget" and "max_steps" in training_raw:
        raise ValueError(
            "[training].max_steps must come from the optimization_budget "
            "intervention's budgets, not the base [training] table — remove it"
        )

    sampler_raw = dict(raw.get("sampler", {}))
    if intervention.name == "sampler_stochasticity" and "sampler_name" in sampler_raw:
        raise ValueError(
            "[sampler].sampler_name must come from the sampler_stochasticity "
            "intervention's sampler_family, not the base [sampler] table — remove it"
        )

    eval_raw = raw.get("evaluation", {})
    evaluation = EvaluationConfig(
        n_true=eval_raw.get("n_true", 1000),
        lambdas=tuple(eval_raw.get("lambdas", (4.0, 8.0))),
    )

    uturn: UTurnConfig | None = None
    if "uturn" in raw:
        uturn_raw = raw["uturn"]
        t_values = _require_key(uturn_raw, "t_values", "uturn")
        n_examples = _require_key(uturn_raw, "n_examples", "uturn")
        uturn = UTurnConfig(
            t_values=tuple(t_values),
            n_examples=n_examples,
            sources=tuple(uturn_raw.get("sources", UTURN_SOURCES)),
        )

    specs: list[ExperimentSpec] = []
    for repeat_id in range(repeats):
        seeds = SeedHierarchy.from_base(base_seed + REPEAT_SEED_STRIDE * repeat_id)
        pair_id = f"{experiment_id}-r{repeat_id:03d}"
        group: list[ExperimentSpec] = []
        for arm in arms:
            latent_dim = arm.overrides.get("dimensions", {}).get("latent_dim", base_latent_dim)
            dims = Dimensions.resolve(
                latent_dim=latent_dim,
                aspect_ratio=aspect_ratio,
                sample_ratio=sample_ratio,
            )
            model = LinearScoreConfig(
                visible_dim=dims.visible_dim,
                **{**model_raw, **arm.overrides.get("model", {})},
            )
            training = TrainingConfig(**{**training_raw, **arm.overrides.get("training", {})})
            sampler_merged = {**sampler_raw, **arm.overrides.get("sampler", {})}
            sampler = (
                SamplerConfig(**sampler_merged)
                if sampler_merged
                else SamplerConfig("sequential_random_stochastic")
            )
            group.append(
                ExperimentSpec(
                    experiment_id=experiment_id,
                    pair_id=pair_id,
                    repeat_id=repeat_id,
                    intervention=intervention.name,
                    condition=arm.condition,
                    dimensions=dims,
                    seeds=seeds,
                    model=model,
                    training=training,
                    sampler=sampler,
                    evaluation=evaluation,
                    n_generate=n_generate,
                    uturn=uturn,
                )
            )
        problems = validate_group(group)
        if problems:
            raise ValueError(
                f"comparison group {pair_id!r} failed paired-disorder validation: "
                + "; ".join(problems)
            )
        specs.extend(group)

    return ExperimentPlan(
        experiment_id=experiment_id,
        intervention=intervention.name,
        comparison_type=intervention.comparison_type,
        repeats=repeats,
        base_seed=base_seed,
        specs=tuple(specs),
        source_config_sha256=sha256_file(path),
    )
