"""Phase 4C paired-experiment engine and canonical run-record schema.

One package, one namespace (docs/PHASE4C_EXPERIMENT_PROTOCOL.md): serializable
per-arm specs (`ExperimentSpec`), the intervention registry (`INTERVENTIONS`),
strict paired-disorder validation (`validate_pair` / `validate_group`),
deterministic plan expansion and pre-run manifests (`load_experiment_config`,
`ExperimentPlan`), the resumable executor (`execute_plan`), the optional
U-turn stage, and the canonical `Phase4CRunRecord` schema
(`docs/PHASE4C_ANALYSIS_SPEC.md` §0) that the engine writes and the analysis
layer (`maskeddiffusion.analysis.ingest`) is the only other code allowed to
read.
"""

from .interventions import INTERVENTIONS, ArmDefinition, Intervention, get_intervention
from .pairs import diff_leaves, validate_group, validate_pair
from .plan import (
    PLAN_SCHEMA_VERSION,
    REPEAT_SEED_STRIDE,
    ExperimentPlan,
    load_experiment_config,
)
from .runner import (
    PAIR_SCHEMA_VERSION,
    RUN_SCHEMA_VERSION,
    execute_plan,
)
from .schema import (
    RUN_RECORD_FILENAME,
    RUN_RECORD_SCHEMA_VERSION,
    UTURN_BLOCK_KEYS,
    Phase4CRunRecord,
    backfill_migration_block,
    build_run_record,
    check_uturn_block,
    load_run_record,
    model_config_digest,
    verify_artifact_hashes,
    write_run_record,
)
from .spec import (
    EvaluationConfig,
    ExperimentSpec,
    spec_fingerprint,
    spec_identities,
)
from .uturn_stage import load_uturn_block, run_uturn_stage, uturn_summary_to_record_block

__all__ = [
    "INTERVENTIONS",
    "PAIR_SCHEMA_VERSION",
    "PLAN_SCHEMA_VERSION",
    "REPEAT_SEED_STRIDE",
    "RUN_RECORD_FILENAME",
    "RUN_RECORD_SCHEMA_VERSION",
    "RUN_SCHEMA_VERSION",
    "UTURN_BLOCK_KEYS",
    "ArmDefinition",
    "EvaluationConfig",
    "ExperimentPlan",
    "ExperimentSpec",
    "Intervention",
    "Phase4CRunRecord",
    "backfill_migration_block",
    "build_run_record",
    "check_uturn_block",
    "diff_leaves",
    "execute_plan",
    "get_intervention",
    "load_experiment_config",
    "load_run_record",
    "load_uturn_block",
    "model_config_digest",
    "run_uturn_stage",
    "spec_fingerprint",
    "spec_identities",
    "uturn_summary_to_record_block",
    "validate_group",
    "validate_pair",
    "verify_artifact_hashes",
    "write_run_record",
]
