# Phase 4C paired-experiment protocol

Status: **active**. This document is the protocol contract referenced by the
engine (`src/maskeddiffusion/experiments/`), its CLI
(`maskeddiffusion-experiment`), the canonical run-record schema
(`src/maskeddiffusion/experiments/schema.py`), and the experiment configs
under `configs/experiments/`. Notation and claim discipline follow
`docs/NOTATION.md` and `docs/RESEARCH_SPEC.md`; the analysis-side contract is
`docs/PHASE4C_ANALYSIS_SPEC.md`.

## 1. What one experiment is

One experiment TOML defines ONE intervention over a shared base
configuration at ONE design cell (fixed `latent_dim`, `aspect_ratio`,
`sample_ratio` — except `finite_d`, which varies `latent_dim` by design).
`load_experiment_config` expands it deterministically into one
`ExperimentSpec` per (repeat, condition):

- `pair_id = "{experiment_id}-r{repeat_id:03d}"` names the comparison group
  of one disorder repeat;
- every arm of a group shares the FULL ten-stream seed hierarchy verbatim
  (paired disorder): same quenched teacher `F`, same training/validation/
  evaluation latent streams, same masking, sampler, and metric seeds;
- only the fields named by the intervention's `permitted_diff_fields` (plus
  the `condition` label) may differ between arms — enforced by a recursive
  diff of the specs' serialized state (`experiments.pairs`), at plan-load
  time and again after execution.

Repeat seeds derive as `base_seed + 10_000 * repeat_id`
(`plan.REPEAT_SEED_STRIDE`, safely above the per-stream offsets of
`SeedHierarchy.from_base`). Disorder averaging happens only across explicit
repeats; a single repeat supports no disorder-averaged statement.

## 2. Interventions

Registered in `experiments.interventions.INTERVENTIONS`:

| name | varies | comparison_type | notes |
|---|---|---|---|
| `v_trainability` | `model.v_policy` (`frozen_zero` vs `trainable`) | `paired_disorder` | open question Q1: `V ≡ 0` is an experimental restriction, not derived for fixed `F` |
| `sampler_stochasticity` | `sampler.sampler_name` within one declared family | `paired_disorder` | e.g. `sequential_random_stochastic` vs `sequential_random_greedy`; arms share the trained checkpoint's configuration and seeds, so they compare sampler-indexed terminal laws `P_{θ,A,k}` |
| `optimization_budget` | `training.max_steps` | `paired_disorder` | identical seeds: the shorter trajectory is a prefix of the longer one |
| `finite_d` | `dimensions.latent_dim` at fixed ratios | `matched_seed_finite_size` | different `latent_dim` ⇒ a different-shape teacher ⇒ a distinct `teacher_id` by construction — never a same-disorder pairing, no matter the seed values; see §2a |

The intervened field must come from the intervention alone; setting it in
the base table is rejected (one source of truth per field).

## 2a. Comparison types (binding)

Every intervention has exactly one `comparison_type`
(`experiments.interventions.Intervention.comparison_type`); this is the
single source of truth every other layer (`experiments.pairs`,
`experiments.runner`'s pair manifest, `analysis.rows.comparison_type`,
`analysis.statistics.paired_differences`) keys off of. Two types exist:

- **`paired_disorder`** (`v_trainability`, `sampler_stochasticity`,
  `optimization_budget`): arms share the full seed hierarchy verbatim AND,
  on CPU, therefore produce content-identical disorder — same quenched
  teacher `F` (equal `teacher_id`), same training/validation/evaluation
  draws. This is the comparison `p4c_paired_differences.csv` computes a
  B−A delta over.
- **`matched_seed_finite_size`** (`finite_d` only): arms share every seed
  VALUE, the same repeat index, and the same `aspect_ratio`/`sample_ratio`,
  but NOT disorder content — a different `latent_dim` is a different-shape
  teacher `F`, so it is necessarily a distinct draw with a DISTINCT
  `teacher_id` (asserted, never merely tolerated: a collision is treated as
  a bug). This is fundamentally not the same kind of comparison as
  `paired_disorder` and must never be described, computed, or reported as
  one:
  - the pair manifest records `comparison_type: "matched_seed_finite_size"`
    and a `comparison_checks` block distinct in shape from a
    `paired_disorder` group's (`same_repeat_index`,
    `same_seed_hierarchy_values`, `same_ratios`, `distinct_dimensions`,
    `distinct_teacher_id`);
  - `analysis.statistics.paired_differences` excludes these rows by
    default;
  - the only place they appear paired-ish is the explicitly labeled,
    non-differenced `p4c_matched_seed_finite_size.csv`
    (`docs/PHASE4C_ANALYSIS_SPEC.md` §3a), one row per run.

## 3. Optional U-turn stage

An experiment TOML may include a `[uturn]` table (`t_values`, `n_examples`,
optional `sources` — default both `train` and `fresh`; same shape as
`UTurnConfig`, see `docs/RESEARCH_SPEC.md` and `uturn.py`'s module
docstring for the protocol itself). When present, every arm runs the
U-turn / reconstruction protocol against its own trained checkpoint after
the evaluate stage, writing a fourth ADR-003 artifact,
`<run_dir>/uturn/`. Arms of a pair share the U-turn config exactly (it is
not an interventable field for any registered intervention).

The canonical run record's `uturn` block is a **reduced summary**, not the
full per-source result: `mask_densities`/`overlap` is a single curve, and
`baseline_recovery`/`excess_recovery` are single scalars (mean over the
`t_values` sweep of the no-recovery baseline and excess recovery). The
curve and scalars come from the **fresh-source** result when `fresh` is
among `sources` (a fresh-teacher-draw recovery curve is interpretable on
its own), falling back to `train` only when `fresh` was not run — a
train-source curve alone must never be read as a memorization signal
(`uturn.py` module docstring). The full per-source data, including the
train-vs-fresh comparison block, is never discarded — it stays in the
stage artifact's own `summary.json`
(`experiments.uturn_stage.uturn_summary_to_record_block`).

## 4. Run layout and resume discipline

Per condition the engine runs train → sample → evaluate (→ uturn, if
configured) into
`<output>/<experiment_id>/<pair_id>/<condition>/{train,samples,eval[,uturn]}/`,
each a standard ADR-003 artifact, then writes:

- `run_manifest.json` (`maskeddiffusion.experiment_run.v1`) — spec, spec
  fingerprint, teacher id, content hashes of the exact train/validation
  tensors used; written LAST, so its presence marks completion;
- `run_record.json` (`maskeddiffusion.p4c_run_record.v1`) — the canonical
  record consumed by the analysis layer (see §5);
- per comparison group, `pair_manifest.json`
  (`maskeddiffusion.experiment_pair.v1`) — recorded teacher-id and
  data-hash equality across arms (required for strict interventions,
  recorded-but-not-required for `finite_d`);
- per experiment, a pre-run `experiment_manifest.json`
  (`maskeddiffusion.experiment_plan.v1`) written BEFORE any run starts.

Resume: a missing run directory is executed fresh; a completed, spec- and
fingerprint-matching, artifact-valid directory is skipped byte-stably.
`run_record.json` is IMMUTABLE once written: on a skip, an existing record
is only re-validated (`load_run_record`), never rebuilt; a missing one (a
run that completed before a record existed for it, or whose record was
lost) is built exactly once, carrying an explicit `migration` block
(`{from_schema: null, to_schema, performed_at, source_artifacts_unchanged:
true}`, `experiments.schema.backfill_migration_block`) so a
migration-constructed record is never indistinguishable from one written
at original completion. Anything else (missing/malformed/partial manifest,
spec or fingerprint mismatch, failing stage validation, disagreeing
resolved config) is REFUSED with the exact problems listed. The engine
never overwrites existing run output; recovery is a manual decision to
move a directory aside.

The evaluation stage reuses `maskeddiffusion.cli.evaluate.evaluate_run`, so
every run inherits the CLI's checkpoint/teacher/sample provenance
verification (semantic checkpoint identity, raw file SHA-256,
checkpoint-recorded training config, sampler identity from the sample
artifact's own manifest).

## 5. Canonical run record

`maskeddiffusion.experiments.schema.Phase4CRunRecord` is the ONLY interface
between the engine's outputs and the analysis layer
(`maskeddiffusion.analysis.ingest`, `maskeddiffusion-p4c-analyze`).
Analysis never reads run directories directly and never infers metadata
from filenames. A record is "complete provenance" — not just a digest — and
carries:

- **completion**: `schema_version`, `status` (currently always
  `"completed"` — only completed runs get a record), `validation_problems`
  (stage-artifact validation findings at record-build time; a nonempty list
  makes the analysis layer reject the row explicitly);
- **identity**: `experiment_id`/`pair_id`/`repeat_id`/`intervention`/
  `condition`, resolved `dimensions`, the full `seeds` hierarchy, the
  `spec_fingerprint`, `teacher_id`, `checkpoint_id` +
  `checkpoint_file_sha256`, `sampler`/`model` identities;
- **the canonical resolved config**: `resolved_config` (the exact
  `RunConfig` dict the run was executed against, read from
  `train/resolved_config.json` — dimensions, seeds, model, training,
  sampler, n_generate, sufficient to reconstruct the run) plus
  `resolved_config_sha256` (the file's own hash, so tampering with the
  linked file — not just the embedded copy — is detectable);
- **artifact links**: `artifacts`, a `{stage: {manifest_path,
  manifest_sha256}}` dict covering every stage's own ADR-003
  `manifest.json` — `train`, `samples`, and `eval` always present, `uturn`
  present iff the arm ran the optional stage;
- **science**: all four MMD comparisons (mixture-level and per-kernel-scale,
  biased and raw-unbiased — raw unbiased MMD² may be negative and is never
  clipped), nearest-training and pair-correlation summaries, optional U-turn
  reduced summary (§3), optimization step and examples seen,
  train/validation data hashes;
- **audit**: `git_commit` (the repo's `HEAD` SHA at run time, from the run
  manifest's own environment block; `None` only if git itself was
  unavailable), `run_dir`, `run_manifest_sha256`, `migration` (`None` for a
  record written at original completion; an explicit block, never a silent
  rewrite, for one built later on resume — §4).

A record with a science digest but no `resolved_config` would be
insufficient to ever reconstruct or re-audit the run it describes — this is
why `resolved_config`/`resolved_config_sha256` and the `artifacts` links are
required fields, not optional ones: `Phase4CRunRecord.__post_init__` raises
if any is missing.

**Hashes are re-verified, not merely stored.** `load_run_record` (used by
resume, `analysis.ingest.load_rows`, and `maskeddiffusion-p4c-analyze`)
recomputes every linked file's SHA-256 by default
(`verify_hashes=True`, `experiments.schema.verify_artifact_hashes`) —
`resolved_config`, every stage's `manifest.json` (including `uturn`, when
present), the checkpoint file, and `run_manifest.json` itself — and raises
on any mismatch or missing file. A stored hash nobody ever recomputes
detects nothing; `verify_hashes=False` exists only for inspecting a record
copied out from its run directory.

## 6. Execution

```bash
uv run maskeddiffusion-experiment --config configs/experiments/<file>.toml \
    --output <output_root> --device cpu            # add --dry-run first
```

`--dry-run` prints the exact run count, dimensions, seeds, identities,
projected sample counts, and paths, and writes nothing. `--plan-only`
writes the pre-run manifest without executing. Always dry-run and
timing-calibrate (a single pair at the target cell) before launching a
multi-repeat campaign; do not launch campaigns as a side effect of editing
or validation (CLAUDE.md forbidden actions).

Determinism and the pair-manifest disorder checks are guaranteed on CPU
only; `mps`/`cuda` runs must be labeled platform-dependent.

## 7. Campaign configs (`configs/experiments/`)

- `smoke_d8/` — D=8 CPU integration checks, one per intervention. Never
  scientifically interpretable.
- `campaign_v1/` — the proposed Phase 4C controlled campaign (3 repeats per
  experiment, `n_generate = 1000`, `n_true = 1000`, kernel scales
  λ ∈ {4, 8}):
  - `v_trainability` at D=32, γ=2, α ∈ {0.5, 1, 2, 4, 8};
  - `sampler_stochasticity` (sequential_random family) at D=32, γ=2,
    α ∈ {0.5, 2, 8};
  - `optimization_budget` (short=500 vs long=5000 steps) at D=32, γ=2,
    α ∈ {0.5, 2, 8};
  - `finite_d` with D ∈ {16, 32, 64} at α=2, γ ∈ {1, 2, 4}.

  Training defaults in these files (`max_steps=2000`, `batch_size=64`,
  `learning_rate=1e-2`) are engineering placeholders pending the timing
  calibration and convergence check of the Phase 4C plan — they are NOT a
  validated optimization protocol, and no `campaign_v1` config has been
  executed. Base seeds are distinct per experiment file so no two campaign
  experiments silently share disorder.
- `pilot/` — one small calibration run (not the campaign): the central
  `campaign_v1` cell (`v_trainability`, D=32, γ=2, α=2), 2 repeats, modest
  settings (`n_generate=200`, `max_steps=200`, `n_true=200`), U-turn enabled
  at 3 `t` values. Executed once on CPU to measure timing and storage; see
  `docs/PHASE4C_PILOT_REPORT.md` for the results and what they do (and do
  not) say about `campaign_v1`'s placeholder settings. Its own output is
  gitignored (`artifacts/pilot/`), not committed.

## 8. Claim discipline (binding)

- Results compare sampler-indexed terminal laws `P_{θ,A,k}` against the
  finite-`F` teacher law under specific diagnostics; never write "the model
  distribution" or "learns the distribution".
- Three repeats support individual points, paired differences, mean and
  sample standard deviation — no bootstrap intervals, no significance
  tests, no capacity-threshold or phase-transition language.
- `finite_d` comparisons are seed-value-matched, not disorder-identical;
  say so wherever they are reported.
- A smoke run is never scientific evidence.
