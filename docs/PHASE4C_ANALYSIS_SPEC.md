# Phase 4C analysis pipeline — reporting specification

Status: **active** — the schema commit (`maskeddiffusion/experiments/schema.py`,
`Phase4CRunRecord`) has landed, together with the parser
(`maskeddiffusion.analysis.ingest`) and the `maskeddiffusion-p4c-analyze` CLI;
§0's deferral list is resolved (see §10). This document specifies the Phase 4C
analysis layer: its tidy-table contracts, validation rules, statistics policy, output
formats (CSVs, report JSON, figures), and provenance manifest. Notation and claim
discipline follow `docs/NOTATION.md` and `docs/RESEARCH_SPEC.md`.

## 0. Session boundaries and dependencies

The multi-session plan gives the engine session (Session 1) the first commit:
`src/maskeddiffusion/experiments/schema.py` defining a validated `Phase4CRunRecord`
(one canonical `run_record.json` per completed run; U-turn data augments the same
record or a schema-defined linked record).

**This session (Session 4) does not ingest repository artifacts.** Until the schema
commit lands, this session delivers only:

1. this reporting specification;
2. the schema-independent downstream pipeline — tidy rows, validation, statistics,
   CSV/JSON/figure/provenance writers — developed and tested against **synthetic
   fixtures only**, built through the package's own row constructor
   (`maskeddiffusion.analysis.synthetic`), never handwritten approximations.

The following were deferred until `experiments/schema.py` landed and are now
implemented (see §10):

- the parser `run_record.json` → `AnalysisRow` (the only schema-dependent function);
- the `maskeddiffusion-p4c-analyze` CLI (its only real input mode is the parser);
- parser tests, which must serialize synthetic records through `Phase4CRunRecord`
  itself.

The tidy-row contract below is the parser's *target*; it is defined here, once, so
the engine schema and this pipeline cannot drift apart silently. It is derived 1:1
from the required-columns list of the Phase 4C analysis request. `AnalysisRow`
**must not** be extended into an alternative artifact format: artifact-level truth
lives only in `Phase4CRunRecord`.

## 1. Inputs

One `AnalysisRow` per completed, validated run record. A run record is one
(intervention, condition, repeat) evaluation of a sampler-indexed terminal law
`P_{θ,A,k}` against the finite-`F` teacher law `P_F` (quenched teacher per repeat).

Comparison-name mapping (engine/evaluate names → tidy column prefixes):

| engine name (`maskeddiffusion-evaluate`) | tidy prefix | meaning |
|---|---|---|
| `model_vs_true` | `model_true` | terminal-law samples vs fresh `P_F` |
| `true_vs_true` | `true_true` | two independent fresh `P_F` batches (finite-sample floor) |
| `train_vs_true` | `train_true` | empirical training rows vs fresh `P_F` |
| `model_vs_train` | `model_train` | terminal-law samples vs empirical training rows |

MMD values in the CSVs are the **uniform-weight mixture over the record's kernel
scales** (`mixture_biased_mmd2`, `mixture_unbiased_mmd2_raw` of
`maskeddiffusion.metrics.mmd`), matching the headline numbers of the evaluation
CLI. Per-kernel-scale values are preserved verbatim in the report JSON
(§6) and are available to figures; they are not flattened into CSV columns (kernel
sets may vary across campaigns; a fixed wide schema would silently break).

## 2. Tidy per-run CSV (`p4c_per_run.csv`)

One row per run record. Exact column order (`analysis.rows.PER_RUN_COLUMNS`):

- Identifiers: `experiment_id`, `pair_id`, `repeat_id`
- Dimensions: `latent_dim` (D), `visible_dim` (N), `train_size` (M),
  `aspect_ratio` (γ = N/D), `sample_ratio` (α = M/D), `visible_load` (M/N, derived)
- Design: `intervention`, `condition`
- Identity: `teacher_id`, `checkpoint_id`, `sampler_name`,
  `sampler_tokens_per_step`, `model_config_digest`
- Seeds (full ten-stream hierarchy + base): `base_seed`, `teacher_seed`,
  `train_data_seed`, `validation_data_seed`, `evaluation_data_seed`, `model_seed`,
  `mask_seed`, `dataloader_seed`, `sampler_order_seed`, `sampler_token_seed`,
  `metric_seed`
- MMD (mixture): `model_true_mmd2_biased`, `model_true_mmd2_unbiased_raw`,
  `true_true_mmd2_biased`, `true_true_mmd2_unbiased_raw`,
  `train_true_mmd2_biased`, `train_true_mmd2_unbiased_raw`,
  `model_train_mmd2_biased`, `model_train_mmd2_unbiased_raw`
- Overlap / correlation: `nearest_training_excess`, `nearest_training_model_mean`,
  `nearest_training_true_mean`, `pair_correlation_rms_error`,
  `pair_correlation_max_abs_error`
- Optional U-turn scalars: `uturn_baseline_recovery`, `uturn_excess_recovery`
  (empty when the run has no U-turn diagnostic; the `q_U(t)` curve itself lives
  only in the report JSON and U-turn figures)
- Optimization: `optimization_step`, `examples_seen`
- Provenance: `artifact_path`, `artifact_sha256`, `record_validated`

Unbiased MMD² values may legitimately be negative (finite-sample U-statistic,
`docs/RESEARCH_SPEC.md`); they are stored raw, never clipped.

## 3. Tidy paired-difference CSV (`p4c_paired_differences.csv`)

One row per `(pair_id, repeat_id)` with exactly two matched conditions,
restricted to `comparison_type == "paired_disorder"` rows
(`analysis.rows.comparison_type`) — same quenched `F`, same
train/validation/evaluation draws across arms
(`v_trainability`/`sampler_stochasticity`/`optimization_budget`).
`comparison_type == "matched_seed_finite_size"` rows (`finite_d`) are
EXCLUDED from this table by default
(`analysis.statistics.paired_differences`,
`include_matched_seed_finite_size=False`): a different `latent_dim` means a
different-shape teacher, so a B−A delta between two `finite_d` arms is not
a same-disorder comparison, and `finite_d` groups can also legitimately
hold more than the two arms this table assumes. See §3a.

- Keys/labels: `pair_id`, `repeat_id`, `intervention`, `condition_a`,
  `condition_b` (the two condition labels, sorted lexicographically — a
  deterministic, recorded convention, not a scientific ordering)
- Shared design: `latent_dim`, `visible_dim`, `train_size`, `aspect_ratio`,
  `sample_ratio`, `visible_load`, `teacher_id`
- Per metric `m` in `AGGREGATED_METRICS` (§5): `m_a`, `m_b`,
  `m_delta_b_minus_a` = `m_b − m_a`
- Context: `sampler_name_a`, `sampler_name_b`, `optimization_step_a`,
  `optimization_step_b`, `examples_seen_a`, `examples_seen_b`

## 3a. Matched-seed finite-size CSV (`p4c_matched_seed_finite_size.csv`)

The explicitly-labeled, NON-paired home for `comparison_type ==
"matched_seed_finite_size"` rows (`finite_d`), produced by
`analysis.statistics.matched_seed_finite_size_frame`. One row per RUN, not
per pair — distinct `latent_dim` arms of a sweep are never differenced
against each other. Columns: `comparison_type` (always
`"matched_seed_finite_size"`, so this table can never be mistaken for §3's
output even if concatenated with it), `pair_id`, `repeat_id`,
`intervention`, `condition`, `teacher_id` (distinct per row within a
sweep — never equal, per `pairs.py`/`rows.py` validation), `latent_dim`,
`aspect_ratio`, `sample_ratio`, `visible_dim`, `train_size`,
`visible_load`, and every metric in `AGGREGATED_METRICS` at its own raw
value (no delta, no aggregation). Any downstream analysis of a `finite_d`
sweep (e.g. plotting a metric against `latent_dim`) must read this table,
never §3's.

## 4. Aggregate CSV (`p4c_aggregate.csv`)

One row per design cell × condition, over **all individual repeats** (repeats are
never dropped or pre-averaged before this step):

- Keys: `intervention`, `condition`, `latent_dim`, `aspect_ratio`,
  `sample_ratio`, `visible_dim`, `train_size`, `n_repeats`
- Per metric `m`: `m_mean`, `m_std_sample` (sample standard deviation, ddof=1;
  empty when `n_repeats < 2`), `m_median` (supplementary)
- `uturn_baseline_recovery` / `uturn_excess_recovery` blocks are appended only
  when at least one row in the input carries U-turn data.

One row per design cell for the **paired differences**
(`p4c_aggregate_paired.csv`): keys `intervention`, `condition_a`, `condition_b`,
dimension columns, `n_pairs`, and per `m_delta_b_minus_a` the same
mean / sample-std / median triple.

## 5. Statistics policy (binding)

- All individual repeats are retained in every output (per-run CSV, report JSON,
  figures). Aggregation is additional, never a replacement.
- Paired A−B differences are computed within `(pair_id, repeat_id)` before any
  cross-repeat aggregation.
- Central tendency: **mean** with **sample standard deviation (ddof=1)**;
  **median** reported as supplementary only.
- **No bootstrap confidence intervals** at n=3 repeats.
- **No significance tests** at n=3 (no p-values, no "significant/insignificant"
  language — that would be test theatre).
- The True–True finite-sample floor is shown **directly** (per-repeat floor values
  in tables and figures), not via a fitted noise model.
- `AGGREGATED_METRICS` = the eight MMD² mixture columns (biased and raw unbiased
  for all four comparisons) + `nearest_training_excess` +
  `pair_correlation_rms_error`.

## 6. Machine-readable report JSON (`p4c_report.json`)

`schema_version: maskeddiffusion.p4c_analysis.v1`. Keys:

- `generated_at`, `provenance` (git sha/dirty flag, package/python versions,
  platform)
- `statistics_policy` — an explicit echo of §5 (machine-checkable:
  `bootstrap_ci: false`, `significance_tests: false`)
- `validation` — counts plus every rejection as `{rule, reason, experiment_ids}`
- `records` — one object per accepted run: every tidy field **plus**
  `mmd_per_lambda` (per-kernel-scale biased / raw-unbiased MMD² for all four
  comparisons) and, when present, `uturn` (`mask_densities`, `overlap` = the
  `q_U(t)` curve, `baseline_recovery`, `excess_recovery`)
- `paired_differences`, `aggregate`, `aggregate_paired`,
  `matched_seed_finite_size` — the exact row content of the four CSVs (§§2–3a)

## 7. Figures

Written under `figures/` as PDF + PNG (300 dpi), one file per
(intervention, design cell, metric) — **separate panels/files, never hidden
aggregation** across cells or interventions.

1. `p4c_fig_<intervention>_<cell>_<metric>_by_condition.*` — raw repeat points per
   condition; **paired conditions connected** by one line per `(pair_id,
   repeat_id)`; True–True floor shown directly (per-repeat floor markers plus a
   dashed median line). Log-y **only** for biased MMD² metrics with all values
   strictly positive (justification: MMD² is nonnegative and spans orders of
   magnitude); raw-unbiased MMD² (can be negative), excess overlaps, correlation
   errors, and all paired differences are always linear.
2. `p4c_fig_<intervention>_<metric>_paired_delta.*` — per-repeat
   `delta_b_minus_a` points, zero reference line, mean marker with **sample
   standard deviation** whiskers (not a confidence interval), linear y.
3. `p4c_fig_uturn_<pair>_<repeat>.*` — when U-turn data exist: `q_U(t)` per
   condition with baseline and excess recovery annotated.

Axis labels name the quantity and estimator (e.g. "Model–True MMD² (mixture,
biased V-statistic)"). Cell labels use realized integers
(`D=…, N=…, M=…`), never bare ratio symbols in code identifiers (ratio symbols
γ/α may appear in math text only, per `docs/NOTATION.md`).

**Wording policy**: no figure, table caption, or report string may contain
"converges"/"converged"/"convergence", "learns the distribution",
"phase transition", "the model distribution", or "exact ancestral sampling"
(`docs/RESEARCH_SPEC.md` prohibited-overclaims list). Permitted framings: "is
consistent with improved distributional agreement", "approaches the finite-F
target under this diagnostic". A wording guard scans every generated
figure's text (`analysis.plots.check_wording`/`check_figure_wording`) and
every string value in the report JSON, recursively
(`analysis.report.check_report_wording`) — not a denylist of "human-readable"
fields, so a new report field can never silently bypass it. The
`maskeddiffusion-p4c-analyze` CLI runs both checks on every invocation (not
only in tests) and exits nonzero, after writing its outputs, if either
finds a violation — the failure is never silent.

## 8. Provenance manifest (`p4c_analysis_manifest.json`)

`schema_version: maskeddiffusion.p4c_analysis_manifest.v1`. Contains: command,
UTC timestamp, environment metadata (git sha + dirty flag, python/torch/package
versions, platform, `uv_lock_sha256` — same fields as ADR 003 manifests), the
statistics policy echo, the validation report (accepted/rejected with reasons),
`inputs` (every consumed record: path + sha256 + role), and `outputs` (every
written file: path + sha256 + size + description). Filenames alone are not
provenance (`.claude/rules/testing-and-reproducibility.md`).

## 9. Validation and rejection rules (binding)

Rows are validated individually, then pairs are validated structurally. Every
rejection is recorded with rule, reason, and affected `experiment_id`s in the
report JSON and the manifest; rejected rows are excluded from all downstream
tables and figures. Rules:

1. `unvalidated_artifact` — a row whose source record failed artifact/schema
   validation (`record_validated` false) is rejected. (The parser sets this flag
   from `Phase4CRunRecord.validation_status`; it never launders an invalid
   artifact into a row.)
2. `duplicate_experiment_id` — duplicate `experiment_id`s: the first occurrence
   (lexicographic order) is kept, all later ones rejected.
3. `dimension_mismatch` — per row: `visible_dim == round(aspect_ratio *
   latent_dim)`, `train_size == round(sample_ratio * latent_dim)`, `visible_load
   == train_size / visible_dim` (`docs/NOTATION.md` invariants). Within a
   `(pair_id, repeat_id)`: all dimension columns must be identical — **except**
   for `comparison_type == "matched_seed_finite_size"` interventions
   (`analysis.rows.MATCHED_SEED_FINITE_SIZE_INTERVENTIONS`, initially
   `finite_d` only, whose arms differ in `latent_dim` by construction);
   there only `aspect_ratio` and `sample_ratio` must match. This
   comparison type is never disorder-identical and its rows are excluded
   from `paired_differences` by default (§3, §3a).
4. `mixed_teachers` — within a `(pair_id, repeat_id)`: `teacher_id` and
   `teacher_seed` must be identical (the two arms see the same quenched `F`).
   Teachers *differ across repeats* by design; that is not a violation. For
   `matched_seed_finite_size` interventions (`finite_d`), only `teacher_seed`
   equality is required — `teacher_id` equality is not merely unrequired but
   actively FORBIDDEN: a different-shape teacher must be a distinct draw, so
   equal `teacher_id`s there is rejected under a distinct rule,
   `finite_size_teacher_collision` (below), not silently accepted.
4a. `finite_size_teacher_collision` — within a `(pair_id, repeat_id)` of a
   `matched_seed_finite_size` intervention: `teacher_id` must be DISTINCT
   across arms (the positive counterpart of rule 4's requirement — a
   different `latent_dim` means a different-shape teacher, so identical
   `teacher_id` indicates the arms were not actually at different
   dimensions). This is a real-bug signal, not tolerated ambiguity.
5. `mixed_data_or_metric_seeds` — within a `(pair_id, repeat_id)`:
   `train_data_seed`, `validation_data_seed`, `evaluation_data_seed`, and
   `metric_seed` must be identical.
6. `unmatched_conditions` — a `(pair_id, repeat_id)` must contain exactly two
   rows with two distinct conditions; also, within a `pair_id`, `intervention`
   must be constant and each condition must map to exactly one sampler identity /
   model-config digest across repeats.
7. `identity_mismatch` — within a `(pair_id, repeat_id)`, `sampler_name`,
   `sampler_tokens_per_step`, and `model_config_digest` must be identical,
   **except** the fields the intervention legitimately varies, per the registry
   `analysis.rows.INTERVENTION_VARYING_FIELDS`, keyed by the engine's
   registered intervention names
   (`experiments.interventions.INTERVENTIONS`): `sampler_stochasticity` →
   `sampler_name`; `v_trainability` → `model_config_digest`;
   `optimization_budget` → no identity field, the difference shows only in
   `optimization_step`/`examples_seen`.
   Unknown interventions are strict (nothing may differ). Campaigns whose
   intervention varies a dimension (e.g. a `sample_ratio` sweep) are **not**
   paired comparisons; they are analyzed across cells, and pairing them is
   rejected by rule 3 — this is intended.
8. `missing_metric` / `nonfinite_metric` — all four MMD comparisons plus
   nearest-training and pair-correlation summaries must be present; biased MMD²
   and the overlap/correlation scalars must be finite (raw unbiased MMD² may be
   negative but not NaN/inf).

## 10. Deferred-work checklist (post-schema) — RESOLVED

`src/maskeddiffusion/experiments/schema.py` has landed. Status of the items:

1. `analysis/ingest.py` implemented: `Phase4CRunRecord`/`run_record.json` →
   `AnalysisRow` (the only artifact-reading code in the analysis layer). The
   tidy row's `experiment_id` is `{pair_id}-{condition}` — unique per run;
   the engine's own `experiment_id` names the whole campaign.
2. `maskeddiffusion-p4c-analyze` CLI added: records root → validated rows →
   all outputs of §§2–8 (wording guard enforced on every generated figure).
3. Parser tests serialize records **through `Phase4CRunRecord`**
   (`tests/unit/test_analysis_ingest.py`), never handwritten dicts.
4. The pipeline test suite and the wording guard run in CI via
   `uv run pytest`.
