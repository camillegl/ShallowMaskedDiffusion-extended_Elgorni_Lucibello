# Phase 4A: experimental-design audit and minimal smoke reproduction

Branch: `guthlac` (now identical to `main` at `c6a716f2e8915c7a01864d1658275f9305586f55`).
Scope: audit `src/maskeddiffusion/`'s experimental design against
`docs/RESEARCH_SPEC.md`, and reproduce the protected empirical setup at
smoke scale. **No large experiment grid was run.** No protected notebook or
result was modified, rerun, or read for anything beyond what was already on
disk.

## 1. Contract audit (item 2 of the Phase 4 request)

Checked by direct code reading, not by running experiments.

| Contract requirement (`docs/RESEARCH_SPEC.md`) | Verified in | Status |
|---|---|---|
| `F` sampled once per repeat, held fixed, reused for every split | `src/maskeddiffusion/teacher.py:26-50` — `HiddenManifoldTeacher.sample` is the only place `F` is drawn; the instance is passed to every split | ✅ matches |
| `F_ia ~ N(0, 1/D)`, `x = sign(Fz)`, `sign(0) := +1` | `teacher.py:20-22` (`sign_pm1`) and `:44-49` (`F = randn(...) / sqrt(latent_dim)`) | ✅ matches, and fixes discrepancy D8 (the legacy `datasets.py` does not patch `sign(0)`; the active package does) |
| Independent latent streams per split (train/validation/evaluation) | `src/maskeddiffusion/randomness.py:15-30` — `train_data_seed`, `validation_data_seed`, `evaluation_data_seed`, `metric_seed` are four distinct, deterministically-derived streams, separate from `teacher_seed` | ✅ matches |
| `D`, `N/D = aspect_ratio`, `M/D = sample_ratio` as primary controls; `visible_load = M/N` derived only | `src/maskeddiffusion/dimensions.py:19-58` — `Dimensions.resolve` takes `latent_dim, aspect_ratio, sample_ratio`; `visible_load` is computed, never accepted as an argument | ✅ matches |
| Deterministic repeat seeds | `randomness.py:44-49` — `SeedHierarchy.from_base(base_seed)` derives every stream by fixed offset; a manifest records the full hierarchy (see §2 below) | ✅ matches |
| `V ≡ 0` is a modeling choice, not derived, and should be recorded as such | `src/maskeddiffusion/models.py:38-39,47,73-80` — `v_policy: Literal["frozen_zero", "trainable"]`, config-selectable, recorded in the manifest | ✅ matches; also see §3 |

No contract violation found. This confirms the earlier `scientific-auditor`
review (Phase 3) generalizes: the active package's experimental machinery
is built correctly to the spec, independent of the retirement work.

## 2. Minimal smoke reproduction (item 1 of the Phase 4 request)

Ran `scripts/reproduce_smoke.sh`'s four steps (train → sample → evaluate →
validate-artifact) against `configs/smoke/smoke.toml`
(`latent_dim=8, aspect_ratio=4.0, sample_ratio=6.0`, 60 training steps, CPU).
Output written to gitignored `artifacts/smoke-run*/` (not committed, not
manifest-pinned, purely local).

- `maskeddiffusion-train`: completed, wrote `teacher.pt`, `checkpoints/`,
  `manifest.json`, `metrics.jsonl` (per-step `train_loss`,
  `validation_loss`, `order_qW`, `order_qV`).
- `maskeddiffusion-sample`: 8 samples generated with the default
  `sequential_random_stochastic` (fair, k=1) sampler.
- `maskeddiffusion-evaluate`: computed `model_vs_true`, `true_vs_true`,
  `train_vs_true`, `nearest_training`, `pair_correlation_error` — see §4.
- `maskeddiffusion-validate-artifact`: both the train and sample artifacts
  validate against their own manifests.

The manifest (`artifacts/smoke-run/manifest.json`) records `git_sha`
(`c6a716f2...`), the full ten-stream seed hierarchy, sampler identity,
objective config, and package/torch/platform versions — the provenance
infrastructure required by `.claude/rules/testing-and-reproducibility.md`
is present and working at this scale.

**Local-environment note**: reproducing this required manually reapplying
the documented `chflags nohidden` workaround before each of the four CLI
invocations (`docs/LOCAL_ENVIRONMENT_TROUBLESHOOTING.md`) — this is a known,
already-documented, machine-local issue, not a package defect.

No scientifically interpretable claim follows from this run (`latent_dim=8`
is far too small; this is an integration check only, per
`configs/smoke/smoke.toml`'s own header comment).

## 3. Open-question readiness (item 3 of the Phase 4 request)

This section reports what infrastructure exists to investigate each open
question, not new experimental findings — none was run beyond the smoke
config above.

**Q1 — Is `V ≡ 0` materially restrictive?** Already investigable with no new
code: `v_policy = "trainable"` is a first-class config option
(`models.py:38`), recorded in every manifest. A Phase 4B comparison run
(`frozen_zero` vs `trainable`, same seeds/dimensions otherwise) would
directly test this.

**Q2 — Source of the persistent MMD gap (finite-size / optimization /
sampler / capacity)?** Partial infrastructure exists, partial does not:
- *Finite-size*: directly testable via a `D` sweep at fixed `aspect_ratio`,
  `sample_ratio` — the config surface already supports this.
- *Sampler mismatch*: directly testable — `samplers.py` defines seven named
  sampler identities (`sequential_random_stochastic/greedy`,
  `parallel_random_stochastic/greedy`, `one_shot_stochastic/greedy`,
  `sequential_confidence_greedy`), each a distinct `P_{θ,A,k}`; running
  `evaluate` against samples from each is already supported by the CLI.
- *Optimization*: `metrics.jsonl` logs `train_loss`/`validation_loss` per
  step, enough to check convergence, but there is no automated "did this
  run converge" check yet — would need a threshold/plateau criterion added
  for Phase 4B, not present today.
- *Capacity*: no direct diagnostic beyond varying `sample_ratio` and reading
  the resulting gap; this is the hardest of the four to isolate and is
  correctly flagged `[CONJ]`/open in `docs/RESEARCH_SPEC.md`.

**Q3 — Does U-turn retrieval work on hidden-manifold data?** **Not currently
investigable.** `git grep -in "uturn\|u_turn\|retrieval"` across
`src/maskeddiffusion/` returns nothing — the U-turn mechanism
(`test_step`/`mask_and_sample` in the frozen legacy `diffusion.py`, itself
just corrected for a naming error in `docs/UPSTREAM_DISCREPANCIES.md` D14)
has never been ported to the active package, and the legacy path only ever
ran against uniform-data checkpoints. Investigating Q3 requires **new
code** — porting or reimplementing a U-turn-style retrieval diagnostic
against `HiddenManifoldTeacher` — before any Phase 4B run could touch it.
This is the one open question this audit cannot mark "ready."

## 4. Diagnostic coverage audit (item 4 of the Phase 4 request)

Requested: True–True, Train–True, Model–Train, Model–True.

- `src/maskeddiffusion/metrics/mmd.py` defines all **four**:
  `true_vs_true` (:125), `train_vs_true` (:135), `model_vs_true` (:145),
  **and `model_vs_train`** (:155).
- `src/maskeddiffusion/cli/evaluate.py`, however, only wires **three** of
  the four into its `results` dict (`:79-81`): `model_vs_true`,
  `true_vs_true`, `train_vs_true`. `model_vs_train` is defined in the
  metrics module but never called from the CLI, and is absent from both
  `summary.json` and the printed stdout output (which itself only prints a
  further-reduced `{model_vs_true, true_vs_true}` subset of the three it
  does compute — `train_vs_true` is written to `summary.json` but not
  printed).

**Concrete gap for Phase 4B**: `maskeddiffusion-evaluate` needs one small,
well-scoped change — call the existing `model_vs_train` function and add it
to `results` — before the full four-way comparison in item 4 can be
produced by the CLI as-is. `git grep -n "model_vs_train" tests/` returns
**zero results**: the function is defined but has no test coverage at all
today. This is a minimal, additive change in scope (no new metric math to
write, `model_vs_train` already exists), but Phase 4B must add a unit test
for it alongside the CLI wiring — it should not be wired into a
scientific-diagnostic CLI untested.

## Recommendation for Phase 4B

Phase 4B should be scoped narrowly, in this order:

1. Wire `model_vs_train` into `maskeddiffusion-evaluate` (small code change
   + test), closing the item-4 gap identified in §4.
2. Decide how to investigate Q3 (U-turn on hidden-manifold data) — this
   needs a design decision (port `mask_and_sample`-equivalent logic, or a
   new implementation) before any code is written, given it doesn't exist
   today.
3. Only then run small, explicit, single-purpose comparison experiments
   (not a grid) — e.g. one `v_policy` A/B pair for Q1, one sampler-identity
   comparison for Q2's sampler-mismatch hypothesis — each saved as a
   manifest-based artifact under a scratch/gitignored path, never
   interpreted beyond what the diagnostic actually measures, per
   `docs/RESEARCH_SPEC.md`'s permitted-claims list.
4. The full `(D, sample_ratio, aspect_ratio)` grid (item 5) should wait
   until 1-3 are resolved and reviewed — running it earlier risks
   generating results under an incomplete diagnostic (missing
   Model–Train) or against an unresolved U-turn question.

No code was changed in this session; this document is read-only analysis
plus one gitignored smoke run.
