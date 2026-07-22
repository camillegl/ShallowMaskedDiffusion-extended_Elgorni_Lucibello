# Migration report — Phase 2 (2026-07-20)

**Phase 3 status: MERGED.** Every "isolated on `guthlac`" / "exists only on `guthlac`; main
is unaffected" statement in the Phase 3B-3G sections below described the state at the time
each phase's commit was made, before review. All of Phase 3 was merged into `main` via
[PR #2](https://github.com/camillegl/ShallowMaskedDiffusion-extended_Elgorni_Lucibello/pull/2)
(merge commit `c6a716f2e8915c7a01864d1658275f9305586f5`); `main` now has the CI workflow,
the retired `train.py`/dependencies, and every other Phase 3 change described here.

Starting point: branch `main`, commit `fea744d`. (Line references in the Phase-1 documents
`docs/UPSTREAM_DISCREPANCIES.md` / `docs/EQUATION_TO_CODE_MAP.md` cite the audit snapshot
`2e2db70`; `fea744d` only removed `rsync-logs.sh`, so those references remain valid.) Phase 2 replaced the active
implementation with the `src/maskeddiffusion` package while preserving legacy
behavior as fixtures and protecting the final-run MMD notebooks.

## What was built

- `src/maskeddiffusion/`: dimensions, hierarchical randomness, teacher,
  masking, linear score, objectives, samplers, direct training loop,
  checkpoints, artifact schema, metrics (MMD/correlations/overlaps), four
  CLIs. ADRs 001–004 record the framework decisions.
- `tests/`: 112 tests (unit, property, integration, regression) including
  legacy-behavior fixtures and MMD notebook-equivalence.
- `tests/fixtures/original_architecture_v1/`: five deterministic fixtures
  pinning legacy behavior (linear model, masking, objective, stochastic
  sampler trajectory, greedy reconstruction) + manifest with source commit
  `fea744d` and environment versions.
- Preservation: `docs/REFERENCE_RESULTS_MANIFEST.md`,
  `artifacts/reference/mmd_final_run/manifest.json`,
  `tests/regression/test_preservation.py`,
  `scripts/validate_reference_artifacts.sh`.

## Legacy modules: status

| Path | Status | Reason |
|---|---|---|
| `diffusion.py` | **kept** (superseded, frozen) | imported by the protected corrected notebook; source of the pinned fixtures |
| `models.py` | **kept** (superseded, frozen) | imported by `diffusion.py` |
| `datasets.py` | **kept** (superseded, frozen) | imported by the protected corrected notebook |
| `train.py` | **deleted on `guthlac`** (Phase 3F, see below) | legacy CLI for reproducing historical runs; retired once its historical Julia/notebook consumers were also retired |
| `experiments-analysis/utils.py`, `run_uturn_experiments.py` | **deleted on `guthlac`** (Phase 3D, see below) | historical (non-protected) notebook consumers were also retired |

No legacy module was deleted at Phase 3A: the protected notebooks'
importability outranks cleanup (stop-condition compliance). `utils.py` and
`run_uturn_experiments.py` were later retired on `guthlac` (not Phase 3A;
not `main`) once their only consumers (historical notebooks) were retired.

## Edits to legacy files (behavior-preserving, sanctioned)

- `diffusion.py`: removed the stray `from turtle import pd` (unused; dragged
  tkinter in at import). Closes part of D13.
- `train.py`: fixed the `rfs10-tanh` → `rfs10_tanh` help text (D10) and
  labeled `--alpha` with its legacy M/N meaning (D1 warning). No behavioral
  change.

## Deletions

**None.** Verified before deciding:

- `experiments.csv` (root) vs `experiments-analysis/experiments.csv`: SHA-256
  **differ** — not duplicates; canonical copy undetermined; deletion blocked.
- `experiments-analysis/results_mmd_distribution_distance.csv` vs
  `experiments-analysis/results/results_mmd_distribution_distance.csv`:
  SHA-256 **differ** — same conclusion.

## Deferred deletion candidates (need owner sign-off; scientific roles NOT
yet extracted)

- `notes/*.typ` + PDFs — still the only home of the upstream equations;
  must not be deleted before the manuscript exists.
- `src-hopfield/`, `data/*.npz`, `julia-code/` — self-contained side studies /
  reference implementations.
- Superseded notebooks (`analysis.ipynb`, `analysis-J.ipynb`,
  `analysis-mnist.ipynb`, `analysis-uturn.ipynb`,
  `analysis_loss_convergence.ipynb`,
  `analysis_mmd_distribution_distance.ipynb`) — historical evidence with
  embedded outputs.
- MNIST support (`BinarizedMNIST` in `datasets.py`, `torchvision`,
  `tensorboard`, `lightning`, `numba`, `tqdm`, `pandas`, `scipy`,
  `matplotlib`, `jupyterlab` deps) — removable only when the legacy modules
  and old notebooks retire.

## Phase 3A — committed at `ed42906cffd0b2b5989eb53e46f00ca6cdde4171` on `main`, pushed to `origin/main`

- Deleted: `scripts/train-cpu.sh`, `scripts/train-gpu.sh`,
  `scripts/train-cpu-mnist.sh`, `scripts/span-slurm.sh`, `scripts/uturn.sh`
  (the legacy cluster/local run scripts deferred above — `train.py` itself
  was **not** deleted, so this closes only the scripts half of that deferred
  candidate), `paper/main-neuralnetworks.typ` (manuscript stub, no
  equations), and `notes/notes_generalization_TODO.typ`.
- The upstream and hidden-manifold theory remain exactly where they were:
  `notes/notes_memorization.typ` (upstream) and `notes/notes_hiddenmanifold.typ`
  (hidden-manifold extension); neither was touched.
- New retirement documents: `docs/FROZEN_LEGACY_RUNTIME.md`,
  `docs/LEGACY_SCIENTIFIC_INDEX.md`, `docs/FINAL_REPOSITORY_MAP.md`.
- This phase performed no Hopfield, DMFT, Julia, notebook, or dependency
  retirement, and no scientific code or notation change.
- Status: **committed and pushed** to `main`/`origin/main`.

## Phase 3B — isolated on the `guthlac` branch (not merged to `main`)

- Two commits on `guthlac`, both created directly through the GitHub API and
  reviewed in a local checkout after the fact:
  - `513fe7755d9dd65718da6b7720264033a0fd61a3` ("docs: archive Hopfield and
    DMFT side study") — added `docs/archive/HOPFIELD_DMFT_ARCHIVE.md` and
    `docs/PHASE3_BRANCH_REPORT.md`, and updated `docs/FROZEN_LEGACY_RUNTIME.md`
    / `docs/FINAL_REPOSITORY_MAP.md` to reflect Phase 3A's committed status.
    No deletion in this commit.
  - `177fd8f84b0b02b799be057259ff74318c8761d7` ("chore: retire Hopfield and
    DMFT side study") — deleted the three `src-hopfield/*.py` files, both
    Hopfield/DMFT theory notes (`notes/notes_hopfield.typ`,
    `notes/notes_dmft_masked_hopfield.typ`), three `data/hopfield_*_N20000_*.npz`
    files, and four `notes/plots/hopfield_*.png` figures; added
    `docs/archive/HOPFIELD_DMFT_RETIREMENT.md` recording exactly what was
    removed and how to recover it.
- **Incomplete retirement — closed by follow-up commit.** The retirement
  commit above left several related tracked files behind on `guthlac` — two
  stale-parameter data files
  (`data/hopfield_T0_mcmc_N10000_S10_seed0.npz`,
  `data/hopfield_T001_mcmc_N10000_S10_seed0.npz`), six figures
  (`notes/plots/hopfield_T001_m_vs_t_mcmc_{pattern,random}.png`,
  `notes/plots/hopfield_T0_m_vs_t_mcmc_{pattern,random}.png`,
  `notes/plots/hopfield_T0_sweeps_mcmc_{pattern,random}.png` — two of the
  first pair were actually referenced by the now-deleted
  `notes/notes_hopfield.typ`, so they were orphaned), and the two compiled
  PDFs (`notes/notes_hopfield.pdf`, `notes/notes_dmft_masked_hopfield.pdf`)
  rendered from the deleted `.typ` sources. These ten paths were deleted by
  the follow-up commit "chore: complete Hopfield and DMFT retirement"; full
  inventory (size, blob, SHA-256, producer) in
  `docs/archive/HOPFIELD_DMFT_ARCHIVE.md`, removal record in
  `docs/archive/HOPFIELD_DMFT_RETIREMENT.md`.
- Phase 3B exists **only** on `guthlac`; `main` is unaffected and remains at
  Phase 3A (`ed42906cffd0b2b5989eb53e46f00ca6cdde4171`).
- This phase (plus its Hopfield/DMFT follow-up) performed no Julia, notebook,
  result, legacy-CLI, dependency, or CI retirement — those remain separate
  steps, addressed later in this same Phase 3 effort per
  `docs/PHASE3_BRANCH_REPORT.md`.

## Phase 3C — obsolete Julia implementations (isolated on `guthlac`)

- Commit "chore: retire obsolete Julia implementations": deleted all tracked
  files under `julia-code/SP/` (saddle-point/ODE theory for the uniform-data
  model, 14 files) and `julia-code/old/` (older Lux-based training
  implementation, 3 files). `julia-code/hiddenmanifold/` is unaffected.
- Full per-file inventory (size, blob, SHA-256, scientific/engineering role)
  recorded in `docs/archive/JULIA_LEGACY_ARCHIVE.md` before deletion.
- Phase 3C exists only on `guthlac`; `main` is unaffected.

## Phase 3D — historical analysis notebooks and utilities (isolated on `guthlac`)

- Commit "chore: retire historical analysis notebooks": deleted six
  non-protected exploratory notebooks (`analysis.ipynb`, `analysis-J.ipynb`,
  `analysis-mnist.ipynb`, `analysis-uturn.ipynb`,
  `analysis_loss_convergence.ipynb`, `analysis_mmd_distribution_distance.ipynb`)
  and their two shared utilities (`utils.py`, `run_uturn_experiments.py`).
  The two protected notebooks
  (`analysis_mmd_distribution_distance_corrected.ipynb`,
  `mmd_results_presentation_1.ipynb`) are untouched.
- Full per-file inventory (size, blob, SHA-256, purpose, consumers) recorded
  in `docs/archive/HISTORICAL_NOTEBOOKS_ARCHIVE.md` before deletion.
- Updated `docs/EQUATION_TO_CODE_MAP.md` and `docs/ORIGINAL_ARCHITECTURE.md`
  to drop the now-stale line-number citation of the deleted
  `run_uturn_experiments.py` driver script; the underlying mechanism it
  drove (`diffusion.py:119-126` `test_step`) is unaffected and unchanged.
- Phase 3D exists only on `guthlac`; `main` is unaffected.

## Phase 3E — superseded result artifacts and bibliography reconciliation (isolated on `guthlac`)

- Commit "chore: remove superseded result artifacts": deleted four orphaned
  result tables (`experiments.csv` at root and in `experiments-analysis/`,
  `experiments-analysis/results_mmd_distribution_distance.csv`, and
  `experiments-analysis/results/results_mmd_distribution_distance.csv`) whose
  sole consumers were the notebooks retired in Phase 3D, and three figures
  (`fig_mmd_vs_alpha_D_sweep.png`, `fig_mmd_vs_alpha_lambdas.png`,
  `fig_mmd_vs_alpha_multiscale.png`) confirmed produced only by the deleted
  uncorrected MMD notebook, not the protected corrected one.
- Reconciled `paper/bibliography.bib` against `notes/bibliography.bib`:
  every key in the paper copy was already present in the notes copy (no
  merge required), so the paper copy was deleted.
- Five figures and 18 `res-exp-*.csv` files were reviewed and **retained**
  because their producer/consumer status could not be established with
  confidence; `notes/plots/` and `paper/plots/` were retained wholesale as
  out of scope for this pass. Full accounting in
  `docs/archive/LEGACY_RESULTS_ARCHIVE.md`.
- No protected artifact (the three `results/*_corrected*.csv` files, the two
  protected notebooks) was touched.
- Phase 3E exists only on `guthlac`; `main` is unaffected.

## Phase 3F — legacy training CLI and dependency cleanup (isolated on `guthlac`)

- Verified (via `git grep`) that no retained script, notebook, test, or
  documentation instruction imports or executes `train.py`, once its
  historical Julia (Phase 3C) and notebook (Phase 3D) consumers were
  retired. Deleted `train.py` (174 lines, blob `4c59502752cd22f597694265c5fa0ac69a80cb14`,
  SHA-256 `278f504e85e1fd6576c2836da3f7e85cd191eaef4078159aa6fe9b9a21d3c5c`).
  `datasets.py`, `diffusion.py`, `models.py` remain frozen and unaffected —
  they are still required by the protected corrected notebook.
- `git grep` confirmed `scipy`/`numba` have no remaining importers (their
  sole prior consumer, the Hopfield/DMFT side study, was already retired)
  and `tensorboard` has no remaining importer (its sole consumer, `train.py`,
  was just deleted). Removed all three from `pyproject.toml`; regenerated
  `uv.lock` via `uv lock` (not hand-edited).
- Moved `jupyterlab` from core `dependencies` to a new `analysis`
  `dependency-groups` entry (`uv sync --group analysis`) — it is
  environment tooling to open notebooks, not an importable dependency of
  any frozen module, and is still needed to interactively run the two
  protected notebooks.
- Verified `uv sync --frozen` (core, no jupyterlab) and
  `uv sync --frozen --group analysis` (adds jupyterlab) both succeed against
  the regenerated lock.
- Phase 3F exists only on `guthlac`; `main` is unaffected and keeps
  `train.py`, `scipy`, `numba`, `tensorboard`, and core-`dependencies`
  `jupyterlab`.

## Phase 3G — CI and repository hardening (isolated on `guthlac`)

- Commit "ci: add repository validation workflow": added
  `.github/workflows/ci.yml` — triggers on PRs and pushes to `main`/`guthlac`,
  Ubuntu, Python 3.12 via `astral-sh/setup-uv`, `uv sync --frozen`, package
  import, ruff check/format, mypy, pytest, protected-artifact validation, the
  four CLI `--help` checks, and `git diff --check`. No experiment runs or
  notebook execution in CI.
- Added `tests/property/test_repository_hardening.py` with three static
  tests: (1) nothing under `src/maskeddiffusion/` imports the root frozen
  modules `datasets`/`diffusion`/`models`; (2) those three modules are not
  duplicated inside `src/maskeddiffusion/`; (3) every file in
  `artifacts/reference/mmd_final_run/manifest.json` exists and matches its
  pinned SHA-256 (same check as `scripts/validate_reference_artifacts.sh`,
  now also runnable under `pytest -q`). The bare-`alpha`/`gamma` enforcement
  requirement was already covered by the existing
  `tests/property/test_notation_enforcement.py::test_no_bare_alpha_identifiers_in_active_package`
  and was not duplicated.
- Test count: 125 → 128 (three new hardening tests; no existing test
  removed or modified).
- Phase 3G exists only on `guthlac`; `main` has neither the CI workflow nor
  the new hardening tests.

## Dependencies removed

- `hydra-core` (never imported anywhere — verified by grep over all .py and
  notebook sources).
- `torch-tb-profiler` (same verification).

Added: `numpy` (explicit), dev group `pytest`/`ruff`/`mypy`. Build system:
hatchling with `src/` layout and console-script entry points.

## Compatibility surfaces left in place

- `LinearScoreConfig(normalization="legacy_init_only", diagonal_policy="free")`
  — the named legacy-compat mode used by regression fixtures (ADR 004).
- The legacy flat modules themselves (table above). No new shims were needed:
  nothing in the new package imports legacy code and vice versa.

## Discrepancy resolutions (see docs/UPSTREAM_DISCREPANCIES.md)

- Fixed **in the active path**: D3 (runtime `1/√N` explicit; legacy convention
  survives as named compat mode), D4 (separate value/mask tensors), D6
  (regularization excludes frozen params), D8 (`sign(0):=+1` in the teacher).
- Fixed in legacy text: D10 (help string), part of D13 (turtle import,
  README/AGENTS.md staleness, `.python-version` and `uv.lock` now real and
  tracked).
- Still open (scientific): D7 (V≡0 under fixed F — the model now supports the
  trainable-V ablation), D11 (joint-law consistency), D12 documentation
  status unchanged. D1 remains open-by-design in the legacy CLI (never
  reinterpreted; active CLI rejects bare `alpha`).

## Phase 4A/4B — experimental-design audit and provenance hardening (isolated on `guthlac`)

- Phase 4A: read-only audit confirming the teacher/seed/dimension invariants
  hold in the active package at smoke scale; identified gaps (Model-vs-Train
  not wired into evaluation, no U-turn implementation, no automated
  convergence criterion) — no code changed.
- Phase 4B: strict scientific config validation; device-aware RNG for
  samplers/validation; evaluation provenance hardening (sample-artifact
  validation, checkpoint-recorded training config reconstruction, semantic
  checkpoint identity plus raw file SHA-256, Model-vs-Train wired in);
  continuous-time objective correctness (exact `t=0`); sampler provenance
  (`tokens_per_step`/temperature rejection); CI and artifact hardening
  (exact protected-path assertion, committed-diff whitespace check).
- Test count: 128 → 200 (docs/PHASE4A_DESIGN_AUDIT.md, PR #3).

## Phase 4C — paired-experiment engine, canonical run-record schema, analysis pipeline (isolated on `guthlac`)

Built across several sessions; delivered as one working tree, not yet
committed at the time of this report. Test count: 200 → 243 (U-turn
subsystem) → 365 (engine, schema, analysis, and their polish pass below).

- **U-turn / reconstruction subsystem** (`src/maskeddiffusion/uturn.py`,
  `cli/uturn.py`): partial masking at controlled `t`, paired mask/order/token
  seeds across train/fresh sources, `q_U(t)` retrieval overlap against the
  `1-t` no-recovery baseline, train-minus-fresh memorization-sensitive
  comparison emitted only when both sources are run.
- **Canonical run-record schema** (`src/maskeddiffusion/experiments/schema.py`,
  package name deliberately plural, distinct from the singular
  `experiment/` engine package): `Phase4CRunRecord`
  (`maskeddiffusion.p4c_run_record.v1`) — the single artifact-level contract
  between the engine and the analysis layer; validated on construction and
  on every load, tamper-detects a mismatched `model_config_digest`.
- **Paired-experiment engine** (`src/maskeddiffusion/experiments/`,
  `cli/experiment.py`, entry point `maskeddiffusion-experiment`): TOML → one
  `ExperimentSpec` per (repeat, condition); four registered interventions
  (`v_trainability`, `sampler_stochasticity`, `optimization_budget`,
  `finite_d`); strict paired-disorder validation (`experiments.pairs`) with a
  `finite_d` exemption for content-identical disorder (seed-value-matched
  only, by construction); byte-stable resumable execution
  (train → sample → evaluate → optional uturn, each an ADR-003 artifact);
  `run_record.json` written last and re-validated (not re-executed) on
  resume.
- **Analysis pipeline** (`src/maskeddiffusion/analysis/`,
  `cli/analyze.py`, entry point `maskeddiffusion-p4c-analyze`): tidy-row
  contract and structural validation (`analysis.rows`, spec §9's eight
  rejection rules), the `run_record.json → AnalysisRow` parser
  (`analysis.ingest`, the only artifact-reading code in the layer),
  statistics (mean/sample-std/median, no bootstrap, no significance tests),
  report/figure/provenance writers, and a wording guard now covering both
  every generated figure's text and every string value in the report JSON
  recursively (`analysis.report.check_report_wording`) — not a field
  denylist, so a new report field cannot silently bypass it.
- **Supporting refactors**: `cli/evaluate.py` extracted a reusable
  `evaluate_run(...)` so the engine's evaluate stage inherits the CLI's full
  checkpoint/teacher/sample provenance verification rather than duplicating
  it; `training.py` added `optimizer_identity(...)` for spec/manifest
  identities.
- **Campaign configs** (`configs/experiments/`): `smoke_d8/` (D=8 CPU
  integration checks, one per intervention plus a U-turn-enabled variant,
  never scientifically interpretable) and `campaign_v1/` (93 planned runs
  across all four interventions at a D=32 aspect-ratio-2 sample-ratio sweep
  plus a finite-D sweep; configs only, dry-run validated — the campaign has
  **not** been executed).
- **Docs**: `docs/PHASE4C_EXPERIMENT_PROTOCOL.md` (engine contract:
  intervention registry, optional U-turn stage and its reduced-summary
  design decision, run layout/resume discipline, canonical run record,
  execution, campaign configs, claim discipline) and
  `docs/PHASE4C_ANALYSIS_SPEC.md` (analysis contract: tidy-table schemas,
  validation rules, statistics policy, output formats, provenance manifest;
  status upgraded from "partial, schema-dependent parts deferred" to
  "active" once the schema landed).
- No scientific claim, equation, or notation changed;
  `docs/ORIGINAL_ARCHITECTURE.md` and `docs/EQUATION_TO_CODE_MAP.md` are
  unaffected and were cross-checked as part of this phase's completion
  requirements. `julia-code/SP/data/`/`plots/` — pre-existing, always
  gitignored local scratch output uncovered when the Phase 3 retirement
  deleted its own `.gitignore` — was re-ignored (root `.gitignore`) with a
  provenance note in `docs/archive/JULIA_LEGACY_ARCHIVE.md`; no file under
  those two paths was touched.

### Phase 4C closure/reconciliation pass (same branch, before first commit)

A structural reconciliation pass across the above, requested before any
Phase 4C code was committed. Test count: 365 → 490.

- **Package unification**: the singular `src/maskeddiffusion/experiment/`
  and plural `src/maskeddiffusion/experiments/` (schema-only) packages were
  merged into one `experiments/` package — engine (`spec`, `interventions`,
  `pairs`, `plan`, `runner`, `uturn_stage`) plus `schema` as siblings. All
  imports, entry points, tests (renamed `test_experiment_*` →
  `test_experiments_*`), and docs updated; a closure test
  (`tests/property/test_phase4c_closure.py`) asserts the singular package
  can never reappear and no file imports it.
- **`comparison_type` replaces the `strict_disorder` boolean**
  (`experiments.interventions.Intervention.comparison_type`,
  `"paired_disorder"` vs `"matched_seed_finite_size"`): `finite_d`'s
  different-`latent_dim` arms are a different-shape, distinct-`teacher_id`
  teacher by construction, never a same-disorder pairing. The engine's pair
  manifest, `analysis.rows`'s pairing validation (a new
  `finite_size_teacher_collision` rule requires distinct `teacher_id`, not
  merely tolerating it), and `analysis.statistics` (which now excludes
  `matched_seed_finite_size` rows from `paired_differences` by default and
  routes them to a new, explicitly labeled
  `p4c_matched_seed_finite_size.csv` / `matched_seed_finite_size_frame`)
  all key off this one field. A real correctness bug this surfaced and
  fixed in passing: `finite_d`'s legitimate `model_config_digest` variation
  (from `visible_dim` differing by construction) was not registered in
  `analysis.rows.INTERVENTION_VARYING_FIELDS`, so every real `finite_d`
  run's rows were being rejected as `identity_mismatch`.
- **Complete run provenance**: `Phase4CRunRecord` gained `status`,
  `resolved_config` + `resolved_config_sha256` (the canonical, reconstructible
  resolved config, not just a science digest), `artifacts` (a
  `{stage: {manifest_path, manifest_sha256}}` link for train/samples/eval
  and optional uturn), and `git_commit`. Hashes are now re-verified, not
  just stored: `load_run_record`'s new `verify_artifact_hashes` recomputes
  every linked file's SHA-256 by default and raises on any mismatch or
  missing file.
- **Explicit resume migration, true immutability**: a completed run's
  `run_record.json`, once written, is only ever re-validated on resume,
  never rebuilt or rewritten (removing a latent bug where the prior
  "rebuild and compare" resume logic would have broken on any
  non-deterministic field). A record missing entirely (pre-dates the
  schema, or was lost) is built exactly once with an explicit `migration`
  block (`experiments.schema.backfill_migration_block`), never silently.
- **Pilot config**: `configs/experiments/pilot/` — one small, actually-executed
  calibration run (not `campaign_v1`) at the central design cell, with
  timing/storage results in `docs/PHASE4C_PILOT_REPORT.md`. `campaign_v1`
  remains an unexecuted plan.
- **Closure tests**: `tests/property/test_phase4c_closure.py` plus
  additions to the schema/ingest/runner test suites cover: no import of the
  old singular package; the schema round-trip through
  engine → record → analysis; every artifact hash mismatch is rejected
  (`tests/unit/test_run_record_hash_verification.py`, real files, real
  tampering); a backfilled record is immutable on the next resume; explicit
  migration metadata is recorded; `finite_d` is never treated as
  `paired_disorder`; the U-turn artifact's hash is checked like any other
  stage's; the wording guard traverses nested report JSON; and analysis
  rejects missing/mixed pair members.
- No scientific claim, equation, or notation changed in this pass either.
