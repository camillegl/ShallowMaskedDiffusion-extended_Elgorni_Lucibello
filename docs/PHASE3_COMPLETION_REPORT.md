# Phase 3 completion report

**Status: MERGED.** This report was written before the merge; every "review branch" /
"`main` is unchanged" statement below describes the state **at the time of writing**, prior
to review. Phase 3 was subsequently merged into `main` via
[PR #2](https://github.com/camillegl/ShallowMaskedDiffusion-extended_Elgorni_Lucibello/pull/2)
(merge commit `c6a716f2e8915c7a01864d1658275f9305586f5`), after CI caught and a follow-up
commit (`76e9fe5`) fixed one cross-platform test-tolerance issue — see the "Post-finalization
addendum" section below, added after that fix, before the merge.

Branch: `guthlac`. Baseline (`main`/`origin/main`): `ed42906cffd0b2b5989eb53e46f00ca6cdde4171`
(Phase 3A). This report covers the branch-local commits from `513fe77` through
`1d98ace` (this document's own finalization commit is separate and lands after it).

**`guthlac` remains a review branch. `main` is unchanged by any commit described
here.** Nothing in this report implies Carlo Lucibello, Filippo Elgorni, or any
other repository collaborator has approved, reviewed, or been notified of these
changes — that review step has not happened.

## Commits, in order

| Commit | Message |
|---|---|
| `513fe77` | docs: archive Hopfield and DMFT side study |
| `177fd8f` | chore: retire Hopfield and DMFT side study |
| `cdbacc6` | docs: reconcile phase 3 retirement records |
| `aa562c1` | chore: complete Hopfield and DMFT retirement |
| `09cb0a9` | docs: add residual-inventory detail dropped from prior Hopfield/DMFT commit |
| `031bb74` | chore: retire obsolete Julia implementations |
| `4f83041` | chore: retire historical analysis notebooks |
| `117a0d3` | chore: remove superseded result artifacts |
| `a893d55` | chore: retire legacy training CLI and unused dependencies |
| `1d98ace` | ci: add repository validation workflow |

`513fe77` and `177fd8f` predate this session (Phase 3B, prior work). `cdbacc6`
through `1d98ace` were performed in this session (Phase 3B1 reconciliation
through Phase 3G).

## Categories removed

1. **Hopfield/DMFT side study** — `src-hopfield/*.py` (3 files), both theory
   notes, all data/figures/PDFs including ten residual files missed by the
   first pass. See `docs/archive/HOPFIELD_DMFT_ARCHIVE.md`,
   `docs/archive/HOPFIELD_DMFT_RETIREMENT.md`.
2. **Obsolete Julia implementations** — `julia-code/SP/` (14 files,
   uniform-data saddle-point/ODE theory) and `julia-code/old/` (3 files,
   older Lux training implementation). `julia-code/hiddenmanifold/`
   untouched (verified empty diff, see below). See
   `docs/archive/JULIA_LEGACY_ARCHIVE.md`.
3. **Historical analysis notebooks and utilities** — six exploratory
   notebooks (`analysis.ipynb`, `analysis-J.ipynb`, `analysis-mnist.ipynb`,
   `analysis-uturn.ipynb`, `analysis_loss_convergence.ipynb`,
   `analysis_mmd_distribution_distance.ipynb`) and two utility scripts
   (`utils.py`, `run_uturn_experiments.py`). Both protected notebooks
   untouched. See `docs/archive/HISTORICAL_NOTEBOOKS_ARCHIVE.md`.
4. **Superseded result artifacts** — two orphaned run logs (`experiments.csv`
   root and `experiments-analysis/`), two orphaned pre-correction MMD result
   tables, three orphaned figures confirmed produced only by the deleted
   uncorrected notebook, and `paper/bibliography.bib` (reconciled against
   `notes/bibliography.bib`, all keys already present, no merge needed). Five
   figures and 18 `res-exp-*.csv` files were reviewed and retained as
   ambiguous rather than guessed at. See
   `docs/archive/LEGACY_RESULTS_ARCHIVE.md`.
5. **Legacy training CLI and dependencies** — `train.py` deleted once its
   historical consumers were retired; `scipy`, `numba`, `tensorboard` removed
   from `pyproject.toml`/`uv.lock` after confirming zero remaining importers;
   `jupyterlab` moved from core `dependencies` to an optional `analysis`
   dependency group. See `docs/MIGRATION_REPORT.md` Phase 3F.
6. **CI and hardening** — `.github/workflows/ci.yml` added; three new static
   tests in `tests/property/test_repository_hardening.py`. See
   `docs/MIGRATION_REPORT.md` Phase 3G.

Additionally, during this session's review pass, a pre-existing (not
introduced by this branch) documentation inaccuracy was found and fixed:
`docs/ORIGINAL_ARCHITECTURE.md`'s U-turn description named the wrong method
(`mask_and_sample` instead of `self.sample`). Logged and resolved as
`docs/UPSTREAM_DISCREPANCIES.md` D14; no code was changed.

## Retained scientific material (unaffected)

- `julia-code/hiddenmanifold/` (`SignChannel.jl`, `SignChannelRenyi.jl`, data,
  plots, scripts) — verified `git diff --exit-code main..guthlac --
  julia-code/hiddenmanifold` is empty.
- `notes/notes_memorization.typ`, `notes/notes_hiddenmanifold.typ`,
  `notes/bibliography.bib` — untouched.
- `notes/plots/**`, `paper/plots/**`, `experiments-analysis/res-exp-*.csv`
  (18 files) — reviewed, retained as out of scope / ambiguous rather than
  guessed at (see `docs/archive/LEGACY_RESULTS_ARCHIVE.md`).
- Five figures under `experiments-analysis/figures/` whose producer could
  not be confidently identified — retained (same document).

## Protected artifacts: status

All eight manifest-pinned files (`artifacts/reference/mmd_final_run/manifest.json`)
verified unchanged by `scripts/validate_reference_artifacts.sh` after every
phase and again in this final pass. Specifically verified empty diff against
`main`:

```
experiments-analysis/analysis_mmd_distribution_distance_corrected.ipynb
experiments-analysis/mmd_results_presentation_1.ipynb
experiments-analysis/results/results_mmd_distribution_distance_corrected.csv
experiments-analysis/results/results_mmd_time_sliced.csv
experiments-analysis/results/results_mmd_distribution_distance_corrected_10k.csv
datasets.py
diffusion.py
models.py
```

`artifacts/reference/mmd_final_run/manifest.json` itself: zero diff against
`main` (not modified at all in this branch).

## Dependency changes

Removed from `pyproject.toml` core `dependencies` (confirmed zero remaining
importers via `git grep` across all tracked `.py`/`.ipynb`/`.jl` sources):
`scipy`, `numba`, `tensorboard`. Moved from core `dependencies` to the new
`[dependency-groups].analysis` group: `jupyterlab`. `uv.lock` regenerated via
`uv lock` (not hand-edited); `uv sync --frozen` and
`uv sync --frozen --group analysis` both verified to succeed.

## CI workflow summary

`.github/workflows/ci.yml`: triggers on `pull_request` and `push` to
`main`/`guthlac`; Ubuntu runner; Python 3.12 via `astral-sh/setup-uv`;
`uv sync --frozen`; package import check; `ruff check`; `ruff format --check`;
`mypy src`; `pytest -q`; `scripts/validate_reference_artifacts.sh`; the four
CLI `--help` checks; `git diff --check`. No experiment runs or notebook
execution.

## Final test count

**128 passed** (was 125 at the start of this session; +3 from
`tests/property/test_repository_hardening.py`). No existing test removed or
modified.

## Validation results (this final pass)

- `uv run ruff check .` — all checks passed.
- `uv run ruff format --check .` — 40 files already formatted.
- `uv run mypy src` — no issues, 21 source files.
- `uv run pytest -q` — 128 passed.
- `./scripts/validate_reference_artifacts.sh` — all 8 protected reference
  files verified.
- `uv run maskeddiffusion-train/-sample/-evaluate/-validate-artifact --help`
  — all four succeed.
- `git diff --check` — clean (no whitespace/conflict markers).

## Reviewer findings (four independent read-only passes over `main..guthlac`)

- **architecture-auditor**: no structural problems found; confirmed no
  deleted file is still imported by `src/maskeddiffusion/`, CI, or tests.
  One stylistic (non-defect) observation about `pyproject.toml`'s
  `dependency-groups`/`optional-dependencies` mix was based on a stale
  reading — the `analysis` group is correctly under `[dependency-groups]`,
  verified directly against the file.
- **reproducibility-reviewer**: all checks passed (`uv.lock` consistency,
  protected-manifest untouched, frozen modules and protected
  notebooks/CSVs byte-identical, `scipy`/`numba`/`tensorboard` confirmed
  unused). Two non-blocking suggestions noted (no CI step cross-checking
  `.python-version` against `pyproject.toml`'s `requires-python`; `uv.lock`
  diff was spot-checked, not exhaustively line-by-line) — not acted on,
  recorded here for future reference.
- **scientific-auditor**: found one pre-existing (not introduced by this
  branch) documentation inaccuracy in `docs/ORIGINAL_ARCHITECTURE.md`'s
  U-turn description (named `mask_and_sample` instead of the actually-called
  `self.sample`) — logged as `docs/UPSTREAM_DISCREPANCIES.md` D14 and fixed
  in this finalization commit (documentation only; `diffusion.py` unchanged).
  No scientific-contract violation found; no orphaned citations from any
  deletion.
- **claim-reviewer**: no unsupported or overclaiming statements found;
  spot-checked "confirmed by grep" claims held up; no implication of author
  approval/review/notification found anywhere in the branch's prose.

## Retained tracked files considered but not deleted, with reasons

- `experiments-analysis/res-exp-*.csv` (18 files) — no consumer found by
  `git grep`, but not named as a candidate in the retirement instructions
  and may back committed figures under `notes/plots/`; flagged for a future,
  separately scoped review rather than deleted on ambiguous grounds.
- `notes/plots/**`, `paper/plots/**` — large committed figure sets tied to
  retained theory notes / the already-deleted `paper/main-neuralnetworks.typ`
  stub; determining per-file orphan status would require tracing every
  Typst image reference, judged out of proportion for this pass.
- Five `experiments-analysis/figures/*.png` (no-suffix variants of
  `fig_empirical_distribution_diagnostics`, `fig_nearest_train_overlap`,
  `fig_pairwise_correlation_error`, `fig_mmd_vs_alpha_gamma_sweep`,
  `fig_unbiased_mmd2_vs_alpha_multiscale`) — producer not confirmed in
  either the protected notebook's current `savefig` calls or the deleted
  uncorrected notebook's; retained rather than guessed at.

## `main..guthlac` diff stat

82 files changed, 1404 insertions(+), 48906 deletions(-) (as of commit
`76e9fe5`, including this document's own commit `d123427` and the
subsequent CI-tolerance fix commit `76e9fe5` described below).

## Post-finalization addendum: CI-exposed cross-platform test tolerance

Opening the pull request to `main` (https://github.com/camillegl/ShallowMaskedDiffusion-extended_Elgorni_Lucibello/pull/2)
triggered this repository's CI workflow for the first time ever. It failed
on `tests/regression/test_mmd_notebook_equivalence.py::test_matches_independent_notebook_fixture`
by ~4e-8 on the Linux runner — a pre-existing test, untouched by any commit
in this report, that had only ever run locally on macOS before. Root-caused
to float32 CPU reduction-order differences between macOS (arm64) and Linux
(x86_64), not a correctness regression: the fixture's `expected` values were
reproduced byte-for-byte by rerunning the independent, hash-verifying
`generate_fixture.py` script, and only the tolerance (`rel=1e-6, abs=1e-9`
→ `rel=5e-6, abs=1e-7`) was widened, with the rationale documented in
`docs/MMD_NOTEBOOK_PROVENANCE.md` and logged as
`docs/UPSTREAM_DISCREPANCIES.md` D15. Committed separately as `76e9fe5`
("test: make MMD equivalence tolerance cross-platform"). Both CI runs on
the PR now pass (128/128 tests). No protected artifact, notebook, or
scientific code was touched by this fix.

## Final remote verification

`git push origin guthlac` completed (fast-forward, no force) through commit
`76e9fe5`. `git ls-remote --heads origin guthlac` matches local HEAD;
`git ls-remote --heads origin main` remains
`ed42906cffd0b2b5989eb53e46f00ca6cdde4171`, unchanged.

## Final clean status

Working tree clean at the time of writing (only this file itself pending, to
be committed in the finalization commit) aside from an untracked local
`julia-code/SP/data/` and `julia-code/SP/plots/` directory tree containing
generated `.txt`/`.csv` outputs that were **never tracked by git** (ignored
via the now-deleted `julia-code/SP/.gitignore`) — left in place as
pre-existing local generated output, not part of any commit or the
`main..guthlac` diff.

## Remaining known scientific questions

- Whether the U-turn retrieval mechanism (`test_step`, `diffusion.py:119-126`)
  has ever been exercised against the hidden-manifold (fixed-`F`) teacher, as
  opposed to only legacy uniform-data checkpoints, is not established by
  this retirement work (see `docs/UPSTREAM_DISCREPANCIES.md` D14).
- The residual MMD gap above the noise floor remains an open empirical
  observation (unchanged by this branch — no scientific code or result was
  touched).
- `V ≡ 0` (frozen mask weights) remains an experimental restriction, not
  derived for hidden-manifold data (unchanged).

## Explicit statements

- **No scientific experiment, training run, MCMC, Julia solver, or notebook
  cell was executed** by any commit in this report. All notebook content was
  read, never run.
- **`guthlac` remains a review branch; `main` is unchanged.** Every deletion
  described here exists only on `guthlac` and remains fully recoverable from
  `phase2-hidden-manifold-foundation` or `ed42906cffd0b2b5989eb53e46f00ca6cdde4171`.
