# Phase 3 completion report

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

76 files changed, 1090 insertions(+), 48902 deletions(-) (as of commit
`1d98ace`; this document's own commit adds a small further delta).

## Final remote verification

Not yet performed — this report is written before the final push. See the
session's final report for `git push` and remote-verification results.

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
