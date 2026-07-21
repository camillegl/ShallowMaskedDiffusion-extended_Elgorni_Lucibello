# Frozen legacy runtime

**Phase 3 status: MERGED.** References to `guthlac` below (`train.py` retirement,
`scipy`/`numba`/`tensorboard` removal) describe Phase 3 work that was subsequently merged
into `main` via [PR #2](https://github.com/camillegl/ShallowMaskedDiffusion-extended_Elgorni_Lucibello/pull/2)
(merge commit `c6a716f2e8915c7a01864d1658275f9305586f5`) — those changes are on `main` too.

Status as of Phase 3A, committed and pushed at
`ed42906cffd0b2b5989eb53e46f00ca6cdde4171` on `main`/`origin/main`. This
document records the operating rules for the three legacy flat modules that
remain in the repository root: `datasets.py`, `diffusion.py`, and
`models.py`.

## What these files are

`datasets.py`, `diffusion.py`, and `models.py` are **frozen compatibility
files**. They are the pre-migration ("legacy") implementation, superseded by
`src/maskeddiffusion/` (see `docs/MIGRATION_REPORT.md`). They are kept solely
because the protected corrected MMD notebook,
[`experiments-analysis/analysis_mmd_distribution_distance_corrected.ipynb`](../experiments-analysis/analysis_mmd_distribution_distance_corrected.ipynb),
imports `datasets` and `diffusion` from the repository root, and `diffusion.py`
in turn imports `models.py`. As long as that notebook is preserved and
required to remain re-runnable, these three modules must remain importable
from the repository root.

`train.py` was a separate, non-protected legacy CLI. It was never
hash-pinned or listed in `artifacts/reference/mmd_final_run/manifest.json`,
and this document made no protection claim about it. It was retired
(deleted) on the `guthlac` branch once its historical consumers (the old
Julia scripts and superseded historical notebooks) were also retired; it
remains present on `main`. See `docs/archive/JULIA_LEGACY_ARCHIVE.md` and
`docs/archive/HISTORICAL_NOTEBOOKS_ARCHIVE.md`.

## Rules

- **Active code must never import them.** Nothing under `src/maskeddiffusion/`
  may import `datasets`, `diffusion`, or `models` from the repository root, and
  these modules must never import from `src/maskeddiffusion/`. This is a
  one-way legacy-to-notebook dependency only.
- **No new features or refactors.** These modules are frozen at their current
  (Phase 2, behavior-preserving) content. Do not add functionality, rename
  symbols, restructure, or "clean up" code in them.
- **Changes require deliberate manifest and preservation-test updates.** Any
  intentional edit to `datasets.py`, `diffusion.py`, or `models.py` must, in
  the same change: update the SHA-256 entries in
  `artifacts/reference/mmd_final_run/manifest.json`, update
  `docs/REFERENCE_RESULTS_MANIFEST.md`, and confirm
  `tests/regression/test_preservation.py` and
  `scripts/validate_reference_artifacts.sh` still pass against the new
  content. An accidental or drive-by edit is a regression, not a maintenance
  action.
- **Retained dependencies must not be removed while these modules remain.**
  Verified by direct import tracing (not by `pyproject.toml` comments), the
  three frozen root modules require, at import/runtime:
  - `datasets.py`: `torch`, `torchvision` (MNIST support).
  - `diffusion.py`: `torch`, `pytorch_lightning` (`lightning`), `tqdm`.
  - `models.py`: `torch` only.
  These — `torch`, `torchvision`, `lightning`, `tqdm` — must stay in
  `pyproject.toml` / `uv.lock` for as long as these modules are kept
  importable. Removing one of them would silently break the protected
  notebook's re-runnability.

## Dependency boundaries

Import tracing (grep over all tracked `.py` files and notebook source, not
`pyproject.toml` comments) separates three categories:

- **Frozen root imports** (required directly by `datasets.py`/`diffusion.py`/
  `models.py`, see above): `torch`, `lightning` (`pytorch_lightning`),
  `torchvision`, `tqdm`.
- **Protected/historical analysis** (required by the protected notebooks
  and/or retained historical analysis material, not by the frozen root
  modules themselves): `numpy`, `pandas`, `matplotlib`.
- **Removed (Phase 6, `guthlac` only)**: `scipy` and `numba` were used
  exclusively by the Hopfield/DMFT side study (`scipy` by
  `src-hopfield/hopfield_saddle_point.py`, `numba` by
  `src-hopfield/mcmc_hopfield.py` — see `docs/archive/HOPFIELD_DMFT_ARCHIVE.md`),
  which was retired on `guthlac` in an earlier commit; a repository-wide
  `git grep` confirmed no remaining `.py`/`.ipynb` source imports either
  package. `tensorboard` was imported only by the now-deleted `train.py`
  (`pytorch_lightning.loggers.TensorBoardLogger`); a `git grep` confirmed no
  remaining consumer. All three were removed from `pyproject.toml` and
  `uv.lock` in "chore: retire legacy training CLI and unused dependencies".
  All three remain present on `main`, which is unaffected.
- **Moved to an optional dependency group**: `jupyterlab` is not an
  importable library dependency of any frozen module or the active
  package — it is the environment needed to open/run any notebook
  (protected or historical). It was moved from core `dependencies` to the
  `analysis` dependency group (`uv sync --group analysis`) in the same
  commit; it was not deleted, since it is still required to interactively
  open the protected notebooks.

## Verification

Protected-file byte-exactness is enforced by
`scripts/validate_reference_artifacts.sh` and
`tests/regression/test_preservation.py`, both checked against
`artifacts/reference/mmd_final_run/manifest.json`. See
`docs/REFERENCE_RESULTS_MANIFEST.md` for the full hash table and known
limitations (e.g. the notebook's recorded kernel used a different Python
version than the current pinned environment).

## Retirement

These modules are not scheduled for deletion in Phase 3A or in the Phase 3B
Hopfield/DMFT retirement on `guthlac`. Their eventual retirement (removal
from the repository root, or extraction of their behavior into
`src/maskeddiffusion/` with an updated protected-notebook dependency) is out
of scope here and requires a separate decision, because the protected
corrected notebook depends on them; see `docs/LEGACY_SCIENTIFIC_INDEX.md` and
`docs/FINAL_REPOSITORY_MAP.md` for how they are expected to be treated in the
post-retirement repository shape.
