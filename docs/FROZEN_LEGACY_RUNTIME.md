# Frozen legacy runtime

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

`train.py` is a separate, non-protected legacy CLI. It is **not** hash-pinned
or listed in `artifacts/reference/mmd_final_run/manifest.json`, and this
document makes no protection claim about it. `train.py` is a deprecated
reproduction tool for historical runs, not a frozen compatibility file.

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
- **Other historical material, not proven requirements of the frozen root
  modules**: `scipy` and `numba` are used exclusively by the Hopfield/DMFT
  side study (`scipy` by `src-hopfield/hopfield_saddle_point.py`, `numba` by
  `src-hopfield/mcmc_hopfield.py` — see `docs/archive/HOPFIELD_DMFT_ARCHIVE.md`;
  that side study is retired on the `guthlac` branch but still present on
  `main`); `tensorboard` is imported explicitly by `train.py`
  (`pytorch_lightning.loggers.TensorBoardLogger`) and by several historical,
  non-protected notebooks, not by the frozen root modules; `jupyterlab` is
  not an importable library dependency — it is the environment needed to
  open/run any notebook (protected or historical), retained for that reason.

**`scipy` and `numba` are dependency-cleanup *candidates*, not proven
removable.** Their only currently-known use is the Hopfield/DMFT side study,
but that conclusion rests on a grep of tracked files on the branch(es)
checked at the time of writing — a full repository-wide import and notebook
scan (across all branches intended to merge, and executed rather than
grepped for notebooks) must be completed before either dependency is
actually removed from `pyproject.toml`/`uv.lock`. No dependency is removed by
any Phase 3A or Phase 3B document or commit referenced here.

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
