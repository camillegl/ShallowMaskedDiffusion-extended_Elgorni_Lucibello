# Frozen legacy runtime

Status as of Phase 3A (2026-07-21), uncommitted until this change is actually
committed. This document records the operating rules for the three legacy flat
modules that remain in the repository root: `datasets.py`, `diffusion.py`, and
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
  The Python dependencies these modules require at import/runtime (e.g. the
  MNIST-support stack noted in `docs/MIGRATION_REPORT.md` — `torchvision`,
  `tensorboard`, `lightning`, `numba`, `tqdm`, `pandas`, `scipy`, `matplotlib`,
  `jupyterlab`) must stay in `pyproject.toml` / `uv.lock` for as long as these
  modules are kept importable. Removing a dependency they need would silently
  break the protected notebook's re-runnability.

## Verification

Protected-file byte-exactness is enforced by
`scripts/validate_reference_artifacts.sh` and
`tests/regression/test_preservation.py`, both checked against
`artifacts/reference/mmd_final_run/manifest.json`. See
`docs/REFERENCE_RESULTS_MANIFEST.md` for the full hash table and known
limitations (e.g. the notebook's recorded kernel used a different Python
version than the current pinned environment).

## Retirement

These modules are not scheduled for deletion in Phase 3A. Their eventual
retirement (removal from the repository root, or extraction of their behavior
into `src/maskeddiffusion/` with an updated protected-notebook dependency) is
out of scope here; see `docs/LEGACY_SCIENTIFIC_INDEX.md` and
`docs/FINAL_REPOSITORY_MAP.md` for how they are expected to be treated in the
post-retirement repository shape.
