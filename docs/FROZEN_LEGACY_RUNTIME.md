# Frozen legacy runtime

Status as of Phase 3A, committed and pushed at `ed42906cffd0b2b5989eb53e46f00ca6cdde4171`. This document records the operating rules for the three legacy flat modules that remain in the repository root: `datasets.py`, `diffusion.py`, and `models.py`.

## What these files are

`datasets.py`, `diffusion.py`, and `models.py` are **frozen compatibility files**. They are the pre-migration implementation, superseded by `src/maskeddiffusion/`. They are retained because the protected corrected MMD notebook imports `datasets` and `diffusion` from the repository root, and `diffusion.py` imports `models.py`.

`train.py` is a separate, non-protected legacy CLI. It is not hash-pinned and is not a protected-notebook dependency.

## Rules

- Active code under `src/maskeddiffusion/` must never import the frozen root modules.
- No new features, refactors, renames, or cleanup should be applied to them.
- Any intentional edit requires coordinated updates to the protected manifest, reference documentation, preservation tests, and validation script.
- Dependencies required for the frozen modules and protected notebooks must remain available until those compatibility surfaces are deliberately retired.

## Dependency boundaries

Import tracing separates three categories:

- **Frozen root imports:** `torch`, `lightning`, `torchvision`, and `tqdm` are required directly or transitively by the retained root compatibility modules.
- **Protected/historical analysis:** `numpy`, `pandas`, and `matplotlib` remain required by retained notebook and analysis material.
- **Other historical material:** `scipy` and `numba` are used by the Hopfield side study; `tensorboard` is tied to legacy training/checkpoint history; `jupyterlab` is interactive tooling. These are not proven requirements of the three frozen root modules themselves.

No dependency is removed in Phase 3B archival work. Removal requires a repository-wide import scan after the relevant historical material is retired.

## Verification

Protected-file byte exactness is enforced by `scripts/validate_reference_artifacts.sh` and `tests/regression/test_preservation.py`, checked against `artifacts/reference/mmd_final_run/manifest.json`.

## Retirement

The frozen modules are not part of the Phase 3B Hopfield/DMFT retirement. Their eventual retirement requires a separate decision because the protected corrected notebook depends on them.
