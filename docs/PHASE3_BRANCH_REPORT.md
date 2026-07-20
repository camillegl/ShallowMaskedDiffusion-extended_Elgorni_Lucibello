# Phase 3 branch report

Branch: `guthlac`

Baseline: `ed42906cffd0b2b5989eb53e46f00ca6cdde4171` (`main` after Phase 3A).

## Implemented in this branch

- Fast-forwarded `guthlac` to the Phase 3A baseline without changing `main`.
- Added `docs/archive/HOPFIELD_DMFT_ARCHIVE.md`, which records the scientific role, Git blob identifiers, data products, figures, dependency implications, limitations, and recovery commands for the Hopfield/DMFT side study.
- Corrected the stale Phase 3A status in `docs/FROZEN_LEGACY_RUNTIME.md`.
- Corrected `docs/FINAL_REPOSITORY_MAP.md` so it distinguishes completed Phase 3A work from pending retirement categories.

## Deliberately not performed in this commit

No scientific source, note, notebook, result, frozen compatibility module, protected artifact, dependency declaration, or lockfile is deleted or modified. In particular, the unique Hopfield and DMFT theory remains present pending review of the archival record.

The following destructive steps remain separate commits:

1. reviewed Hopfield/DMFT source and artifact retirement;
2. obsolete Julia inventory and retirement;
3. historical notebook and utility retirement;
4. non-protected result cleanup;
5. `train.py` and dependency retirement;
6. CI and final repository hardening.

This separation is intentional: the side-study sources and notes contain unique scientific content, and deletion must remain reversible and independently reviewable.
