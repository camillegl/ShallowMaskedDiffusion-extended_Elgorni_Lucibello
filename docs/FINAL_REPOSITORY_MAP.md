# Final repository map (post-retirement target)

This document describes the **intended** shape of the repository after all
planned retirement phases (Hopfield, DMFT, Julia, notebook, result, and
dependency retirement) are complete. It is a target, not a status report:
it does not claim every retirement step has already run. Reaching the shape
below in no way validates `src/maskeddiffusion/` — validation is a separate,
ongoing activity carried out by the test suite and reviewers, not a side
effect of deleting old files.

**Phase 3A is complete**, committed at `ed42906cffd0b2b5989eb53e46f00ca6cdde4171`
on `main`/`origin/main` — it performed only the narrow deletions listed in
the Phase 3A instructions (obsolete stubs and legacy cluster scripts) and
begins none of the retirement phases named above. **Phase 3B (Hopfield/DMFT
archival and retirement) is isolated on the `guthlac` branch and is not
merged to `main`** — two commits there
(`513fe7755d9dd65718da6b7720264033a0fd61a3` archival,
`177fd8f84b0b02b799be057259ff74318c8761d7` retirement) removed the
Hopfield/DMFT side study from that branch's working tree; see
`docs/archive/HOPFIELD_DMFT_ARCHIVE.md` and
`docs/archive/HOPFIELD_DMFT_RETIREMENT.md`. Julia retirement, historical-
notebook retirement, non-protected result cleanup, legacy-CLI (`train.py`)
retirement, dependency reduction, and CI hardening remain separate,
unreviewed, reviewable steps — none has begun.

## Retained paths

```
.claude/
artifacts/reference/
configs/
docs/
scripts/
src/maskeddiffusion/
tests/

experiments-analysis/
    analysis_mmd_distribution_distance_corrected.ipynb
    mmd_results_presentation_1.ipynb
    results/results_mmd_distribution_distance_corrected.csv
    results/results_mmd_time_sliced.csv
    results/results_mmd_distribution_distance_corrected_10k.csv

notes/
    notes_memorization.typ
    notes_hiddenmanifold.typ
    bibliography.bib

julia-code/
    hiddenmanifold/

datasets.py
diffusion.py
models.py

README.md
CLAUDE.md
AGENTS.md
CITATION.cff
pyproject.toml
uv.lock
.python-version
```

## Status of retained legacy files

`datasets.py`, `diffusion.py`, and `models.py` are retained as **frozen
compatibility code**, not active implementation. They are not part of
`src/maskeddiffusion/` and must never be imported by it (see
`docs/FROZEN_LEGACY_RUNTIME.md` for the full operating rules). Their presence
in this target map reflects that the protected corrected MMD notebook depends
on them for re-runnability, not that they are maintained or extended going
forward.

## What this map does not cover / pending retirement categories

- **Hopfield/DMFT sources, notes, data, and figures**: archived and (on
  `guthlac` only) fully deleted, including the initially-missed residual
  files; see `docs/archive/HOPFIELD_DMFT_ARCHIVE.md` and
  `docs/archive/HOPFIELD_DMFT_RETIREMENT.md`. Merging this retirement into
  `main` requires a distinct, separately reviewed change — it is not decided
  here.
- **`julia-code/SP/` and `julia-code/old/`**: archived and (on `guthlac`
  only) deleted; see `docs/archive/JULIA_LEGACY_ARCHIVE.md`. Not superseded
  by `julia-code/hiddenmanifold/`, which is retained and unaffected.
- **Historical, non-protected notebooks and utilities** (e.g. `analysis.ipynb`,
  `analysis-J.ipynb`, `analysis-mnist.ipynb`, `analysis-uturn.ipynb`,
  `analysis_loss_convergence.ipynb`, `analysis_mmd_distribution_distance.ipynb`):
  pending exact inventory and archival; its exact disposition (deletion vs.
  relocation vs. permanent archival) is deferred and not decided here.
- **`train.py`**: deprecated, non-protected legacy CLI, not hash-pinned; its
  retirement is deferred until its historical consumers (old scripts,
  superseded notebooks) are also retired (see `docs/FROZEN_LEGACY_RUNTIME.md`).
- **`paper/bibliography.bib`**: its fate is **not finalized** by this map. It
  is marked here for later comparison against `notes/bibliography.bib` (which
  of the two, if either, is canonical, and whether they should be merged) —
  a decision explicitly deferred.
- **`experiments-analysis/figures/`** (the 20 PNG outputs of the corrected
  notebook): not listed above; its retention or archival is bundled with the
  future "result retirement" phase in `docs/LEGACY_SCIENTIFIC_INDEX.md`, not
  decided here.

## Recovery

Anything removed in a completed or future retirement phase remains
recoverable from the `phase2-hidden-manifold-foundation` tag, from the
Phase 3A baseline `ed42906cffd0b2b5989eb53e46f00ca6cdde4171`, or from
subsequent branch history (e.g. `guthlac`'s pre-retirement commits), per
`docs/LEGACY_SCIENTIFIC_INDEX.md`.
