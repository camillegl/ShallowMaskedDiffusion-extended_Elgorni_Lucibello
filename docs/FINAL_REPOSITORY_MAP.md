# Final repository map (post-retirement target)

This document describes the intended shape of the repository after all planned retirement phases are complete. It is a target, not a statement that every retirement step has already run.

Phase 3A is complete at `ed42906cffd0b2b5989eb53e46f00ca6cdde4171`. Phase 3B archival work is isolated on the `guthlac` branch. Hopfield/DMFT deletion, Julia retirement, historical-notebook retirement, result cleanup, legacy-CLI retirement, dependency reduction, and CI hardening remain separate reviewable steps. Deleting old material does not validate `src/maskeddiffusion/`.

## Retained target paths

```text
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

## Retained compatibility code

`datasets.py`, `diffusion.py`, and `models.py` remain frozen compatibility code because the protected corrected MMD notebook depends on them. They are not active implementation and must not be imported by `src/maskeddiffusion/`.

## Pending retirement categories

- Hopfield/DMFT sources, notes, data, and figures: archived in `docs/archive/HOPFIELD_DMFT_ARCHIVE.md`; deletion requires a distinct reviewed change.
- `julia-code/SP/` and `julia-code/old/`: pending exact inventory and archival.
- Historical, non-protected notebooks and utilities: pending exact inventory and archival.
- Non-protected result tables and generated figures: pending canonicality review.
- `train.py`: deprecated, non-protected legacy CLI; retirement follows removal of its historical consumers.
- `paper/bibliography.bib`: pending comparison with `notes/bibliography.bib`.

## Recovery

Retired paths remain recoverable from `phase2-hidden-manifold-foundation`, from the Phase 3A baseline `ed42906cffd0b2b5989eb53e46f00ca6cdde4171`, and from subsequent branch history.
