# Final repository map (post-retirement target)

This document describes the **intended** shape of the repository after all
planned retirement phases (Hopfield, DMFT, Julia, notebook, result, and
dependency retirement) are complete. It is a target, not a status report:
none of those retirement phases have run yet, and this document does not
claim they have. Phase 3A (this change) performs only the narrow deletions
listed in the accompanying Phase 3A instructions; it does not begin any of the
retirement phases named above. Reaching the shape below in no way validates
`src/maskeddiffusion/` — validation is a separate, ongoing activity carried
out by the test suite and reviewers, not a side effect of deleting old files.

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

## What this map does not cover

- **Everything not listed above and not explicitly named in
  `docs/LEGACY_SCIENTIFIC_INDEX.md`** (e.g. `src-hopfield/`, `notes/notes_hopfield.typ`,
  `notes/notes_dmft_masked_hopfield.typ`, `julia-code/SP/`, `julia-code/old/`,
  the historical analysis notebooks, and non-protected result tables/figures)
  is scheduled for retirement in a later phase. Its exact disposition
  (deletion vs. relocation vs. permanent archival) is deferred to that phase
  and is not decided here.
- **`train.py`** is not listed above and its fate is likewise deferred; it is
  a legacy CLI, not a hash-pinned protected file (see
  `docs/FROZEN_LEGACY_RUNTIME.md`), and Phase 3A makes no claim about it.
- **`paper/bibliography.bib`** — its fate is **not finalized** by this map. It
  is marked here for later comparison against `notes/bibliography.bib` (which
  of the two, if either, is canonical, and whether they should be merged) —
  a decision explicitly deferred, not made in Phase 3A.
- **`experiments-analysis/figures/`** (the 20 PNG outputs of the corrected
  notebook) is not listed above; its retention or archival is bundled with
  the future "result retirement" phase in `docs/LEGACY_SCIENTIFIC_INDEX.md`,
  not decided here.

## Recovery

Anything removed in a future retirement phase remains recoverable from the
`phase2-hidden-manifold-foundation` tag or from `main` history prior to that
phase's commit(s), per `docs/LEGACY_SCIENTIFIC_INDEX.md`.
