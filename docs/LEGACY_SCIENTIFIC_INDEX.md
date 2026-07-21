# Legacy scientific index

Archival index of material scheduled for **later** retirement (not Phase 3A).
This document exists so that later retirement phases (Hopfield, DMFT, Julia,
notebook, result, or dependency retirement — all explicitly out of scope here)
have a single map of what exists, why, and how to recover it if removed. No
file listed here is deleted, moved, or modified by Phase 3A. Deleting old code
does not, by itself, validate the new implementation in
`src/maskeddiffusion/`; retirement and validation are separate activities.

Recovery baseline for everything below: commit
`98b5afb5a6a2c649ad53c49077a177c8c48a399f` on `main`, and the stable tag
`phase2-hidden-manifold-foundation`, both of which predate any Phase 3A
deletion. `git checkout phase2-hidden-manifold-foundation -- <path>` recovers
any listed file from that point in history even after a future retirement
phase deletes it from the working tree.

## `src-hopfield/`, `notes/notes_hopfield.typ`, `notes/notes_dmft_masked_hopfield.typ`

**Status: deleted from the `guthlac` branch** at commit
`177fd8f84b0b02b799be057259ff74318c8761d7` ("chore: retire Hopfield and DMFT
side study"), preceded by an archival commit
`513fe7755d9dd65718da6b7720264033a0fd61a3`. **This retirement exists only on
`guthlac` — `main` is unaffected and still has all of this material at
`ed42906cffd0b2b5989eb53e46f00ca6cdde4171`.** The full scientific summary,
code map, dependency tracing, and per-file inventory (with Git blob hashes)
live in `docs/archive/HOPFIELD_DMFT_ARCHIVE.md`; the exact list of removed
paths and recovery commands is in `docs/archive/HOPFIELD_DMFT_RETIREMENT.md`.
Deletion did not remove the unique scientific content — it removes it from
this branch's working tree while keeping it fully recoverable from Git
history.

- **Purpose**: `src-hopfield/hopfield_saddle_point.py` (replica-symmetric
  saddle-point solver), `src-hopfield/mcmc_hopfield.py` (MCMC sampler),
  `src-hopfield/plot_hopfield.py`, `notes/notes_hopfield.typ` (clamped-Hopfield
  replica theory), `notes/notes_dmft_masked_hopfield.typ` (autoregressive/
  masked-Hopfield DMFT-style exploration).
- **Unique scientific content**: yes — confirmed by the Phase 3B1 audit
  (`docs/archive/HOPFIELD_DMFT_ARCHIVE.md`) before deletion; self-contained
  side-study code and notes with no duplicate elsewhere in the repository.
- **Retirement status**: performed on `guthlac` only, **not merged to
  `main`** and not yet approved for the main line. Treat this as a reviewable
  branch-local change, not a completed repository-wide retirement.
- **Replacement / retained source**: none — this was the only written record
  of both theories and the only implementation of the clamped-Hopfield
  numerics; nothing in `src/maskeddiffusion/` replaces it.
- **Recovery** (both baselines are currently identical for these paths):
  `git show phase2-hidden-manifold-foundation:<path>` or
  `git show ed42906cffd0b2b5989eb53e46f00ca6cdde4171:<path>`, or
  `git checkout ed42906cffd0b2b5989eb53e46f00ca6cdde4171 -- <path>` to restore
  a working copy on `guthlac` if ever needed.
- **Known incomplete-retirement gap** (not fixed by this reconciliation pass):
  the retirement commit did not remove two stale-parameter data files
  (`data/hopfield_T0_mcmc_N10000_S10_seed0.npz`,
  `data/hopfield_T001_mcmc_N10000_S10_seed0.npz`), six figures under
  `notes/plots/hopfield_*mcmc*.png` (two of which were referenced by the
  now-deleted `notes/notes_hopfield.typ` and are now orphaned), or the two
  compiled PDFs (`notes/notes_hopfield.pdf`,
  `notes/notes_dmft_masked_hopfield.pdf`) rendered from the deleted `.typ`
  sources — all still tracked on `guthlac`. See `docs/MIGRATION_REPORT.md`'s
  Phase 3B section for the full list; resolving this is left to a follow-up
  commit, not performed here.

## `julia-code/SP/`

- **Purpose**: Julia "SP" (saddle-point) implementation —
  `MaskedDiffusion_SP.jl`, `ODE.jl`, `helpers.jl`, `common.jl`, `methods.jl`,
  plus its own `Project.toml`/`Manifest.toml`, scripts, data, and notebooks.
- **Unique scientific content**: treated as scoped to the uniform-data /
  original masked-diffusion saddle-point theory; not traced against
  `notes/notes_memorization.typ` line-by-line in this index, so its exact
  overlap with the retained theory is not fully established here.
- **Planned retirement phase**: a future "Julia retirement" phase (not Phase
  3A).
- **Replacement / retained source**: `julia-code/hiddenmanifold/` is the
  retained, relevant theory code for the hidden-manifold extension (see
  below); `julia-code/SP/` is not superseded by it and its disposition is
  deferred to that later phase.
- **Recovery**: `git show phase2-hidden-manifold-foundation:julia-code/SP/<file>`.

## `julia-code/old/`

- **Purpose**: Older Julia implementation —
  `lux_train_uniform_data.jl`, own `Project.toml`/`Manifest.toml`.
- **Unique scientific content**: presumed superseded (directory name "old"),
  not independently verified in this index.
- **Planned retirement phase**: a future "Julia retirement" phase (not Phase
  3A).
- **Replacement / retained source**: superseded by later Julia and/or Python
  code; exact successor not traced here.
- **Recovery**: `git show phase2-hidden-manifold-foundation:julia-code/old/<file>`.

## Historical analysis notebooks

`experiments-analysis/analysis.ipynb`, `analysis-J.ipynb`, `analysis-mnist.ipynb`,
`analysis-uturn.ipynb`, `analysis_loss_convergence.ipynb`, and
`analysis_mmd_distribution_distance.ipynb` (superseded by the protected
`analysis_mmd_distribution_distance_corrected.ipynb`).

- **Purpose**: historical exploratory reports — uniform-data training curves,
  U-turn experiments, MNIST runs, and the pre-correction MMD study.
- **Unique scientific content**: historical evidence with embedded outputs;
  `analysis_mmd_distribution_distance.ipynb` in particular is superseded
  content and must never be quoted as a current result (see
  `.claude/rules/notebooks.md`).
- **Planned retirement phase**: a future "notebook retirement" phase (not
  Phase 3A).
- **Replacement / retained source**: the two protected notebooks
  (`analysis_mmd_distribution_distance_corrected.ipynb`,
  `mmd_results_presentation_1.ipynb`) are the current authoritative and
  presentation records respectively; these historical notebooks are not
  replaced, only superseded as reports.
- **Recovery**: `git show phase2-hidden-manifold-foundation:experiments-analysis/<name>.ipynb`.

## Legacy result tables and figures

`experiments-analysis/*.csv` and `experiments-analysis/figures/*.png` not
listed in `artifacts/reference/mmd_final_run/manifest.json` (i.e. everything
except the three protected CSVs enumerated in
`docs/REFERENCE_RESULTS_MANIFEST.md`), plus the root-level `experiments.csv`
duplicate flagged in `docs/MIGRATION_REPORT.md`.

- **Purpose**: generated experiment artifacts committed as evidence for
  historical (non-protected) notebooks and figures.
- **Unique scientific content**: unresolved — `docs/MIGRATION_REPORT.md`
  records that `experiments.csv` (root) and
  `experiments-analysis/experiments.csv`, and the two
  `results_mmd_distribution_distance.csv` copies, have **differing** SHA-256
  hashes, so no canonical copy has been determined. This index does not
  resolve that ambiguity.
- **Planned retirement phase**: a future "result retirement" phase (not Phase
  3A); duplicate resolution should happen before or during that phase, not by
  silent deletion.
- **Replacement / retained source**: the three protected CSVs remain the
  authoritative final-run record; these other tables/figures are unreplaced
  historical evidence.
- **Recovery**: `git show phase2-hidden-manifold-foundation:<path>`.

## Stable recovery tag

`phase2-hidden-manifold-foundation` — the tag pinned at the start of Phase 3A,
before any of the retirement phases above and before the Phase 3A deletions in
`docs/FINAL_REPOSITORY_MAP.md`. Use it as the recovery point for anything
listed in this index: `git show <tag>:<path>` or
`git checkout <tag> -- <path>`.

## Explicitly authoritative material (not archived, not scheduled for retirement)

- `notes/notes_memorization.typ` remains the authoritative record of the
  upstream masked-diffusion theory.
- `notes/notes_hiddenmanifold.typ` remains the authoritative record of the
  hidden-manifold extension theory.
- `julia-code/hiddenmanifold/` remains relevant theory code for the
  hidden-manifold extension and is not archived by this index.
