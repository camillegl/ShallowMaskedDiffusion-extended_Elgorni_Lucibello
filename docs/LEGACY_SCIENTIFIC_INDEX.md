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

## `src-hopfield/`

- **Purpose**: Python implementation of the clamped-Hopfield side-track —
  `hopfield_saddle_point.py` (replica-symmetric saddle-point solver),
  `mcmc_hopfield.py` (MCMC sampler), `plot_hopfield.py`.
- **Unique scientific content**: yes — self-contained side-study code with no
  duplicate elsewhere in the repository.
- **Planned retirement phase**: a future "Hopfield retirement" phase (not
  Phase 3A).
- **Replacement / retained source**: theory remains in
  `notes/notes_hopfield.typ` (see below); no replacement Python module exists
  yet in `src/maskeddiffusion/`.
- **Recovery**: `git show phase2-hidden-manifold-foundation:src-hopfield/<file>`.

## `notes/notes_hopfield.typ`

- **Purpose**: Typst theory notes for the clamped-Hopfield model (replica
  calculation), companion to `src-hopfield/`.
- **Unique scientific content**: yes.
- **Planned retirement phase**: paired with the future Hopfield retirement
  phase; not retired independently of `src-hopfield/`.
- **Replacement / retained source**: none planned; this remains the only
  written record of that theory unless/until superseded.
- **Recovery**: `git show phase2-hidden-manifold-foundation:notes/notes_hopfield.typ`.

## `notes/notes_dmft_masked_hopfield.typ`

- **Purpose**: Typst theory notes for a DMFT (dynamical mean-field theory)
  treatment of masked Hopfield dynamics.
- **Unique scientific content**: yes.
- **Planned retirement phase**: a future "DMFT retirement" phase (not Phase
  3A); may be retired together with or independently of the Hopfield
  material, depending on later scoping.
- **Replacement / retained source**: none planned.
- **Recovery**: `git show phase2-hidden-manifold-foundation:notes/notes_dmft_masked_hopfield.typ`.

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
