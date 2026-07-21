# Legacy scientific index

**Phase 3 status: MERGED.** Every "deleted from the `guthlac` branch" / "Only on
`guthlac` — `main` is unaffected" status line below described the state at the time each
section was written, before review. All of Phase 3 was merged into `main` via
[PR #2](https://github.com/camillegl/ShallowMaskedDiffusion-extended_Elgorni_Lucibello/pull/2)
(merge commit `c6a716f2e8915c7a01864d1658275f9305586f5`) — these deletions are now on
`main` too.

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
- **Residual retirement gap: closed.** The initial retirement commit did not
  remove two stale-parameter data files
  (`data/hopfield_T0_mcmc_N10000_S10_seed0.npz`,
  `data/hopfield_T001_mcmc_N10000_S10_seed0.npz`), six figures under
  `notes/plots/hopfield_*mcmc*.png`, or the two compiled PDFs
  (`notes/notes_hopfield.pdf`, `notes/notes_dmft_masked_hopfield.pdf`). These
  ten paths were removed in the follow-up commit "chore: complete Hopfield
  and DMFT retirement" on `guthlac`; full inventory in
  `docs/archive/HOPFIELD_DMFT_ARCHIVE.md`, removal record in
  `docs/archive/HOPFIELD_DMFT_RETIREMENT.md`.

## `julia-code/SP/` (retired)

**Status: deleted from the `guthlac` branch** (commit "chore: retire obsolete
Julia implementations"). **Only on `guthlac` — `main` is unaffected.**

- **Purpose**: Julia "SP" (saddle-point) implementation —
  `MaskedDiffusion_SP.jl`, `ODE.jl`, `helpers.jl`, `common.jl`, `methods.jl`,
  plus its own `Project.toml`/`Manifest.toml`, scripts, data, and notebooks.
- **Unique scientific content**: scoped to the uniform-data / original
  masked-diffusion saddle-point theory and its retrieval-overlap ODE; full
  per-file inventory (size, blob, SHA-256, role) in
  `docs/archive/JULIA_LEGACY_ARCHIVE.md`.
- **Replacement / retained source**: `julia-code/hiddenmanifold/` is the
  retained, relevant theory code for the hidden-manifold extension;
  `julia-code/SP/` targeted a different data model and observable, so it is
  not superseded by it, only retired as out-of-scope legacy material.
- **Recovery**: `git show phase2-hidden-manifold-foundation:julia-code/SP/<file>`
  or `git show ed42906cffd0b2b5989eb53e46f00ca6cdde4171:julia-code/SP/<file>`.

## `julia-code/old/` (retired)

**Status: deleted from the `guthlac` branch** (commit "chore: retire obsolete
Julia implementations"). **Only on `guthlac` — `main` is unaffected.**

- **Purpose**: Older Lux.jl/Zygote-based neural-network training
  implementation on uniform data — `lux_train_uniform_data.jl`, own
  `Project.toml`/`Manifest.toml`.
- **Unique scientific content**: treated as superseded (directory name
  "old", distinct/older toolchain from `julia-code/SP/`); no formal
  cross-check against a specific successor is recorded — see
  `docs/archive/JULIA_LEGACY_ARCHIVE.md` for the documented uncertainty.
- **Replacement / retained source**: superseded by later Julia work and
  ultimately by the active Python package `src/maskeddiffusion/`.
- **Recovery**: `git show phase2-hidden-manifold-foundation:julia-code/old/<file>`
  or `git show ed42906cffd0b2b5989eb53e46f00ca6cdde4171:julia-code/old/<file>`.

## Historical analysis notebooks and utilities (retired)

**Status: deleted from the `guthlac` branch** (commit "chore: retire
historical analysis notebooks"). **Only on `guthlac` — `main` is
unaffected.**

`experiments-analysis/analysis.ipynb`, `analysis-J.ipynb`, `analysis-mnist.ipynb`,
`analysis-uturn.ipynb`, `analysis_loss_convergence.ipynb`,
`analysis_mmd_distribution_distance.ipynb` (superseded by the protected
`analysis_mmd_distribution_distance_corrected.ipynb`), plus utilities
`experiments-analysis/utils.py` and `experiments-analysis/run_uturn_experiments.py`.

- **Purpose**: historical exploratory reports — uniform-data training curves,
  U-turn experiments, MNIST runs, and the pre-correction MMD study — plus
  their shared checkpoint-loading/U-turn-batch-runner utilities.
- **Unique scientific content**: historical evidence with embedded outputs;
  `analysis_mmd_distribution_distance.ipynb` in particular is superseded
  content and must never be quoted as a current result (see
  `.claude/rules/notebooks.md`). Full per-file inventory (size, blob,
  SHA-256, purpose, consumers) in
  `docs/archive/HISTORICAL_NOTEBOOKS_ARCHIVE.md`.
- **Replacement / retained source**: the two protected notebooks
  (`analysis_mmd_distribution_distance_corrected.ipynb`,
  `mmd_results_presentation_1.ipynb`) are the current authoritative and
  presentation records respectively; these historical notebooks are not
  replaced, only superseded as reports. The U-turn mechanism itself
  (`diffusion.py:119-126` `test_step`) is retained; only the batch-driver
  script is retired — see `docs/EQUATION_TO_CODE_MAP.md`,
  `docs/ORIGINAL_ARCHITECTURE.md`.
- **Recovery**: `git show phase2-hidden-manifold-foundation:experiments-analysis/<name>`
  or `git show ed42906cffd0b2b5989eb53e46f00ca6cdde4171:experiments-analysis/<name>`.

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
