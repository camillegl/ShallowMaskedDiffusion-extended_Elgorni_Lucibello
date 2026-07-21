# Migration report — Phase 2 (2026-07-20)

Starting point: branch `main`, commit `fea744d`. (Line references in the Phase-1 documents
`docs/UPSTREAM_DISCREPANCIES.md` / `docs/EQUATION_TO_CODE_MAP.md` cite the audit snapshot
`2e2db70`; `fea744d` only removed `rsync-logs.sh`, so those references remain valid.) Phase 2 replaced the active
implementation with the `src/maskeddiffusion` package while preserving legacy
behavior as fixtures and protecting the final-run MMD notebooks.

## What was built

- `src/maskeddiffusion/`: dimensions, hierarchical randomness, teacher,
  masking, linear score, objectives, samplers, direct training loop,
  checkpoints, artifact schema, metrics (MMD/correlations/overlaps), four
  CLIs. ADRs 001–004 record the framework decisions.
- `tests/`: 112 tests (unit, property, integration, regression) including
  legacy-behavior fixtures and MMD notebook-equivalence.
- `tests/fixtures/original_architecture_v1/`: five deterministic fixtures
  pinning legacy behavior (linear model, masking, objective, stochastic
  sampler trajectory, greedy reconstruction) + manifest with source commit
  `fea744d` and environment versions.
- Preservation: `docs/REFERENCE_RESULTS_MANIFEST.md`,
  `artifacts/reference/mmd_final_run/manifest.json`,
  `tests/regression/test_preservation.py`,
  `scripts/validate_reference_artifacts.sh`.

## Legacy modules: status

| Path | Status | Reason |
|---|---|---|
| `diffusion.py` | **kept** (superseded, frozen) | imported by the protected corrected notebook; source of the pinned fixtures |
| `models.py` | **kept** (superseded, frozen) | imported by `diffusion.py` |
| `datasets.py` | **kept** (superseded, frozen) | imported by the protected corrected notebook |
| `train.py` | **kept** (superseded, deprecated in help text) | legacy CLI for reproducing historical runs; `--alpha` help now states the M/N convention and points to `maskeddiffusion-train` |
| `experiments-analysis/utils.py`, `run_uturn_experiments.py` | **deleted on `guthlac`** (Phase 3D, see below) | historical (non-protected) notebook consumers were also retired |

No legacy module was deleted at Phase 3A: the protected notebooks'
importability outranks cleanup (stop-condition compliance). `utils.py` and
`run_uturn_experiments.py` were later retired on `guthlac` (not Phase 3A;
not `main`) once their only consumers (historical notebooks) were retired.

## Edits to legacy files (behavior-preserving, sanctioned)

- `diffusion.py`: removed the stray `from turtle import pd` (unused; dragged
  tkinter in at import). Closes part of D13.
- `train.py`: fixed the `rfs10-tanh` → `rfs10_tanh` help text (D10) and
  labeled `--alpha` with its legacy M/N meaning (D1 warning). No behavioral
  change.

## Deletions

**None.** Verified before deciding:

- `experiments.csv` (root) vs `experiments-analysis/experiments.csv`: SHA-256
  **differ** — not duplicates; canonical copy undetermined; deletion blocked.
- `experiments-analysis/results_mmd_distribution_distance.csv` vs
  `experiments-analysis/results/results_mmd_distribution_distance.csv`:
  SHA-256 **differ** — same conclusion.

## Deferred deletion candidates (need owner sign-off; scientific roles NOT
yet extracted)

- `notes/*.typ` + PDFs — still the only home of the upstream equations;
  must not be deleted before the manuscript exists.
- `src-hopfield/`, `data/*.npz`, `julia-code/` — self-contained side studies /
  reference implementations.
- Superseded notebooks (`analysis.ipynb`, `analysis-J.ipynb`,
  `analysis-mnist.ipynb`, `analysis-uturn.ipynb`,
  `analysis_loss_convergence.ipynb`,
  `analysis_mmd_distribution_distance.ipynb`) — historical evidence with
  embedded outputs.
- MNIST support (`BinarizedMNIST` in `datasets.py`, `torchvision`,
  `tensorboard`, `lightning`, `numba`, `tqdm`, `pandas`, `scipy`,
  `matplotlib`, `jupyterlab` deps) — removable only when the legacy modules
  and old notebooks retire.

## Phase 3A — committed at `ed42906cffd0b2b5989eb53e46f00ca6cdde4171` on `main`, pushed to `origin/main`

- Deleted: `scripts/train-cpu.sh`, `scripts/train-gpu.sh`,
  `scripts/train-cpu-mnist.sh`, `scripts/span-slurm.sh`, `scripts/uturn.sh`
  (the legacy cluster/local run scripts deferred above — `train.py` itself
  was **not** deleted, so this closes only the scripts half of that deferred
  candidate), `paper/main-neuralnetworks.typ` (manuscript stub, no
  equations), and `notes/notes_generalization_TODO.typ`.
- The upstream and hidden-manifold theory remain exactly where they were:
  `notes/notes_memorization.typ` (upstream) and `notes/notes_hiddenmanifold.typ`
  (hidden-manifold extension); neither was touched.
- New retirement documents: `docs/FROZEN_LEGACY_RUNTIME.md`,
  `docs/LEGACY_SCIENTIFIC_INDEX.md`, `docs/FINAL_REPOSITORY_MAP.md`.
- This phase performed no Hopfield, DMFT, Julia, notebook, or dependency
  retirement, and no scientific code or notation change.
- Status: **committed and pushed** to `main`/`origin/main`.

## Phase 3B — isolated on the `guthlac` branch (not merged to `main`)

- Two commits on `guthlac`, both created directly through the GitHub API and
  reviewed in a local checkout after the fact:
  - `513fe7755d9dd65718da6b7720264033a0fd61a3` ("docs: archive Hopfield and
    DMFT side study") — added `docs/archive/HOPFIELD_DMFT_ARCHIVE.md` and
    `docs/PHASE3_BRANCH_REPORT.md`, and updated `docs/FROZEN_LEGACY_RUNTIME.md`
    / `docs/FINAL_REPOSITORY_MAP.md` to reflect Phase 3A's committed status.
    No deletion in this commit.
  - `177fd8f84b0b02b799be057259ff74318c8761d7` ("chore: retire Hopfield and
    DMFT side study") — deleted the three `src-hopfield/*.py` files, both
    Hopfield/DMFT theory notes (`notes/notes_hopfield.typ`,
    `notes/notes_dmft_masked_hopfield.typ`), three `data/hopfield_*_N20000_*.npz`
    files, and four `notes/plots/hopfield_*.png` figures; added
    `docs/archive/HOPFIELD_DMFT_RETIREMENT.md` recording exactly what was
    removed and how to recover it.
- **Incomplete retirement — closed by follow-up commit.** The retirement
  commit above left several related tracked files behind on `guthlac` — two
  stale-parameter data files
  (`data/hopfield_T0_mcmc_N10000_S10_seed0.npz`,
  `data/hopfield_T001_mcmc_N10000_S10_seed0.npz`), six figures
  (`notes/plots/hopfield_T001_m_vs_t_mcmc_{pattern,random}.png`,
  `notes/plots/hopfield_T0_m_vs_t_mcmc_{pattern,random}.png`,
  `notes/plots/hopfield_T0_sweeps_mcmc_{pattern,random}.png` — two of the
  first pair were actually referenced by the now-deleted
  `notes/notes_hopfield.typ`, so they were orphaned), and the two compiled
  PDFs (`notes/notes_hopfield.pdf`, `notes/notes_dmft_masked_hopfield.pdf`)
  rendered from the deleted `.typ` sources. These ten paths were deleted by
  the follow-up commit "chore: complete Hopfield and DMFT retirement"; full
  inventory (size, blob, SHA-256, producer) in
  `docs/archive/HOPFIELD_DMFT_ARCHIVE.md`, removal record in
  `docs/archive/HOPFIELD_DMFT_RETIREMENT.md`.
- Phase 3B exists **only** on `guthlac`; `main` is unaffected and remains at
  Phase 3A (`ed42906cffd0b2b5989eb53e46f00ca6cdde4171`).
- This phase (plus its Hopfield/DMFT follow-up) performed no Julia, notebook,
  result, legacy-CLI, dependency, or CI retirement — those remain separate
  steps, addressed later in this same Phase 3 effort per
  `docs/PHASE3_BRANCH_REPORT.md`.

## Phase 3C — obsolete Julia implementations (isolated on `guthlac`)

- Commit "chore: retire obsolete Julia implementations": deleted all tracked
  files under `julia-code/SP/` (saddle-point/ODE theory for the uniform-data
  model, 14 files) and `julia-code/old/` (older Lux-based training
  implementation, 3 files). `julia-code/hiddenmanifold/` is unaffected.
- Full per-file inventory (size, blob, SHA-256, scientific/engineering role)
  recorded in `docs/archive/JULIA_LEGACY_ARCHIVE.md` before deletion.
- Phase 3C exists only on `guthlac`; `main` is unaffected.

## Phase 3D — historical analysis notebooks and utilities (isolated on `guthlac`)

- Commit "chore: retire historical analysis notebooks": deleted six
  non-protected exploratory notebooks (`analysis.ipynb`, `analysis-J.ipynb`,
  `analysis-mnist.ipynb`, `analysis-uturn.ipynb`,
  `analysis_loss_convergence.ipynb`, `analysis_mmd_distribution_distance.ipynb`)
  and their two shared utilities (`utils.py`, `run_uturn_experiments.py`).
  The two protected notebooks
  (`analysis_mmd_distribution_distance_corrected.ipynb`,
  `mmd_results_presentation_1.ipynb`) are untouched.
- Full per-file inventory (size, blob, SHA-256, purpose, consumers) recorded
  in `docs/archive/HISTORICAL_NOTEBOOKS_ARCHIVE.md` before deletion.
- Updated `docs/EQUATION_TO_CODE_MAP.md` and `docs/ORIGINAL_ARCHITECTURE.md`
  to drop the now-stale line-number citation of the deleted
  `run_uturn_experiments.py` driver script; the underlying mechanism it
  drove (`diffusion.py:119-126` `test_step`) is unaffected and unchanged.
- Phase 3D exists only on `guthlac`; `main` is unaffected.

## Dependencies removed

- `hydra-core` (never imported anywhere — verified by grep over all .py and
  notebook sources).
- `torch-tb-profiler` (same verification).

Added: `numpy` (explicit), dev group `pytest`/`ruff`/`mypy`. Build system:
hatchling with `src/` layout and console-script entry points.

## Compatibility surfaces left in place

- `LinearScoreConfig(normalization="legacy_init_only", diagonal_policy="free")`
  — the named legacy-compat mode used by regression fixtures (ADR 004).
- The legacy flat modules themselves (table above). No new shims were needed:
  nothing in the new package imports legacy code and vice versa.

## Discrepancy resolutions (see docs/UPSTREAM_DISCREPANCIES.md)

- Fixed **in the active path**: D3 (runtime `1/√N` explicit; legacy convention
  survives as named compat mode), D4 (separate value/mask tensors), D6
  (regularization excludes frozen params), D8 (`sign(0):=+1` in the teacher).
- Fixed in legacy text: D10 (help string), part of D13 (turtle import,
  README/AGENTS.md staleness, `.python-version` and `uv.lock` now real and
  tracked).
- Still open (scientific): D7 (V≡0 under fixed F — the model now supports the
  trainable-V ablation), D11 (joint-law consistency), D12 documentation
  status unchanged. D1 remains open-by-design in the legacy CLI (never
  reinterpreted; active CLI rejects bare `alpha`).
