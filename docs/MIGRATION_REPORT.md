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
| `experiments-analysis/utils.py`, `run_uturn_experiments.py` | kept | used by historical (non-protected) notebooks |

No legacy module was deleted: the protected notebooks' importability outranks
cleanup (stop-condition compliance).

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
- `scripts/train-*.sh`, `span-slurm.sh`, `uturn.sh` — tied to the legacy CLI;
  delete together with `train.py` when historical reproduction is no longer
  needed.
- MNIST support (`BinarizedMNIST` in `datasets.py`, `torchvision`,
  `tensorboard`, `lightning`, `numba`, `tqdm`, `pandas`, `scipy`,
  `matplotlib`, `jupyterlab` deps) — removable only when the legacy modules
  and old notebooks retire.

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
