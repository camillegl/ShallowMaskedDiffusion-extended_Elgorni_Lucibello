# Historical notebooks and analysis-utility archive

Status: Phase 3 archival record, `guthlac` branch only.

## Protected notebooks — never modified, retained permanently

- `experiments-analysis/analysis_mmd_distribution_distance_corrected.ipynb`
  — the current, protected MMD analysis; see
  `docs/REFERENCE_RESULTS_MANIFEST.md` / `docs/FROZEN_LEGACY_RUNTIME.md`.
  **The uncorrected sibling notebook retired below must never be cited as
  the current MMD result** — its estimator had a known correction applied
  in the "corrected" notebook.
- `experiments-analysis/mmd_results_presentation_1.ipynb` — protected
  presentation notebook built on the corrected results.

Deleting the exploratory notebooks below is repository maintenance; it does
not validate, re-derive, or strengthen any scientific claim in the protected
notebooks.

## Retired notebooks

| Path | Size (bytes) | Git blob | SHA-256 | Purpose | Superseded? | Retained dependent? |
|---|---|---|---|---|---|---|
| `experiments-analysis/analysis.ipynb` | 1590136 | `43aa4455863f15ebefd27aaceeb7ec404a873318` | `a8ef3f4f070a811e73bfe3770f671788616d425298b2e60d39b595b0adb830a` | General-purpose exploration notebook: walks `logs/` run directories, loads legacy-CLI `train.py` checkpoints via `diffusion.MaskedDiffusion.load_from_checkpoint`, evaluates train/val loss and samples. Uses uniform-data legacy runs, not the hidden-manifold package. | Superseded by `src/maskeddiffusion/` training/evaluation CLIs and the protected MMD notebook for reported results. | None |
| `experiments-analysis/analysis-J.ipynb` | 665268 | `2ce9012d44687f9c818e2e751bbd6ea96e72c9ab` | `361e694d0737ebe556224fa506837742e28a300a66f36d9c6c05976733a4580` | Loads legacy-CLI `W`/`J` weight matrices per sample ratio (`alpha`) and computes/plots `q`,`v` order-parameter histograms via `analyze_W_dict`. Imports `collect_experiment_data` from `experiments-analysis/utils.py`. | Exploratory weight-matrix analysis specific to legacy-CLI runs; no retained equivalent. | None (only consumer of `utils.py` besides `analysis-uturn.ipynb`, also retired) |
| `experiments-analysis/analysis-mnist.ipynb` | 209872 | `614068356f12efb03dec37bd6ec39083591eb2be` | `122a7318fcbc14b68679f151f593ab4864b017e0189e1a6a3814c06f6fc708c` | Same structure as `analysis.ipynb` but for MNIST-derived binarized data (`BinarizedMNIST` in legacy `datasets.py`). | MNIST support is a legacy-CLI-only data path; superseded by the hidden-manifold package's own data path. | None |
| `experiments-analysis/analysis-uturn.ipynb` | 923043 | `cbabd52438d74c158d4718acca1e606c373e279d` | `0b1499c993e9804ff74ceb74e6437a382c45a05b7d04e5fc69046d3d23b1fd5` | Loads U-Turn sampler results (produced by `run_uturn_experiments.py`) and plots them against legacy-CLI runs. Imports `collect_experiment_data` from `utils.py`. | Exploratory U-Turn-sampler evaluation on legacy-CLI runs; no retained equivalent notebook. | None |
| `experiments-analysis/analysis_loss_convergence.ipynb` | 275399 | `fe96faed6d13b304f8b818dd8ff64bf0889ee4e7` | `bd9054d7a56936874dedff78da9e3577633afefebbdcc8beb1c0c88b3ce05b0` | "Finite-sample train/test loss convergence for shallow masked diffusion" — a finite-dimensional empirical sanity check on train/test loss using `RandomFeaturesDataset`/`MaskedDiffusion` from the active package (not the legacy CLI). Explicitly self-described as "not a proof". | Exploratory sanity check, not a source of any reported number; no retained equivalent. | None |
| `experiments-analysis/analysis_mmd_distribution_distance.ipynb` | 522346 | `689eae1686b54695c7a1148ac49b7f37eab8ea2f` | `9ca7e5604ec16402f7acccd66414bcc9d1ade896df39f312808e7b3bd79fb2d` | **Superseded, uncorrected** MMD distribution-distance analysis — the predecessor to the protected `..._corrected.ipynb`. Already marked stale per `.claude/rules/notebooks.md`'s "superseded notebook" rule prior to this retirement; not physically deleted before now. | Superseded by `analysis_mmd_distribution_distance_corrected.ipynb`, which applies a correction to this notebook's MMD estimator. Its numbers must not be cited as current. | None (protected notebooks do not read this file) |

## Retired utilities

| Path | Size (bytes) | Git blob | SHA-256 | Purpose | Consumers |
|---|---|---|---|---|---|
| `experiments-analysis/utils.py` | 4760 | `8fcfe0ef598e0ecf9dc2152ac09aeb21ee55fe07` | `081fd6aef589a759fce6c1aa04794a27a3496ffa0ed7555f18fda6ed98e6c47` | `load_model_and_data` / `collect_experiment_data` helpers for walking legacy-CLI `logs/` run directories and loading `diffusion.MaskedDiffusion` checkpoints. | Only `analysis-J.ipynb` and `analysis-uturn.ipynb` import from this module (`git grep` confirmed) — both retired in this same commit. No protected notebook or retained script imports it. |
| `experiments-analysis/run_uturn_experiments.py` | 11003 | `013667bd4cd2c90ef4c15063ad2b431b59b16e2c` | `052dda9cd43f6f9c2711d850271b3f7d524aed1bcebf45afc1d57cba8237db1` | Standalone script: runs the U-Turn sampler on legacy-CLI trained checkpoints and writes analysis plots/CSVs (imports `diffusion.MaskedDiffusion` and `utils.collect_experiment_data`). | Only consumer identified is `analysis-uturn.ipynb`'s implicit dependency on its output files; not imported by any retained module. |

## Notebooks were not executed

Per `.claude/rules/notebooks.md`, notebook cells were not run as part of this
archival or retirement. All "purpose" descriptions above come from reading
notebook markdown/code cells and existing embedded outputs read-only.

## Recovery

```bash
git show phase2-hidden-manifold-foundation:<path>
git checkout phase2-hidden-manifold-foundation -- <path>
git show ed42906cffd0b2b5989eb53e46f00ca6cdde4171:<path>
git checkout ed42906cffd0b2b5989eb53e46f00ca6cdde4171 -- <path>
```
