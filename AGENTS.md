# AGENTS.md

## Project Overview

ShallowMaskedDiffusion is a research codebase studying **shallow masked diffusion models** as associative memory systems. The core investigation: how simple (linear) models trained with a masked diffusion objective memorize vs. generalize over discrete binary data distributions.

Authors: Filippo Elgorni, Carlo Lucibello (Bocconi University).

---

## Repository Layout

```
ShallowMaskedDiffusion/
├── train.py                       # Main entry point — trains a masked diffusion model
├── diffusion.py                   # MaskedDiffusion LightningModule (loss, sampling)
├── models.py                      # Model backbones: Linear, RandomFeature, Tensor
├── datasets.py                    # Dataset classes: UniformIsing, RandomFeatures, BinarizedMNIST
├── pyproject.toml                 # Python deps (uv-managed, Python 3.11)
├── scripts/                       # SLURM submission scripts
├── experiments-analysis/          # Post-training analysis
│   ├── utils.py                   # Checkpoint loading, overlap computation
│   ├── run_uturn_experiments.py   # U-Turn sampler batch runner
│   └── *.ipynb                    # Jupyter analysis notebooks
├── src-hopfield/                  # Clamped Hopfield model: RS theory + MCMC validation
│   ├── hopfield_saddle_point.py   # RS saddle-point equations, T=0 and finite-β
│   ├── mcmc_hopfield.py           # Numba MCMC (Glauber T=0, Metropolis T=0.01)
│   └── plot_hopfield.py           # Theory figures (m vs t, phase diagrams)
├── notes/                         # Typst research notes & compiled PDFs
├── paper/                         # Manuscript (Typst) + bibliography
├── data/                          # Cached numerical results (npz) for theory/MCMC
├── julia-code/                    # Reference Julia implementations
└── logs/                          # TensorBoard logs + checkpoints (gitignored)
```

---

## Environment Setup

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Run any script (uv manages the venv automatically)
uv run python train.py --help
```

Python version: **3.11** (pinned in `.python-version`). Core deps: PyTorch, Lightning, NumPy, SciPy, Numba, Matplotlib, pandas, TensorBoard.

---

## Training

```bash
# Minimal example
uv run python train.py --L 1024 --alpha 0.1 --l2reg 0.0 --dataset uniform --model linear

# MNIST
uv run python train.py --L 784 --alpha 0.1 --dataset binarized_mnist --model linear

# Random features dataset with expanded model
uv run python train.py --L 512 --dataset rf3.5 --model rfs10_tanh
```

Key arguments:
| Argument | Default | Meaning |
|---|---|---|
| `--L` | 128 | Data dimensionality |
| `--alpha` | 0.1 | Training samples / L ratio |
| `--dataset` | `uniform` | `uniform`, `rf<factor>`, `binarized_mnist` |
| `--model` | `linear` | `linear`, `rfs<expansion>_<act>`, `tensor<nbins>` |
| `--epochs` | 100 | Training epochs |
| `--batch-size` | 512 | Batch size |
| `--lr` | 1e-3 | Learning rate |
| `--l2reg` | 0.0 | L2 regularization strength |
| `--seed` | -1 | Random seed (-1 = unset) |
| `--test` | false | Run test phase after training |
| `--freeze-mask-weights` | false | Freeze the V weight matrix |
| `--bias` | false | Include bias terms |

Logs are written to `logs/{model}_{dataset}_L{L}_alpha{alpha}_l2reg{l2reg}/` in TensorBoard format. Each run also saves the dataset split and generated samples.

### SLURM (cluster)

```bash
sbatch scripts/train-cpu.sh 2000 0.1 0.0   # L alpha l2reg
sbatch scripts/train-gpu.sh
```

---

## Analysis

After training, use the scripts and notebooks in `experiments-analysis/`:

```bash
# U-Turn experiment batch runner
cd experiments-analysis
uv run python run_uturn_experiments.py --L 2000 --l2reg 0.01 --dataset uniform \
    --model linear --output-dir .
```

Interactive analysis lives in the Jupyter notebooks:
- `analysis.ipynb` — main results
- `analysis-uturn.ipynb` — U-Turn sampler analysis
- `analysis-mnist.ipynb` — MNIST-specific plots

---

## Core Abstractions

### `MaskedDiffusion` ([diffusion.py](diffusion.py))
PyTorch Lightning module wrapping any backbone. Key methods:
- `_compute_loss()` — weighted binary cross-entropy on randomly masked positions
- `sample()` — autoregressive unmasking (fair / greedy / verygreedy strategies)
- `mask_and_sample()` — masks a fraction of an input, then decodes it

### Backbones ([models.py](models.py))
- `LinearBackbone` — W (unmasked tokens) + V (mask indicator), optional bias
- `RandomFeatureScore` — fixed random first layer, learnable output projection
- `TensorBackbone` — time-dependent linear models with per-bin weight matrices

### Datasets ([datasets.py](datasets.py))
All implement the PyTorch `Dataset` interface with save/load:
- `UniformIsingDataset` — binary vectors in {-1, +1}
- `RandomFeaturesDataset` — data drawn from a random feature model
- `BinarizedMNIST` — flattened binarized MNIST images

---

## Coding Conventions

- **Framework**: PyTorch + PyTorch Lightning; no raw training loops.
- **Package manager**: `uv` — always run scripts with `uv run python …`.
- **No test suite** — correctness is verified via analysis notebooks and overlap metrics logged during training.
- **Logging**: TensorBoard via Lightning's `TensorBoardLogger`; metrics include `train_loss`, `train_acc`, `l2_loss`.
- Comments are minimal by design; logic is in variable names and structure.

---

## Clamped Hopfield Side-Track

[src-hopfield/](src-hopfield/) holds a self-contained study of the **clamped Hopfield model** — a fraction `1 - t` of spins is fixed to the first stored pattern and the remaining `t` fraction obeys RS saddle-point equations. This is the theoretical companion to the masked-diffusion sampling dynamics.

```bash
# Compute & cache theory curves, then plot
uv run python src-hopfield/hopfield_saddle_point.py
uv run python src-hopfield/plot_hopfield.py

# MCMC validation (Numba-jitted; T=0 Glauber and T=0.01 Metropolis)
uv run python src-hopfield/mcmc_hopfield.py
```

Cached results land in [data/](data/) as `.npz`; figures land in [notes/plots/](notes/plots/) and feed [notes/notes_hopfield.typ](notes/notes_hopfield.typ).

---

## Research Notes

- [notes/notes_memorization.typ](notes/notes_memorization.typ) — associative memory interpretation of masked diffusion + replica calculations.
- [notes/notes_hopfield.typ](notes/notes_hopfield.typ) — clamped Hopfield RS theory and MCMC validation.
- [notes/notes_generalization_TODO.typ](notes/notes_generalization_TODO.typ) — work-in-progress on generalization.
- [paper/main-neuralnetworks.typ](paper/main-neuralnetworks.typ) — manuscript draft.

Edit Typst sources with the Typst CLI or the Typst web app; PDFs in this repo are committed alongside their `.typ` sources.
