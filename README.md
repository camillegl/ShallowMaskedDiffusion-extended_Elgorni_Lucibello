# Shallow Masked Diffusion

This repository contains code for training and evaluating shallow masked diffusion models, as described in the `notes/` directory. 


## Setup Python Environment

Install the [uv](https://docs.astral.sh/uv/) python package and environment manager:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Code Usage

### Training
```
uv run python train.py --L 1024 --alpha 0.1 --l2reg 0.0 --dataset uniform --model linear
```

To train on a slurm cluster, use the scripts in the `slurm-jobs/` directory.

### Analysis

For analyzing results and produce plots, run the Jupyter notebook `analysis.ipynb`.




## Code Organization

TODO

## Notes

Notes are written in [Typst](https://typst.app/) and are located in the `notes/` directory. 

You can edit the notes locally using the VSCode extension `Tinymist Typst`. You can also install the
`Typst Math` extension for a Lyx-like math editing experience.

The notes are also synced with a typst web app project that you can edit at this link https://typst.app/project/w62GPZe8S65lWDqfCZvKxW.

## Plots 

Some plots are saved in `notes/plots`.