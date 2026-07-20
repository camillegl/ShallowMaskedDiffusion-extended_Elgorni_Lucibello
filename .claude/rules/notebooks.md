# Notebook rules

Applies to `experiments-analysis/*.ipynb` and any future notebooks.

- Notebooks are **reports and exploratory interfaces**, not libraries. No essential
  implementation (estimators, data generation, samplers) may exist only in a notebook; promote
  such code to a module and have the notebook call it.
- Known current violations (do not add more): the MMD estimator, the sign-channel `0 → +1`
  patch, and per-notebook sampler copies live only in notebooks (see `docs/REPO_AUDIT.md`).
- Notebook outputs embedded in `.ipynb` files are review artifacts, not source data. Do not
  treat cell outputs as inputs to further analysis without a manifest tying them to a commit
  and configuration.
- When a notebook is superseded (e.g., `analysis_mmd_distribution_distance.ipynb` is
  superseded by `..._corrected.ipynb`), mark the stale one clearly at the top rather than
  deleting it, and never quote results from the stale one.
- Do not execute expensive notebook sweeps as a side effect of editing or validation.
