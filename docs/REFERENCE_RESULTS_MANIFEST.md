# Reference results manifest — protected MMD final run

Machine-readable counterpart: `artifacts/reference/mmd_final_run/manifest.json`.
Preservation test: `tests/regression/test_preservation.py`; shell check:
`scripts/validate_reference_artifacts.sh`.

Recorded at commit `fea744d41b4c960c8dbfe8df7f6da82f297ba8b7`, 2026-07-20.
**The notebooks were not rerun during the migration.** All hashes were taken
from the clean working tree (both notebooks were git-clean when hashed).

## Protected notebooks

| Path | SHA-256 | Size | Status |
|---|---|---|---|
| `experiments-analysis/analysis_mmd_distribution_distance_corrected.ipynb` | `3bc29d1904fe1444db0e5815bc4da28c4ec40b895c4889c2513d87a715b5966c` | 680,987 B | tracked, clean |
| `experiments-analysis/mmd_results_presentation_1.ipynb` | `107b32c77bc88bd2bf7c53adb1e26b17c0bb2c58a2b2f3d36ff764fe4daa6d43` | 856,650 B | tracked, clean |

**Corrected final-run notebook** — authoritative record of the FINAL_RUN
experiment (100k MMD samples, kernel scales λ∈{4,8}, sampler k=1, objective
`repo_uniform_t`, V frozen at 0, bias off). Kernel metadata: kernelspec
`python3` (display name ".venv"), language_info 3.14.6. Results are both
embedded in outputs (10 cells carry outputs) and loaded from the results CSVs
below. Its markdown claims are hedged and were not modified.

**Presentation notebook** — historical presentation artifact,
**presentation-only, not authoritative**; may contain older or
presentation-specific language, preserved verbatim. No kernelspec recorded in
its metadata; language_info 3.12.3. It recomputes nothing: it only loads the
two result CSVs below and plots.

## Required result dependencies (loaded from disk)

| Path | SHA-256 | Needed by |
|---|---|---|
| `experiments-analysis/results/results_mmd_distribution_distance_corrected.csv` | `28cf3165…818f4409` | both notebooks (final-run conclusions) |
| `experiments-analysis/results/results_mmd_time_sliced.csv` | `dd84fcc8…2659db5c8cbf2`* | both notebooks (time-sliced diagnostic) |
| `experiments-analysis/results/results_mmd_distribution_distance_corrected_10k.csv` | `6c7dce56…67bc18498`* | supporting evidence (10k predecessor run) |

\* Full hashes in the JSON manifest.

## Figure dependencies

The 20 PNGs in `experiments-analysis/figures/` are **outputs** of the
corrected notebook (written via `savefig`), preserved in git as evidence but
not inputs to either notebook; they are not hash-pinned.

## Module dependencies

The corrected notebook imports the legacy flat modules **`datasets` and
`diffusion`** from the repository root (plus `torch`, `numpy`, `pandas`,
`matplotlib`). These modules must remain importable from the repo root for
the notebook to be re-runnable; the migration therefore retains them (see
`docs/MIGRATION_REPORT.md`), and `datasets.py`, `diffusion.py`, and
`models.py` (imported by `diffusion.py`) are hash-pinned in the JSON manifest
so a change to them fails the preservation test rather than silently breaking
re-runnability. The presentation notebook needs only pandas/numpy/matplotlib.

## Known limitations

- The scratch resume files `results/*_checkpoint.csv` referenced by the
  corrected notebook are gitignored working files; only
  `results_mmd_distribution_distance_full_checkpoint.csv` exists on disk
  (untracked) and is not required to support the recorded conclusions.
- Environment at recording time: macOS (Darwin 25.5.0), uv 0.11.28; the
  notebook's own kernel recorded Python 3.14.6 (the environment it was last
  executed with), which predates the repo's current 3.12-pinned uv
  environment — re-running it today would use a different interpreter than
  the recorded outputs.
- Hashes pin byte-exact content; any intentional future edit to these files
  must update both manifests and the preservation test in the same change.
