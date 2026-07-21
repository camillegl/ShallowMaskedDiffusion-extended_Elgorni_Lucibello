# Repository Audit

Snapshot at commit `2e2db70` (2026-07-20). Status column: **A** = authoritative source of
truth for its domain, **N** = non-authoritative (report, artifact, or legacy). Migration
actions are recommendations only — **nothing was deleted, renamed, or behaviorally modified
in this audit**.

| Path | Purpose | Status | Scientific relevance | Duplication | Action | Evidence / risk |
|---|---|---|---|---|---|---|
| `diffusion.py` | MaskedDiffusion Lightning module: loss, samplers | A (behavior) | Core | Samplers re-implemented in several notebooks | keep; later extract into a package | Read fully; dead import `from turtle import pd` at line 2 (harmless). Risk: low |
| `models.py` | Backbones: Linear, RandomFeature, Tensor | A (behavior) | Core | — | keep; later extract | Read fully. Risk: low |
| `datasets.py` | UniformIsing, RandomFeatures (hidden-manifold), BinarizedMNIST | A (behavior) | Core | Hidden-manifold generation duplicated in notebooks | keep; later extract | `sign(0)=0` not patched here (see UPSTREAM_DISCREPANCIES D8). Risk: medium |
| `train.py` | Training CLI (argparse) | A (behavior) | Core | — | keep | `--alpha` = M/L legacy convention (see NOTATION). Risk: low |
| `experiments-analysis/utils.py` | Checkpoint loading, overlaps | A (behavior) | High | — | keep | Path-relative imports of flat modules. Risk: medium on migration |
| `experiments-analysis/run_uturn_experiments.py` | U-turn batch runner | A (behavior) | High | — | keep | Risk: low |
| `experiments-analysis/analysis_mmd_distribution_distance_corrected.ipynb` | Newest hidden-manifold MMD study (FINAL_RUN) | N (report) — but currently the **only** home of the MMD estimator + protocol | High | Supersedes `analysis_mmd_distribution_distance.ipynb`; the MMD estimator and sign(0) patch exist only in notebooks (the generator itself is `datasets.py`) | extract MMD logic to a module later; keep notebook as report | Logic trapped in notebook. Risk: high (silent divergence) |
| `experiments-analysis/analysis_mmd_distribution_distance.ipynb` | Superseded MMD study | N | Historical | Superseded by `_corrected` | archive later | Risk: confusion if used |
| `experiments-analysis/mmd_results_presentation_1.ipynb` | Presentation of MMD results | N | Medium | Re-derives from same data | keep as report | Not independently verified |
| `experiments-analysis/analysis*.ipynb` (others) | Legacy uniform/MNIST/U-turn analyses | N | Medium | Notebook-local sampler copies | keep as reports | Embedded outputs up to 1.5 MB each |
| `experiments-analysis/*.csv`, `*.png`, `figures/`, `results/` | Generated experiment artifacts committed as source | N | Evidence for figures | `experiments.csv` exists at root AND here; `results_mmd_distribution_distance.csv` exists at two paths | keep for now; add manifests later | No manifest links artifacts to commit+args. Risk: medium |
| `experiments.csv` (root) | Duplicate of `experiments-analysis/experiments.csv` | N | Low | Yes | resolve duplication later | Canonical copy undetermined |
| `notes/notes_memorization.typ` | Upstream theory: masked objective, linear score, replica calc | A (upstream equations) | Core | — | keep | The de-facto "paper"; see PROVENANCE |
| `notes/notes_hiddenmanifold.typ` | Hidden-manifold entropy theory (replica) | A (extension theory) | Core | Overloads `α` as Rényi order (see NOTATION) | keep | Computes entropy of P_F; no MMD prediction |
| `notes/notes_hopfield.typ`, `src-hopfield/` | Clamped-Hopfield side-track (RS theory + MCMC) | A (side-track) at audit time; **deleted from `guthlac` at `177fd8f84b0b02b799be057259ff74318c8761d7` (branch only, not merged to `main`)** | Medium | — | archived — see `docs/archive/HOPFIELD_DMFT_ARCHIVE.md`, `docs/archive/HOPFIELD_DMFT_RETIREMENT.md` | Self-contained; recoverable from `phase2-hidden-manifold-foundation` or `ed42906cffd0b2b5989eb53e46f00ca6cdde4171` |
| `notes/*.pdf`, `paper/plots/`, `notes/plots/` | Compiled/generated figures | N | Evidence | — | keep | Non-ASCII filenames (λ) are a cross-platform hazard |
| `paper/main-neuralnetworks.typ` | Manuscript stub (no equations) | **deleted in Phase 3A, committed at `ed42906cffd0b2b5989eb53e46f00ca6cdde4171` on `main`** | Low today | — | deleted | Was not a source of equations; upstream/extension theory remain in `notes/notes_memorization.typ` / `notes/notes_hiddenmanifold.typ`. See `docs/LEGACY_SCIENTIFIC_INDEX.md`, `docs/FINAL_REPOSITORY_MAP.md` |
| `data/*.npz` | Cached Hopfield theory/MCMC results | N (generated, used as input) | Medium | — | keep; add manifests later | Provenance is filename-encoded only |
| `julia-code/` | Reference Julia implementations (incl. hiddenmanifold) | N (reference) | Medium | Parallel implementation of sign-channel data | keep | Not traced in this audit |
| `scripts/*.sh` | SLURM + local run scripts | A (ops) | Low | — | `train-cpu.sh`, `train-gpu.sh`, `train-cpu-mnist.sh`, `span-slurm.sh`, `uturn.sh` **deleted in Phase 3A, committed at `ed42906cffd0b2b5989eb53e46f00ca6cdde4171` on `main`**; `reproduce_smoke.sh` and `validate_reference_artifacts.sh` kept | README points to nonexistent `slurm-jobs/` |
| `pyproject.toml` | Dependency spec | A (deps) | — | `hydra-core` declared but never imported | drop hydra later | Placeholder description; no build-system → project not installable |
| `uv.lock` | Lockfile (546 KB, on disk) | N — **gitignored**, so collaborators don't get it | — | — | decide: commit it (recommended) | Reproducibility gap |
| `AGENTS.md` | Legacy agent instructions | N (superseded by `CLAUDE.md` + `docs/`) | — | Overlaps CLAUDE.md | rewrite/trim later | Stale: claims Python 3.11 and a `.python-version` file; neither is true (pyproject requires ≥3.12; no such file) |
| `README.md` | Human readme | N | — | — | fix later | Points to nonexistent `slurm-jobs/` |
| `__pycache__/` (root, on disk) | Bytecode | N | — | — | ignore | Untracked |
| `.claude/` | Claude Code config, rules, agents, skills | A (agent ops) | — | — | keep | Created/extended by this audit |
| `docs/` | Source-of-truth documents | A | — | — | keep | Created by this audit |

## Cross-cutting findings

- **No test suite and no CI** exist (`tests/`, `.github/` absent). Correctness is currently
  verified only through notebooks and logged metrics.
- **Notebook-trapped logic** is the main migration hazard: the MMD estimator, the sign-channel
  patch (`0 → +1`), and per-notebook sampler copies exist only inside `.ipynb` files. (The
  hidden-manifold *generator* is properly in `datasets.py` and is reused by the notebooks.)
- **Flat-module imports** (`from diffusion import ...` relying on cwd) will break under any
  `src/` or packaging migration.
- **Git hygiene**: broken ref `refs/heads/main 2`, stale branches, uncommitted deletion of
  `rsync-logs.sh`.
