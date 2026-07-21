# ShallowMaskedDiffusion — hidden-manifold extension

Shallow (linear) masked diffusion models as associative memories: how do they
memorize vs. generalize when trained on data from a hidden low-dimensional
manifold? Original study by Filippo Elgorni and Carlo Lucibello (Bocconi
University); this repository extends it to a random-features "hidden
manifold" teacher.

## Scientific setup

Teacher (quenched per repeat):

- `F ∈ R^{N×D}`, entries `F_ia ~ N(0, 1/D)`, sampled once and held fixed;
- samples `x = sign(F z)` with `z ~ N(0, I_D)` fresh per example, and the
  convention `sign(0) := +1`, so `x ∈ {−1,+1}^N`.

Dimensions and ratios (see `docs/NOTATION.md` — bare `alpha` is banned in
active code because it historically meant different things):

| name | symbol | meaning |
|---|---|---|
| `latent_dim` | D | latent dimension |
| `visible_dim` | N | data dimension (`round(aspect_ratio·D)`) |
| `train_size` | M | training samples (`round(sample_ratio·D)`) |
| `aspect_ratio` | γ = N/D | teacher geometry |
| `sample_ratio` | α = M/D | training load |
| `visible_load` | M/N | derived metadata only |

A linear masked score is trained with a masked-BCE objective; generated
samples follow a **sampler-indexed terminal law** `P_{θ,A,k}` (the trained
single-site conditionals are not known to form a coherent joint law, and no
sampler here is ancestral sampling). Distributional agreement is measured by
MMD against fresh samples from the same fixed F — an MMD decrease supports
"approaches the finite-F target under this diagnostic," nothing stronger.
Full contract: `docs/RESEARCH_SPEC.md`.

## Installation

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12 (pinned in
`.python-version`; dependency resolution pinned in `uv.lock`):

```
uv sync
```

To also install interactive notebook tooling (`jupyterlab`, needed to open
the protected analysis notebooks, not required for tests or the CLIs):

```
uv sync --group analysis
```

CI (`.github/workflows/ci.yml`, on `main` since the Phase 3 merge) runs
`uv sync --frozen`, imports `maskeddiffusion`, ruff check/format, mypy, the
full test suite, protected-artifact validation, and the five CLI `--help`
checks on every push/PR to `main` and `guthlac`.

## Usage

Smoke check (tiny CPU run, integration only — never interpret scientifically):

```
scripts/reproduce_smoke.sh
```

Train / sample / evaluate / validate (all take `--config <toml> --output
<dir> --device cpu|cuda|mps|auto --dry-run`):

```
uv run maskeddiffusion-train    --config configs/smoke/smoke.toml --output runs/demo
uv run maskeddiffusion-sample   --config configs/smoke/smoke.toml --output runs/demo-samples \
       --checkpoint runs/demo/checkpoints/final.pt --n-samples 100
uv run maskeddiffusion-evaluate --config configs/smoke/smoke.toml --output runs/demo-eval \
       --checkpoint runs/demo/checkpoints/final.pt --teacher runs/demo/teacher.pt \
       --samples runs/demo-samples
uv run maskeddiffusion-uturn    --config configs/smoke/smoke.toml --output runs/demo-uturn \
       --checkpoint runs/demo/checkpoints/final.pt --teacher runs/demo/teacher.pt \
       --t-values 0.0 0.25 0.5 0.75 --n-examples 8
uv run maskeddiffusion-validate-artifact runs/demo
```

Every run writes a self-describing artifact directory (`manifest.json`,
`resolved_config.json`, `metrics.jsonl`, `summary.json`; ADR 003). Verify the
protected reference results: `scripts/validate_reference_artifacts.sh`. Run
the test suite: `uv run pytest -q`.

## Repository structure

- `src/maskeddiffusion/` — the active implementation (teacher, masking,
  linear score, objectives, samplers, training, metrics, CLI).
- `tests/` — unit/property/integration/regression tests, including
  `tests/fixtures/original_architecture_v1/` which pins the legacy
  implementation's exact behavior.
- `docs/` — authoritative scientific and engineering documents
  (`RESEARCH_SPEC.md`, `NOTATION.md`, `ORIGINAL_ARCHITECTURE.md`, ADRs,
  migration and provenance records; see also `docs/FROZEN_LEGACY_RUNTIME.md`,
  `docs/LEGACY_SCIENTIFIC_INDEX.md`, and `docs/FINAL_REPOSITORY_MAP.md` for
  the legacy-retirement plan).
- `diffusion.py`, `models.py`, `datasets.py` — **frozen legacy** flat modules,
  kept only because the protected notebook imports them; superseded by the
  active package (see `docs/MIGRATION_REPORT.md`, `docs/FROZEN_LEGACY_RUNTIME.md`).
  `train.py`, the separate non-protected legacy CLI, was retired in the
  Phase 3 retirement (merged to `main`) once its historical consumers (old
  scripts, superseded notebooks) were also retired (see
  `docs/archive/JULIA_LEGACY_ARCHIVE.md`,
  `docs/archive/HISTORICAL_NOTEBOOKS_ARCHIVE.md`).
- `experiments-analysis/` — analysis notebooks and recorded results.
- `notes/`, `julia-code/` — theory notes and side studies. The upstream and
  hidden-manifold theory live in `notes/notes_memorization.typ` and
  `notes/notes_hiddenmanifold.typ` respectively; there is no separate
  manuscript directory (see `docs/LEGACY_SCIENTIFIC_INDEX.md`). The
  clamped-Hopfield/DMFT side study formerly at `src-hopfield/` and
  `notes/notes_hopfield.typ` / `notes/notes_dmft_masked_hopfield.typ` was
  retired in the Phase 3 retirement (merged to `main`); see
  `docs/archive/HOPFIELD_DMFT_ARCHIVE.md` and
  `docs/archive/HOPFIELD_DMFT_RETIREMENT.md` for the record and recovery
  commands.

## Reference results (protected)

Two notebooks are preserved verbatim as the record of the final MMD run
(hashes in `docs/REFERENCE_RESULTS_MANIFEST.md`, enforced by tests):

- `experiments-analysis/analysis_mmd_distribution_distance_corrected.ipynb` —
  the authoritative final-run record (100k-sample MMD, λ∈{4,8}).
- `experiments-analysis/mmd_results_presentation_1.ipynb` — a historical
  presentation artifact; **presentation-only, not a source of truth**.

The reference results were produced by the legacy implementation via those
notebooks. The new package has **not** rerun the 100k final experiment; its
MMD implementation is verified equivalent to the notebook's on small inputs
(`tests/regression/test_mmd_notebook_equivalence.py`), no more.

## Claim limitations

Finite-dimensional empirical evidence only. No capacity threshold or
asymptotic statement is established; the observed persistent Model-vs-True
MMD gap above the noise floor is an open question. `V ≡ 0` (frozen mask
channel) in recorded experiments is a modeling restriction, not a derived
symmetry (docs/UPSTREAM_DISCREPANCIES.md, D7).

## Citation

Original study: Filippo Elgorni & Carlo Lucibello, *Shallow Masked Diffusion*
(theory notes in `notes/notes_memorization.typ`; no manuscript directory
exists in this repository at present). Hidden-manifold extension: this
repository, theory in `notes/notes_hiddenmanifold.typ`
(`https://github.com/camillegl/ShallowMaskedDiffusion-extended_Elgorni_Lucibello`).
