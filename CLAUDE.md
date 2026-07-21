# ShallowMaskedDiffusion — project instructions

Research codebase studying shallow (linear) masked diffusion models as associative
memories: memorization vs. generalization on discrete binary data, extended to a
hidden-manifold (random-features, sign-channel) teacher. Authors: Filippo Elgorni, Carlo
Lucibello (Bocconi); hidden-manifold extension layered on top.

## Source-of-truth hierarchy

1. `docs/RESEARCH_SPEC.md` — scientific contract for the hidden-manifold extension.
2. `docs/NOTATION.md` — canonical notation and naming rules.
3. `docs/ORIGINAL_ARCHITECTURE.md` — the upstream masked-diffusion model as implemented.
4. `docs/EQUATION_TO_CODE_MAP.md` — equation → implementation traceability.
5. Executable code and regression tests.
6. Notebooks — reports and exploration only, never the sole home of essential logic.

Also: `docs/UPSTREAM_DISCREPANCIES.md` (known paper/code/doc disagreements — record new ones
there, never resolve silently), `docs/REPO_AUDIT.md`, `docs/PROVENANCE.md`,
`docs/CLAUDE_CODE.md` (agent tooling). `AGENTS.md` is legacy and partially stale (e.g., its
Python-3.11 claim is wrong); where it conflicts with `docs/`, `docs/` wins.

## Notation (essentials — full rules in docs/NOTATION.md)

- `D` latent dim, `N` visible dim (legacy code calls it `L`), `M` train size;
  `γ = N/D = aspect_ratio`, `α = M/D = sample_ratio`, `M/N = visible_load` (derived only).
- **`alpha` is overloaded four ways** across the repo (legacy CLI M/N; notebooks M/D; Rényi
  order and replica index in the notes). Never reinterpret a stored value; convert
  explicitly: `legacy alpha = sample_ratio / aspect_ratio`. Bare `alpha`/`gamma` identifiers
  are banned in new Python interfaces.

## Scientific invariants

- Teacher: `F ∈ R^{N×D}`, `F_ia ~ N(0,1/D)` quenched per repeat; `z ~ N(0,I_D)` fresh per
  sample; `x = sign(Fz)`, `sign(0) := +1`. The finite-F law `P_F`, the empirical train set,
  and the disorder average `E_F P_F` are three different objects — always say which.
- Generated samples follow a sampler-indexed terminal law `P_{θ,A,k}`; never "the model
  distribution", never "exact ancestral sampling" (joint-law consistency is unproven).
- A falling MMD supports "approaches the finite-F target under this diagnostic" — never
  "learns the distribution". The persistent gap above the noise floor is an open empirical
  observation, not a capacity threshold.
- `V ≡ 0` (frozen mask weights) is an experimental restriction, not derived for
  hidden-manifold data. Do not transfer uniform-data symmetry arguments to fixed `F`.

## Repository conventions

- Package manager: `uv` only. Python 3.12 (`.python-version`); `uv.lock` is un-gitignored
  and must stay in version control.
- Active implementation: `src/maskeddiffusion/` (typed dataclasses + TOML config, direct
  PyTorch loop — see `docs/adr/`). Legacy flat modules `diffusion.py`, `models.py`, and
  `datasets.py` are frozen compatibility modules required by the protected corrected MMD
  notebook (`docs/REFERENCE_RESULTS_MANIFEST.md`, `docs/FROZEN_LEGACY_RUNTIME.md`) and must
  stay importable. `train.py` was a separate, deprecated historical CLI, never hash-pinned
  and never a protected-notebook dependency; it was retired (deleted) in the Phase 3
  retirement, now merged to `main`, once the old Julia scripts and superseded notebooks
  that depended on it were also retired (see `docs/archive/JULIA_LEGACY_ARCHIVE.md`,
  `docs/archive/HISTORICAL_NOTEBOOKS_ARCHIVE.md`).
- Protected artifacts (never modify/rerun/strip):
  `experiments-analysis/analysis_mmd_distribution_distance_corrected.ipynb`,
  `experiments-analysis/mmd_results_presentation_1.ipynb`, and the result CSVs pinned in
  `artifacts/reference/mmd_final_run/manifest.json`.
- Theory notes in `notes/` (Typst) — `notes/notes_memorization.typ` is the authoritative
  upstream theory; `notes/notes_hiddenmanifold.typ` is the authoritative hidden-manifold
  extension theory. `paper/main-neuralnetworks.typ` was a non-authoritative stub, removed in
  Phase 3A; it remains recoverable from the stable tag `phase2-hidden-manifold-foundation`
  (see `docs/LEGACY_SCIENTIFIC_INDEX.md`). Rules in `.claude/rules/` apply.

## Verified commands (ran 2026-07-20, Phase 2; re-verified after the Phase 3 merge to `main`)

- `uv run pytest -q` — full test suite (243 tests as of Phase 4C U-turn; was 200 at
  Phase 4B provenance hardening PR #3, 125 at Phase 2, 128 after Phase 3's
  CI-hardening tests — see `docs/MIGRATION_REPORT.md`).
- `scripts/reproduce_smoke.sh` — tiny CPU end-to-end run; integration check only.
- `scripts/validate_reference_artifacts.sh` — verify protected-artifact hashes.
- `uv run maskeddiffusion-train --help` (likewise `-sample`, `-evaluate`,
  `-uturn`, `-validate-artifact`).
- `train.py`, the legacy CLI (`--alpha` meaning M/L), was retired (deleted) in Phase 3,
  now merged to `main` — it no longer exists on any current branch; see
  `docs/MIGRATION_REPORT.md`'s Phase 3F section and `docs/archive/JULIA_LEGACY_ARCHIVE.md`.
- `.github/workflows/ci.yml` (on `main` since the Phase 3 merge) runs the equivalent of the
  commands above plus the five CLI `--help` checks and a committed-range whitespace check,
  on every push/PR to `main` and `guthlac`.

## Forbidden actions

- Committing, branching, or pushing without explicit user instruction.
- Deleting/regenerating committed numerical results (`data/*.npz`,
  `experiments-analysis/*.csv|results/`) or notes PDFs.
- Launching training runs, MMD campaigns, or notebook sweeps unless asked.
- Renaming or reinterpreting legacy `alpha`/`L` values; introducing new bare
  `alpha`/`gamma` identifiers.
- Stating scientific claims beyond the permitted vocabulary of `docs/RESEARCH_SPEC.md`.

## Completion requirements

Before declaring a change done: run the verified import check; if scientific code or docs
changed, confirm `docs/ORIGINAL_ARCHITECTURE.md` / `docs/EQUATION_TO_CODE_MAP.md` still
match, and log any new disagreement in `docs/UPSTREAM_DISCREPANCIES.md`. Use the
`validate-repository-change` skill for a structured check. Report failures plainly; never
report a skipped check as passed.
