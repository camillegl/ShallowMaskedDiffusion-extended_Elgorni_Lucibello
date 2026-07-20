# Provenance

Audit date: 2026-07-20.

## Repository state at audit time

- Remote: `origin` → `https://github.com/camillegl/ShallowMaskedDiffusion-extended_Elgorni_Lucibello.git`
- Branch: `main`
- Commit: `2e2db70d35469318bbbeb2f3649c8586f1177200`
  ("Run FINAL_RUN Experiment A at 100k MMD samples (λ∈{4,8}); persistent gap above noise floor")
- Dirty state: one uncommitted deletion (`rsync-logs.sh`), left untouched by this audit.
  (During the audit session the user committed that deletion as `fea744d` "Remove unused
  rsync-logs.sh"; all audit findings refer to the code state at `2e2db70`, which `fea744d`
  does not alter.)
- Git anomalies (not fixed here): a broken ref file `.git/refs/heads/main 2` (name contains a
  space) emits a warning; several stale parallel branches exist (`main-merge-tmp`, `guthlac`,
  `claude/*`).

## Lineage

- The repository name and `AGENTS.md` identify this as an extension of *ShallowMaskedDiffusion*
  by Filippo Elgorni and Carlo Lucibello (Bocconi University). No upstream git remote for the
  parent repository is configured; parent lineage is known only from documentation, not from
  git history.
- The hidden-manifold extension (MMD experiments, `RandomFeaturesDataset` sign-channel data,
  `notes/notes_hiddenmanifold.typ`, `julia-code/hiddenmanifold/`) is layered on top of the
  original uniform-data associative-memory study.

## Paper source

- `paper/main-neuralnetworks.typ` exists but is a **stub** (title and outline only; abstract is
  "TODO"; it contains no equations). All authoritative upstream equations are in
  `notes/notes_memorization.typ`; the hidden-manifold theory is in
  `notes/notes_hiddenmanifold.typ`. Statements in `docs/ORIGINAL_ARCHITECTURE.md` that cite
  "the paper" therefore cite the notes, and this limitation is recorded rather than papered
  over. No quotations or equation numbers from an external published paper are used.

## What was extracted, and how

The following knowledge was extracted by direct reading of source at commit `2e2db70`:

- Objective, mask representation, linear score, regularization, and all samplers: `diffusion.py`,
  `models.py`, `train.py`, `datasets.py` (full reads; see `docs/ORIGINAL_ARCHITECTURE.md`).
- Hidden-manifold teacher, quenched-disorder protocol, MMD diagnostic:
  `experiments-analysis/analysis_mmd_distribution_distance_corrected.ipynb` (the newest
  hidden-manifold notebook — newest by mtime 2026-07-17, and matching the latest commits
  `2b4220c`/`2e2db70` about λ∈{4,8} and FINAL_RUN), `datasets.py`,
  `notes/notes_hiddenmanifold.typ`.
- Engineering state: `pyproject.toml`, `uv.lock` (on disk, gitignored), `.gitignore`, `scripts/`,
  git refs.

## Unverified

- `experiments-analysis/mmd_results_presentation_1.ipynb` figures were not independently
  re-derived (only cross-referenced).
- The role of `julia-code/hiddenmanifold/` relative to the Python MMD pipeline is assumed
  independent; not traced in detail.
- No training runs, MMD campaigns, or notebooks were executed during this audit; all findings
  are from static reading.
- Whether any archived experiment used `k>1`, `bias=True`, or unfrozen `V` was not exhaustively
  checked against stored artifacts.
