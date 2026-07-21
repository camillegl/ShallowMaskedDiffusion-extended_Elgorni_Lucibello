# Phase 3 branch report

Branch: `guthlac`

Baseline: `ed42906cffd0b2b5989eb53e46f00ca6cdde4171` (`main` after Phase 3A).

## Implemented in this branch

- Fast-forwarded `guthlac` to the Phase 3A baseline without changing `main`.
- Commit `513fe7755d9dd65718da6b7720264033a0fd61a3` ("docs: archive Hopfield
  and DMFT side study"): added `docs/archive/HOPFIELD_DMFT_ARCHIVE.md`, which
  records the scientific role, Git blob identifiers, data products, figures,
  dependency implications, limitations, and recovery commands for the
  Hopfield/DMFT side study; corrected the stale Phase 3A status in
  `docs/FROZEN_LEGACY_RUNTIME.md`; corrected `docs/FINAL_REPOSITORY_MAP.md`
  so it distinguishes completed Phase 3A work from pending retirement
  categories. No deletion in this commit.
- Commit `177fd8f84b0b02b799be057259ff74318c8761d7` ("chore: retire Hopfield
  and DMFT side study"): deleted the three `src-hopfield/*.py` files, both
  Hopfield/DMFT theory notes, three current-parameter `data/hopfield_*.npz`
  files, and four `notes/plots/hopfield_*.png` figures; added
  `docs/archive/HOPFIELD_DMFT_RETIREMENT.md` recording exactly what was
  removed and how to recover it. **This step (item 1 in the "destructive
  steps" list below, as originally planned) has now happened** — the
  Hopfield/DMFT source and artifact retirement is complete on this branch,
  though not yet reviewed/merged, and a Phase 3B1 verification pass found it
  left several related tracked files behind (two stale-parameter `.npz`
  files, six additional figures, two compiled PDFs) — see
  `docs/MIGRATION_REPORT.md`'s Phase 3B section and
  `docs/LEGACY_SCIENTIFIC_INDEX.md` for the exact list.

## Deliberately not performed on this branch

No protected artifact, frozen compatibility module, active numerical code,
dependency declaration, or lockfile is deleted or modified by either commit.
`main` is unaffected; this retirement exists only on `guthlac` pending
review and a decision on whether/how to merge it.

The following destructive steps remain separate, not-yet-started commits:

1. ~~reviewed Hopfield/DMFT source and artifact retirement~~ — **done** in
   `177fd8f84b0b02b799be057259ff74318c8761d7`, pending review (this document)
   and merge decision; incomplete in scope (see above).
2. obsolete Julia inventory and retirement;
3. historical notebook and utility retirement;
4. non-protected result cleanup;
5. `train.py` and dependency retirement;
6. CI and final repository hardening.

This separation is intentional: each category is independently reviewable
and reversible before being merged into `main`.
