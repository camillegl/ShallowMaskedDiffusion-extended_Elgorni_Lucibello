# Hopfield and DMFT retirement record

This change removes the isolated Hopfield/DMFT side study from the `guthlac` branch after its scientific role and provenance were recorded in `docs/archive/HOPFIELD_DMFT_ARCHIVE.md`.

## Removed sources and notes

- `src-hopfield/hopfield_saddle_point.py`
- `src-hopfield/mcmc_hopfield.py`
- `src-hopfield/plot_hopfield.py`
- `notes/notes_hopfield.typ`
- `notes/notes_dmft_masked_hopfield.typ`

## Removed recorded outputs

- `data/hopfield_T0_theory.npz`
- `data/hopfield_T0_mcmc_N20000_S10_seed0.npz`
- `data/hopfield_T001_mcmc_N20000_S10_seed0.npz`
- `notes/plots/hopfield_m_vs_t_beta10.png`
- `notes/plots/hopfield_phase_diagram_beta10.png`
- `notes/plots/hopfield_T0_m_vs_t.png`
- `notes/plots/hopfield_T0_phase_diagram.png`

## Scope and interpretation

This is repository retirement, not scientific validation. It does not establish any claim about the hidden-manifold model, the active package, replica symmetry, or the exploratory DMFT closure. The active implementation and all protected MMD artifacts are untouched.

## Recovery

Every removed path is recoverable from either:

```bash
git checkout phase2-hidden-manifold-foundation -- <path>
git checkout ed42906cffd0b2b5989eb53e46f00ca6cdde4171 -- <path>
```

The archive document records the Git blob identifiers for the removed scientific files and outputs.

## Known gap (found during Phase 3B1 verification, not fixed by this commit)

This retirement did not remove every Hopfield/DMFT-related tracked file.
Left behind on `guthlac`:

- `data/hopfield_T0_mcmc_N10000_S10_seed0.npz`,
  `data/hopfield_T001_mcmc_N10000_S10_seed0.npz` — stale-parameter data
  (`N=10000`, not the current script's `N=20000`).
- `notes/plots/hopfield_T001_m_vs_t_mcmc_pattern.png`,
  `notes/plots/hopfield_T001_m_vs_t_mcmc_random.png` — these were embedded in
  the now-deleted `notes/notes_hopfield.typ` and are consequently orphaned.
- `notes/plots/hopfield_T0_m_vs_t_mcmc_pattern.png`,
  `notes/plots/hopfield_T0_m_vs_t_mcmc_random.png`,
  `notes/plots/hopfield_T0_sweeps_mcmc_pattern.png`,
  `notes/plots/hopfield_T0_sweeps_mcmc_random.png` — produced but never
  embedded in either note.
- `notes/notes_hopfield.pdf`, `notes/notes_dmft_masked_hopfield.pdf` —
  compiled output of the two deleted `.typ` sources, now orphaned.

Resolving this is left to a follow-up commit; it is not performed here.
