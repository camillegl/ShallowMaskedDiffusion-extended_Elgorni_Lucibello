# Hopfield and DMFT side-study archive

Status: Phase 3B archival record. This document describes a self-contained side study separate from the active hidden-manifold implementation in `src/maskeddiffusion/`. Archiving or later removing these files does not validate the active package and does not transfer the results below to the hidden-manifold teacher model.

## Scientific scope

The side study considers a classical Hopfield model with `N` Ising spins and `P = alpha N` independent random patterns. A fraction `1-t` of the spins is clamped to the first pattern and the remaining fraction `t` is free. Its main observable is the retrieval overlap with the first pattern. Hopfield `alpha = P/N` is local to this side study and must not be confused with the active package's `sample_ratio = M/D` or the legacy CLI's `alpha = M/N`.

The Python code contains a finite-temperature replica-symmetric saddle solver, a zero-temperature branch and spinodal calculation, plotting utilities, and finite-size Glauber and Metropolis-Hastings experiments. These calculations concern the clamped Hopfield model only.

`notes/notes_dmft_masked_hopfield.typ` studies an autoregressive/masked Hopfield setting with a Martin-Siggia-Rose generating-functional construction and partial overlaps. Its Gaussian/second-cumulant closure is an assumption requiring validation; it is not an exact DMFT solution.

## Source inventory at the pre-Phase-3B baseline

| Path | Git blob | Role |
|---|---|---|
| `src-hopfield/hopfield_saddle_point.py` | `fe3ca09cbfeb50ac7a47341c18ab3dd7a772aed2` | RS saddle equations, free energy, spinodal and theory cache |
| `src-hopfield/mcmc_hopfield.py` | `15e6aa546516e3ceee23b5a4b6f8a1267e9515af` | Glauber and Metropolis-Hastings finite-size checks |
| `src-hopfield/plot_hopfield.py` | `b132c74049823a20e1c43dddafa29c4d8eaf9123` | Theory sweeps and dedicated figures |
| `notes/notes_hopfield.typ` | `f724eb6478d7b1b1384886e472cb149a7da291aa` | Unique written conditional-Hopfield derivation |
| `notes/notes_dmft_masked_hopfield.typ` | `e579277544e9704edef06bf57f2c87a95412bac5` | Unique exploratory masked-Hopfield DMFT derivation |

## Recorded data products

| Path | Git blob | Producer |
|---|---|---|
| `data/hopfield_T0_theory.npz` | `2b67b648f16aaf564a49c407157b514e69b7feef` | `hopfield_saddle_point.py` |
| `data/hopfield_T0_mcmc_N20000_S10_seed0.npz` | `66817f58cd5abad378468374322ad98810464162` | `mcmc_hopfield.py` |
| `data/hopfield_T001_mcmc_N20000_S10_seed0.npz` | `26b34de300ff5aae788981ab0ee1ee7dfb187daf` | `mcmc_hopfield.py` |

The MCMC filenames encode `N=20000`, ten disorder samples and seed zero. This is weaker provenance than the active package's manifest-based artifact format.

## Recorded figures

| Path | Git blob | Producer |
|---|---|---|
| `notes/plots/hopfield_m_vs_t_beta10.png` | `b7326256cded9f88dbafc09f705f6889ee9a5d95` | `plot_hopfield.py` |
| `notes/plots/hopfield_phase_diagram_beta10.png` | `ab4001a41b9e25a5df66d672374e69a996ad2ee7` | `plot_hopfield.py` |
| `notes/plots/hopfield_T0_m_vs_t.png` | `551fdd9e2f35122f7dc480386fcf793dc8df0fa1` | `plot_hopfield.py` |
| `notes/plots/hopfield_T0_phase_diagram.png` | `45e215c2924bfe5f54fcb61f60af48b307f4195d` | `plot_hopfield.py` |

## Dependency implications

`numba` is used by the Hopfield MCMC kernels and `scipy` by the saddle solver. `matplotlib` is shared with retained notebooks. No dependency is removed in the archival step. Dependency removal must follow a repository-wide import scan after the side-study and historical notebooks are retired.

## Limitations

- The finite-size simulations do not validate replica symmetry.
- The DMFT second-cumulant closure is not exact.
- None of these calculations establishes a theorem or threshold for the hidden-manifold model.
- Ignored or untracked outputs are outside Git's archival guarantee.
- Removing the side study is repository maintenance, not scientific validation.

## Recovery

The complete side study remains recoverable from either baseline:

```bash
git show phase2-hidden-manifold-foundation:<path>
git checkout phase2-hidden-manifold-foundation -- <path>
git show ed42906cffd0b2b5989eb53e46f00ca6cdde4171:<path>
git checkout ed42906cffd0b2b5989eb53e46f00ca6cdde4171 -- <path>
```
