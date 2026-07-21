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

## Residual data products (removed in the follow-up retirement)

These were not caught by the initial Phase 3B deletion at
`177fd8f84b0b02b799be057259ff74318c8761d7`; they are the same MCMC outputs at
a different parameter (`N=10000` instead of `N=20000`) and its dependent
figures/PDFs. Removed by the follow-up commit "chore: complete Hopfield and
DMFT retirement".

| Path | Size (bytes) | Git blob | SHA-256 | Producer |
|---|---|---|---|---|
| `data/hopfield_T0_mcmc_N10000_S10_seed0.npz` | 91402 | `832e29f681b068734029ef87523c6198e1a18973` | `ea56ad40ed4ae78fe0d2503e82be4a3d611fb627d81ab5e4a5ac1b832c793ba` | `mcmc_hopfield.py` |
| `data/hopfield_T001_mcmc_N10000_S10_seed0.npz` | 46084 | `40639ff87dcbda6fc1ced2dafb4324435b73969e` | `b918003eae6a7a39c00026d7c300f8477ae41e1bfca94ed1212f8b18db99553` | `mcmc_hopfield.py` |

## Residual figures (removed in the follow-up retirement)

| Path | Size (bytes) | Git blob | SHA-256 | Producer |
|---|---|---|---|---|
| `notes/plots/hopfield_T001_m_vs_t_mcmc_pattern.png` | 201320 | `d106c9ae7df46d3d9b86b40b81f5784eaa25870d` | `dcd1d8dabb3b78c7d509ce909149e4bfa64365bba7cd6028e150a815e7ce74c` | `plot_hopfield.py` (embedded in deleted `notes_hopfield.typ`) |
| `notes/plots/hopfield_T001_m_vs_t_mcmc_random.png` | 198668 | `0848d897e2073711c7c704204531a1cd2c4bad16` | `006d990791816dfc03e7d5cdaf1ad9b2ab17f365439e510876d293f51c1e7e1` | `plot_hopfield.py` (embedded in deleted `notes_hopfield.typ`) |
| `notes/plots/hopfield_T0_m_vs_t_mcmc_pattern.png` | 200318 | `8438b81aa32410ecdaacfd2f218bab1cb8b27399` | `00110d3e2ea02d3e801957b46556e7499a0486238a1f41355a232220ab24551` | `plot_hopfield.py` (produced, never embedded) |
| `notes/plots/hopfield_T0_m_vs_t_mcmc_random.png` | 199782 | `9184422005760cbb547975f9e0214cd026b91830` | `e0bc78d54662a140185479c781eade0dc0178719a3c1ad6a096dc61b721bf32` | `plot_hopfield.py` (produced, never embedded) |
| `notes/plots/hopfield_T0_sweeps_mcmc_pattern.png` | 73589 | `afe0b7aeb0ac98702b7237d7fbba91e62ee054dd` | `fae5917bcfca0a3207e5991ba1c47e2a1c8427cfa2157582cb18b81d181b8e9` | `plot_hopfield.py` (produced, never embedded) |
| `notes/plots/hopfield_T0_sweeps_mcmc_random.png` | 83732 | `e69cf0ffc3f376bdbd8b5a47cb6c1c10e88a90f8` | `93d037cb84f5e5b07eda90f88d933f09cde7805890289d3107e3cee428a910d` | `plot_hopfield.py` (produced, never embedded) |

## Residual compiled PDFs (removed in the follow-up retirement)

Typst-compiled output of the two `.typ` sources deleted at
`177fd8f84b0b02b799be057259ff74318c8761d7`; the compiled PDFs were left
behind as orphaned artifacts.

| Path | Size (bytes) | Git blob | SHA-256 | Source |
|---|---|---|---|---|
| `notes/notes_hopfield.pdf` | 826081 | `aa294b43d5ef86691742c9f263d6d3484ff56375` | `aa6592a3d85ec3406f36c7be1ae19f3d3685e828438d0c147c33cbcb4317390` | compiled from deleted `notes/notes_hopfield.typ` |
| `notes/notes_dmft_masked_hopfield.pdf` | 627791 | `b189cc2579bbd3b9cae1dc2704e51321f3c62d4d` | `da2985e1a32caa78add9deeabd6cbae600215fda760978224cb1e1602b934ab` | compiled from deleted `notes/notes_dmft_masked_hopfield.typ` |

## Why these are orphaned

All ten paths above are outputs (data, figures, or compiled PDFs) whose only
source (`src-hopfield/*.py` or the two `.typ` notes) was already deleted at
`177fd8f84b0b02b799be057259ff74318c8761d7`. No retained script, notebook, or
protected artifact reads or references any of them; the two "pattern"
`T001`/`T0` figures were embedded only in the now-deleted `notes_hopfield.typ`
and are consequently orphaned, and the remaining four figures were generated
but never embedded anywhere.

## Recovery (residual files)

```bash
git show ed42906cffd0b2b5989eb53e46f00ca6cdde4171:data/hopfield_T0_mcmc_N10000_S10_seed0.npz
git checkout ed42906cffd0b2b5989eb53e46f00ca6cdde4171 -- <path>
```

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
