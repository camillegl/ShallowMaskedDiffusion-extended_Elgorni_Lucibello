# Julia legacy implementation archive

Status: Phase 3 archival record, `guthlac` branch only. This document
inventories the tracked Julia code under `julia-code/` at the point of
retirement, distinguishing what was removed from what is retained.

## `julia-code/hiddenmanifold/` — retained, outside retirement scope

This directory is **not** touched by this retirement and remains permanently
part of the repository.

- `SignChannel.jl` — Bayes-optimal replica-symmetric saddle-point solver for
  the **Shannon entropy** of the hidden-manifold output distribution under
  the noiseless sign channel `P_out(y|x) = δ_{y, sign(x)}`. Implements the
  saddle equations from `notes/notes_hiddenmanifold.typ` §4 (`q̂ = γ I(q)`,
  the Fisher-information integral `I(q)`, and the entropy functional `s`).
- `SignChannelRenyi.jl` — replica-symmetric **quenched Rényi entropy** of
  order `α = n+1` for the same sign-channel setting, implementing
  `notes/notes_hiddenmanifold.typ` §4.3 (the `q₀=0` reduction of the general
  Rényi saddle system to a single scalar equation for `q₁`).
- `data/entropy_sign_renyi_alpha*.csv`, `data/entropy_sign_shannon.csv` —
  computed entropy sweeps consumed by `scripts/plot_entropy.jl`.
- `plots/entropy_sign_renyi_all.pdf` — rendered sweep figure.
- `scripts/sweep_renyi.jl`, `scripts/sweep_shannon.jl`, `scripts/plot_entropy.jl`
  — drivers for the above.

These solvers compute the disorder-averaged (Rényi order `α`, distinct from
the sample-ratio `α` used elsewhere — see `docs/NOTATION.md`) entropy of the
teacher's output law under the hidden-manifold sign channel; they are the
active Julia-side counterpart to the hidden-manifold research direction and
are explicitly excluded from all retirement categories below.

## `julia-code/SP/` — retired (original/uniform-data saddle-point implementation)

**Scientific role.** A self-contained replica-symmetric saddle-point solver
and ODE integrator for the **original (uniform-data) masked-diffusion
model** — not the hidden-manifold extension. It solves for order parameters
`q, δq` as a function of `α = M/L` (Hopfield-style sample ratio, local to
this Julia code — do not confuse with the active package's
`sample_ratio = M/D`) and integrates a backward ODE for the retrieval
overlap `m(t)` along the reverse diffusion trajectory. This is **not
equivalent** to the hidden-manifold entropy solvers retained in
`julia-code/hiddenmanifold/`: it targets a different data model (uniform
Rademacher/Ising patterns, not a random-feature/sign-channel teacher) and a
different observable (retrieval overlap trajectory vs. entropy).

| Path | Size (bytes) | Git blob | SHA-256 | Role |
|---|---|---|---|---|
| `julia-code/SP/.gitignore` | 12 | `d629dcae41b2cc0142f3119327ce609f12d2cdcf` | `53102a6465567af5ff5f12d2e8d515bfbd3466461b269a05c61dfad2369e613` | ignore rules |
| `julia-code/SP/Manifest.toml` | 117721 | `0e2934a4c0cf6a29e5d4accb7144cb5ee6cb24e1` | `6850b0b38392f37d00e4f8d978bebfa529ecea16b3880d21f4e4572c02e531e` | resolved Julia environment |
| `julia-code/SP/Project.toml` | 1275 | `1832a6353f1043a527ea0b2c783219ccb095d58d` | `a1a5413655ac06ed50030637df0062e7d20f97952a0404ead2ebe6fc3aad799` | declared Julia dependencies |
| `julia-code/SP/common.jl` | 10747 | `f70acf8e106a3bd3fe3d9c0b9c09334f5ae8f607` | `46e5131b08dab999437c07e10d9a7315f02fbdd9a65d8a06e92d6b271105ff5` | special functions (`G`, `H`, `Φ`, log-erfc, etc.), shared helpers |
| `julia-code/SP/methods.jl` | 1199 | `1ef35199b924876e1748b931abf1bf23346c62eb` | `4aec9aa09329d08746250182e51e5111639c6c668da32edb2f607542d6bdb6` | Gaussian-quadrature integration helpers (`∫D`, `∫d`, truncated variants) |
| `julia-code/SP/MaskedDiffusion_SP.jl` | 8037 | `87a503643fda6eec25a38d5ad271186ed0a82c64` | `6e567773d3d74f304330230ba1cb421bcd46b3efee54b242f81a8b82292c8a` | RS saddle-point equations and fixed-point solver for `(q, δq)` given `(α, λ)` |
| `julia-code/SP/ODE.jl` | 8477 | `b9cc01797548cf78d9d30668868957f358cf70cf` | `65f41da66b04235cfe608b29a561ef52ec7fe4ebc157e97030433cb1603425` | backward-ODE integration of the retrieval-overlap trajectory `m(t)` |
| `julia-code/SP/helpers.jl` | 1466 | `d8e77f04634fa66f93d0c39e94702c80520fb970` | `c18f81a0789412471907b3bdc4cabedf91e09f14fb3961bec37e9210f8705e` | DataFrame/CSV read-write utilities for result files |
| `julia-code/SP/scripts/hebbian_datasets.jl` | 6319 | `6b0fad3c1d78cafa23578d02f2203b8b5924c112` | `87ad3aa9897b32e8606a52e6b0b6398dd5bfad12bc5ee0c41d63b4fc2820657` | Hebbian retrieval-accuracy formulas (`greedy`/`fair` algorithms) as a function of `(α, t)` |
| `julia-code/SP/scripts/plots.jl` | 35881 | `035a9c717e6e013b6d5559482a3ca47ce357b852` | `f8fd20daa8ceb6efa745efcdd43ff174808197f6fd81ad60858ac757e530567` | figure-generation driver combining saddle-point and ODE results |
| `julia-code/SP/scripts/solve_ode.jl` | 6178 | `1c9e023a0bef63b6a2e886102266410205752520` | `c608304a985a8376f5a96c8807941c62c92641972ae1a8ab46754d23076f905` | batch driver: solves the ODE for `m(t)` across a range of `α` values and writes results |
| `julia-code/SP/scripts/test_mu_solver.jl` | 2296 | `89046de32138bcec1f4149290422e8fdf2bad32e` | `cc0217ac74f6ed2e8fb9674e634886df40a4415c04567f9ce265a792f68129` | ad hoc benchmark/consistency check for the Newton solver used in the saddle-point equations |
| `julia-code/SP/notebooks/plots.ipynb` | 50628729 | `a893ef116e96973ed184b0cc6daddf5c0bfbec68` | `6849a1d43fbda82e0a454870fee66c4d37242e7d36685af1f6f8e27e721e3b` | "All paper plots" — replica-computation comparison figures for the uniform-data model (large notebook, embedded plot outputs) |
| `julia-code/SP/notebooks/variance_simulation.ipynb` | 601105 | `710b80fc22303e99e5e72731d93a55e169438063` | `9d4d9a388dbcf7043ec4e2d1b44405de6f40bbbafce483815d0bdc81d93c5b` | direct Monte Carlo simulation of the masked-diffusion update equation, used as an empirical cross-check against the replica prediction |

All of the above concern the **uniform-data model only** — they predate and
are independent of the hidden-manifold extension and are not consumed by
`src/maskeddiffusion/`, any retained notebook, or any protected artifact.

## `julia-code/old/` — retired (older Lux/uniform-data implementation)

**Scientific role.** An earlier neural-network-based (Lux.jl/Zygote/Enzyme)
training implementation of the masked-diffusion backbone on **uniform
data**, superseded first by later Julia work and ultimately by the active
Python package `src/maskeddiffusion/`. `DiffusionBackbone` is a
`LuxCore.AbstractLuxLayer` wrapping a masking-aware linear backbone trained
with `AdamW` via `Optimisers.jl`.

| Path | Size (bytes) | Git blob | SHA-256 | Role |
|---|---|---|---|---|
| `julia-code/old/Project.toml` | 751 | `f95a3c2f1cf601a3a1591d3e23586328a43e3c2e` | `92fbb7d8a58976de35750f625b68f9f2af8a10826b02f0f62e0dc6f5df3a7f` | declared dependencies (Lux, Flux, Zygote, NLopt, Optimization*) |
| `julia-code/old/Manifest.toml` | 59977 | `4071508fb2ed1d7ab38abfc5daa9fc96dfc5a556` | `01d6ee48d96a43a1abee50d2ddf8f477961ee82e2e41ee6be69f032cd345af` | resolved Julia environment |
| `julia-code/old/lux_train_uniform_data.jl` | 4096 | `f2959820091b7587576aa52b8548c61ed60b65ab` | `996477d49f1aa56419b1b843f6bcb46f167eeb8180c6a504be77d9dfe66642` | `DiffusionBackbone` layer definition and training loop on uniform `{-1,0,1}`-encoded data |

**Why treated as superseded.** This is a distinct, older code path from
`julia-code/SP/` (different toolchain — Lux/Zygote neural-network training
vs. saddle-point/ODE theory solving — and a different `Project.toml`/
`Manifest.toml` pair, i.e. a separate Julia environment). It predates the
active Python package's typed-dataclass/direct-PyTorch-loop architecture
(`docs/adr/`) that superseded it. There is some uncertainty about whether
its exact numerical behavior (optimizer schedule, initialization) was ever
formally cross-checked against a successor — no such cross-check is recorded
in this repository's history, so this archive records that gap rather than
asserting behavioral equivalence.

## Dependency implications

Julia dependencies (Lux, Flux, Zygote, Enzyme, DifferentialEquations,
Optimization*, PyCall, GSL, etc.) are declared entirely within
`julia-code/SP/Project.toml` and `julia-code/old/Project.toml`, which are
themselves deleted by this retirement. No `pyproject.toml` Python dependency
is implicated by this change — the Python dependency cleanup (`scipy`,
`numba`, `tensorboard`) is scoped separately in Phase 6 and traces to the
Hopfield/DMFT side study and other Python-only usage, not to this Julia
code.

## Recovery

Every removed path remains recoverable from either baseline:

```bash
git show phase2-hidden-manifold-foundation:<path>
git checkout phase2-hidden-manifold-foundation -- <path>
git show ed42906cffd0b2b5989eb53e46f00ca6cdde4171:<path>
git checkout ed42906cffd0b2b5989eb53e46f00ca6cdde4171 -- <path>
```

## Limitations

- This archive documents scientific/engineering role from reading source; it
  does not re-derive or re-verify the saddle-point equations or ODE
  integration against `notes/notes_memorization.typ`.
- Removing this code is repository maintenance, not scientific validation;
  it does not affect any claim about the hidden-manifold model or the active
  package.
- `julia-code/hiddenmanifold/` is unaffected; see
  `git diff --exit-code main..guthlac -- julia-code/hiddenmanifold` in
  `docs/PHASE3_BRANCH_REPORT.md` for the emptiness check.
