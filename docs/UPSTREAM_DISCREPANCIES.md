# Upstream discrepancies

Each disagreement between paper/notes, code, and documentation, recorded independently.
Status values: **resolved**, **intentionally preserved**, **open**. Resolution type:
**scientific** (changes what may be claimed) or **engineering** (naming/plumbing only).
References are at commit `2e2db70`.

## D1 — `alpha` conventions (M/N vs M/D vs Rényi order vs replica index)

- **Evidence.** `train.py:31,157`: `--alpha` = M/L (=M/N). Corrected MMD notebook: `alpha` =
  M/D, converts explicitly (`config.alpha = alpha/gamma`). `notes/notes_hiddenmanifold.typ:61,71`:
  `α` = Rényi order `n+1`; and `:96,132`: `α` also serves as the disorder-replica index.
  Four distinct meanings.
- **Consequence.** Mixing conventions silently rescales the load axis by γ; a reader of
  train.py + notebook together is off by a factor γ.
- **Resolution.** Contract names in `docs/NOTATION.md` (`sample_ratio`, `aspect_ratio`,
  `visible_load`); bare `alpha` banned in new interfaces; legacy values never reinterpreted.
- **Type.** Engineering (naming), with scientific stakes if confused.
- **Status.** Open (legacy CLI unchanged by design; new-code rule in force).

## D2 — `L` vs `N`

- **Evidence.** All upstream code uses `L` for the visible dimension; the hidden-manifold
  contract uses `N` (`datasets.py` uses `n_visible`).
- **Consequence.** Harmless if the equivalence is stated; confusing otherwise.
- **Resolution.** `docs/NOTATION.md` fixes `L ≡ N ≡ visible_dim`.
- **Type.** Engineering. **Status.** Resolved (documented).

## D3 — Runtime `1/√L` normalization

- **Evidence.** notes-mem:165 puts `1/√L` inside the score; `models.py:61` has no runtime
  factor; `models.py:51` initializes `W = randn/√L`.
- **Consequence.** At fixed L a reparametrization of the trained model, but effective
  learning-rate and L2 scaling differ across L; the literal notes equation is not what runs.
- **Proposed resolution.** Keep code behavior; document that the notes' `1/√L` is absorbed
  into initialization scale, or amend the notes. Decide before any cross-L scaling claims.
- **Type.** Scientific (affects cross-L comparisons). **Status.** Resolved for the active
  package (Phase 2): `src/maskeddiffusion/models.py` puts `1/√N` explicitly in the forward
  pass by default; the legacy convention survives only as the named
  `normalization="legacy_init_only"` compat mode used by regression fixtures. The
  notes-vs-legacy-code mismatch itself is intentionally preserved as history.

## D4 — Mask encoding: tuple `(x, m)` vs in-band token 0

- **Evidence.** notes-mem:158 vs `diffusion.py:15,81`, `models.py:60`.
- **Consequence.** None for ±1 data (0 is reserved); breaks if data could take value 0 —
  which interacts with D8.
- **Resolution.** Documented equivalence in `docs/ORIGINAL_ARCHITECTURE.md`.
- **Type.** Engineering. **Status.** Intentionally preserved.

## D5 — Loss normalization `1/(L·B)` explicit in code, implicit in notes

- **Evidence.** `diffusion.py:89` vs notes-mem:124-126 (expectation form).
- **Consequence.** Scale of reported losses and of the effective λ; already accounted for by
  the `l2coeff` construction.
- **Type.** Engineering. **Status.** Resolved (documented).

## D6 — L2 regularizes frozen parameters

- **Evidence.** `diffusion.py:56-60` sums squared norms over **all** registered parameters,
  including frozen ones (`RandomFeatureScore.W1/V1`, frozen Hebbian `W`).
- **Consequence.** Reported `l2loss` is inflated by non-optimizable constants for those
  variants; gradients are unaffected. Linear-model runs with V frozen at 0 are unaffected
  numerically (0² = 0).
- **Proposed resolution.** Restrict `sqnorm` to `requires_grad` parameters when the code is
  next touched; until then, interpret logged `l2_loss` accordingly.
- **Type.** Engineering. **Status.** Resolved in the active package (Phase 2):
  `objectives.l2_regularization` penalizes only trainable parameters
  (`models.regularized_parameters`, tested). The legacy quirk is pinned by the objective
  fixture and left unchanged in the frozen legacy module.

## D7 — Status of `V → 0` (and bias) claims

- **Evidence.** notes-mem:299: "It can be shown that for the data we consider **[Make this
  statement more precise!]** we have m^v=0 and q^v=0"; notes-mem:483 discards the v term.
  This is an RS-level assertion about **uniform** data, self-flagged as imprecise — not a
  theorem. `notes_hiddenmanifold.typ` contains no diffusion model and no V statement. All
  hidden-manifold MMD runs set `freeze_mask_weights=True`, `bias=False`.
- **Consequence.** The experiments' `V ≡ 0` restriction has no theoretical cover under fixed
  `F` (coordinate exchangeability broken); it is a candidate cause of the residual MMD gap.
- **Proposed resolution.** Either derive the population-optimal `V` for the sign-channel
  teacher, or run the ablation with `V` trainable. Until then, describe `V ≡ 0` as an
  experimental restriction.
- **Type.** Scientific. **Status.** Open (recorded as unresolved derivation).

## D8 — `sign(0)` handling divergence

- **Evidence.** `datasets.py:53` uses raw `torch.sign` (sign(0)=0, a value colliding with the
  mask token); the corrected notebook patches `x[x==0]=1` in its data paths; `train.py`
  never patches.
- **Consequence.** Measure-zero in exact arithmetic but possible in practice; a 0-valued
  data spin is indistinguishable from a mask token, silently corrupting the masked objective
  for that coordinate.
- **Proposed resolution.** Apply `sign(0) := +1` inside `RandomFeaturesDataset` (a
  behavior-affecting change — deferred for the legacy module).
- **Type.** Engineering with scientific edge. **Status.** Resolved in the active package
  (Phase 2): `teacher.sign_pm1` implements `sign(0) := +1`; outputs are tested to be
  strictly ±1. The legacy `RandomFeaturesDataset` is intentionally left unpatched (frozen;
  the protected notebook applies its own patch).

## D9 — MMD estimator (and sign-patch) exist only in notebooks

- **Evidence.** No `.py` module contains an MMD/Hamming-kernel estimator; it is defined
  inside several notebooks, with the corrected notebook superseding
  `analysis_mmd_distribution_distance.ipynb`. (The hidden-manifold *generator* does live in
  `datasets.py` — only scoring and the sign patch are notebook-trapped.)
- **Consequence.** Silent divergence risk between notebook copies; the headline diagnostic
  is not under module/test control.
- **Proposed resolution.** Extract to a module via the `migrate-legacy-component` skill.
- **Type.** Engineering. **Status.** Resolved (Phase 2): `src/maskeddiffusion/metrics/mmd.py`
  implements the corrected notebook's exact kernel and estimators (chunked, unclipped raw
  U-statistic), verified equivalent on small inputs by
  `tests/regression/test_mmd_notebook_equivalence.py`. The notebooks themselves are
  untouched.

## D10 — `rfs` model-string: CLI hyphen vs parser underscore

- **Evidence.** `train.py:154` help (and AGENTS.md examples elsewhere use underscore) show
  `rfs10-tanh`-style hyphen; `diffusion.py:21` splits on `_`, so only `rfs10_tanh` parses.
- **Consequence.** Following the help text crashes model construction.
- **Proposed resolution.** Fix the help string.
- **Type.** Engineering. **Status.** Resolved (Phase 2): `train.py` help now shows
  `rfs10_tanh`; the parser was not changed.

## D11 — Objective/sampler relationship (no coherent joint law established)

- **Evidence.** Training predicts all masked sites simultaneously and independently
  (`diffusion.py:83-89`; notes-mem:173 "factorized over output positions"); sequential k=1
  sampling composes single-site conditionals. Only notes-mem:702 acknowledges neglected
  correlations, and only for the accuracy ODE.
- **Consequence.** The terminal law is sampler-dependent; "ancestral sampling of the learned
  joint" is not a licensed description. Objective/sampler mismatch is one live explanation
  of the residual MMD gap.
- **Resolution.** Sampler-indexed language `P_{θ,A,k}` mandated by the contract.
- **Type.** Scientific. **Status.** Open (documented; consistency unproven).

## D12 — Fixed-time vs continuous-time objectives

- **Evidence.** Default objective samples `t~U(0,1)` (continuous mixture). The corrected
  notebook calls its setting `repo_uniform_t` and lists "fixed-mask training mismatch" among
  gap explanations; `TensorBackbone` (`models.py:131-213`) adds discretized time-binning not
  present in the notes.
- **Consequence.** Terminology must distinguish the continuous-t objective (what runs) from
  fixed-t ablations (discussed, not the default).
- **Type.** Scientific bookkeeping. **Status.** Intentionally preserved (documented).

## D13 — Stale operational documentation

- **Evidence.** AGENTS.md claims Python 3.11 pinned in `.python-version` (no such file;
  `pyproject.toml` requires ≥3.12); README points to nonexistent `slurm-jobs/`; `hydra-core`
  is a dependency but never imported; `uv.lock` exists but is gitignored; undocumented CLI
  flags (`--alpha-val`, `--eps`, `--save-dataset`, `--pbar`, `--exp-dir`, `--hebbian`);
  stray `from turtle import pd` at `diffusion.py:2` (unused; pulls in tkinter at import).
- **Consequence.** Misleads humans and agents; the turtle import can crash headless
  machines at import time (it did not in the audited environment — import check passed).
- **Proposed resolution.** CLAUDE.md rewritten (Phase 1); remaining items in Phase 2.
- **Type.** Engineering. **Status.** Resolved (Phase 2): turtle import removed; README
  rewritten; AGENTS.md replaced by a deprecation pointer; `hydra-core` and
  `torch-tb-profiler` removed after verifying zero imports; `.python-version` (3.12)
  created and `uv.lock` un-gitignored (both present in the working tree, awaiting the
  user's commit); undocumented CLI flags now documented
  by deprecation labels in `train.py` help or superseded by the new CLI.

## D14 — `docs/ORIGINAL_ARCHITECTURE.md`'s U-turn description names the wrong method

- **Evidence.** `docs/ORIGINAL_ARCHITECTURE.md`'s item 4 ("U-turn experiments") describes
  `test_step` (`diffusion.py:119-126`, on the `guthlac` branch after the retirement of the
  driver script `experiments-analysis/run_uturn_experiments.py` that this text used to also
  cite) as reconstructing "with `mask_and_sample` (default greedy)". Direct reading of
  `diffusion.py:114-126`'s U-turn block shows it actually calls
  `self.sample(batch_size, k=1, xt=xt)` — the k-token generative sampler
  (`diffusion.py:147-231`, item 1 in the same doc, "always stochastic (fair); no greedy
  option") — not `mask_and_sample` (`diffusion.py:233-298`, item 2, the method that *does*
  support a `greedy` decoding). Found by the `scientific-auditor` reviewer during the
  Phase 3 repository-retirement branch (`guthlac`) reconciliation; this mismatch predates
  that branch (present identically on `main`) and was not introduced by any Phase 3
  deletion — the branch's edit only removed the now-stale driver-script citation and
  otherwise reaffirmed the existing (already-inaccurate) sentence.
- **Consequence.** A reader following `docs/ORIGINAL_ARCHITECTURE.md`'s U-turn description
  would incorrectly expect the logged `test/uturn_overlap_t{t}` metric to reflect
  `mask_and_sample`'s greedy, per-step-unmasking decoding, when it actually reflects
  `self.sample`'s single-shot (`k=1`... actually `k` positions per call, `xt` partially
  masked) stochastic reconstruction. Whether the U-turn mechanism has ever been exercised
  against the hidden-manifold (fixed-`F`) teacher, as opposed to only legacy uniform-data
  checkpoints, is not established by this record.
- **Proposed resolution.** Correct `docs/ORIGINAL_ARCHITECTURE.md` item 4 to name
  `self.sample`, not `mask_and_sample`.
- **Type.** Engineering (documentation accuracy only; no behavior change implied).
  **Status.** Resolved (Phase 3, same commit): `docs/ORIGINAL_ARCHITECTURE.md` item 4
  corrected to name `self.sample`. No code changed (`diffusion.py` remains frozen and
  byte-identical to `main`). The open question of whether U-turn has been exercised
  against the hidden-manifold teacher remains unresolved and is not addressed by this fix.

## D15 — MMD notebook-equivalence fixture tolerance too tight across platforms

- **Evidence.** Adding CI (`.github/workflows/ci.yml`, on the `guthlac` PR) was the first
  time `tests/regression/test_mmd_notebook_equivalence.py` ran on a non-macOS machine.
  `test_matches_independent_notebook_fixture` failed on the GitHub Actions Ubuntu runner:
  `mmd2_unbiased_lambda_4_raw` obtained `-0.03878245353698728` vs. fixture-recorded
  `-0.038782413800557414 ± 3.9e-08` (the effective bound from the fixture's original
  `rel=1e-6, abs=1e-9` tolerance) — a difference of ~4e-8, just past the boundary. Both
  `compute_mmd` (`src/maskeddiffusion/metrics/mmd.py:43-44`) and the notebook-cell
  transcription cast inputs to `torch.float32` identically on both platforms; the
  divergence is attributable to float32 CPU reduction-order differences between macOS
  (arm64) and Linux (x86_64), not a dtype mismatch or an algorithmic difference (verified:
  both implementations use `torch.float32`, confirmed by direct grep).
- **Consequence.** The regression test was flaky-by-platform rather than by-run: it always
  passed locally (macOS) and always failed on the CI runner (Linux) — deterministic per
  platform, not nondeterministic per run — but would have blocked every future CI run on
  this branch.
- **Resolution.** Rerunning the fixture generator, `tests/fixtures/mmd_notebook_reference_v1/generate_fixture.py`
  (which re-verifies the protected notebook's SHA-256 and the extracted block's SHA-256
  before executing it, per `docs/MMD_NOTEBOOK_PROVENANCE.md`), reproduced every `expected`
  value byte-for-byte; only the `tolerance` field was changed, from `rel=1e-6, abs=1e-9` to
  `rel=5e-6, abs=1e-7`. This was verified sufficient (`pytest tests/regression/test_mmd_notebook_equivalence.py
  -vv` — 12/12 passed) and remains 5-6 orders of magnitude below any scientifically
  meaningful MMD difference discussed elsewhere in this repo (~1e-2-1e-1, e.g. the
  persistent MMD gap above the noise floor). The protected notebook itself was never
  modified or executed; only the independent, rerunnable fixture-generation script's
  tolerance constant and the resulting `fixture.json` were changed.
- **Type.** Engineering (test infrastructure/tolerance only — no scientific claim, MMD
  implementation, or protected artifact was touched). **Status.** Resolved (Phase 3,
  `guthlac`): see `docs/MMD_NOTEBOOK_PROVENANCE.md`'s "Cross-platform tolerance" note.
