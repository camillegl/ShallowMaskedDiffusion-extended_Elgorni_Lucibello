# Phase 4C pilot run — timing and storage calibration

Status: **integration/calibration measurement only, not a scientific result.**
Two repeats at D=32 with `max_steps=200`/`n_generate=200`/`n_true=200` is not a
converged, publishable configuration — see `docs/RESEARCH_SPEC.md`; this
document exists to calibrate the campaign's engineering parameters, not to
report a memorization/generalization finding.

## What was run

`configs/experiments/pilot/v_trainability_d32_g2_a2_pilot.toml`: the central
`campaign_v1` cell (`v_trainability`, D=32, γ=2, α=2 — same design point as
`configs/experiments/campaign_v1/v_trainability_d32_g2_a2.toml`), two
conditions (`frozen_zero_v`/`trainable_v`), 2 repeats, `n_generate=200`,
`max_steps=200`, `n_true=200`, U-turn enabled at `t ∈ {0.2, 0.5, 0.8}` with 8
examples per source. 4 runs total. Executed once on CPU
(`uv run maskeddiffusion-experiment ... --device cpu`), then analyzed once
(`uv run maskeddiffusion-p4c-analyze`). Machine: this development machine
(Apple Silicon macOS), not a calibration for any specific cluster/CI
hardware — treat absolute numbers as order-of-magnitude, not portable
benchmarks.

## Timing

| Stage | Measured | Notes |
|---|---|---|
| `maskeddiffusion-experiment` (4 runs: train+sample+eval+uturn each) | **3.42 s wall**, 1.99 s user, 0.72 s sys | ≈0.86 s/run average, all four stages included |
| — of which, one run's `train` stage alone | **0.079 s** (`train/summary.json` `wall_clock_s`, 200 steps) | negligible at this scale: D=32 is a tiny linear model; training is not the bottleneck |
| `maskeddiffusion-p4c-analyze` (4 records → tables/report/figures) | **4.63 s wall**, 3.28 s user, 0.48 s sys | dominated by MMD kernel computation (`n_true=200`, quadratic in sample count) and figure rendering, not I/O |
| Peak RSS, experiment run | 289 MB | |
| Peak RSS, analyze run | 328 MB | |

Training time is essentially free at D=32; the per-run cost is dominated by
sampling + MMD evaluation + U-turn reconstruction (all O(n_generate) or
O(n_true²)-ish kernel work), not the optimizer loop.

## Storage

| Item | Measured |
|---|---|
| One full run (train+samples+eval+uturn artifacts) | ~264–296 KB |
| Per-stage breakdown (one run) | train 108 KB, samples 68 KB, uturn 60 KB, eval 16 KB |
| 4-run pilot experiment root | 1.1 MB |
| Analysis output (4 tables + report JSON + manifest + figures) | 2.8 MB |
| Checkpoint file (`final.pt`) | 80 KB |
| Samples tensor (`samples.pt`, 200×64 bool-ish) | 53 KB |

Storage is dominated by the analysis figures (PDF+PNG per metric per cell),
not the run artifacts themselves, at this scale.

## MMD sanity (integration check, not a scientific claim)

All four runs' `model_vs_true` biased MMD² (0.0117–0.0136) sit visibly above
the `true_vs_true` floor (0.0087–0.0095) — a decreasing-toward-floor pattern
that is at least dimensionally sane for a barely-trained (200-step) model,
consistent with the diagnostic behaving as designed. This is not evidence of
anything about memorization/generalization at 200 steps and is not reported
as such — see the wording-policy prohibition on convergence/learning claims
in `docs/PHASE4C_ANALYSIS_SPEC.md` §7.

## Extrapolation to `campaign_v1` at its current placeholder settings

`campaign_v1`'s 14 configs total 93 planned runs (see
`docs/PHASE4C_EXPERIMENT_PROTOCOL.md` §7) at `n_generate=1000`, `n_true=1000`,
`max_steps=2000` — 5× the pilot's sample counts, 10× the pilot's training
steps, and 23.25× the pilot's run count. Since:

- training cost is negligible at this scale even at 10× more steps (order
  0.1–1 s/run, still small relative to eval);
- MMD/eval cost scales with `n_true` (kernel Gram matrices ~O(n_true²) per
  comparison) — 5× `n_true` is plausibly 10–25× the per-run eval cost;
- U-turn cost was **not** scaled in `campaign_v1` (no `[uturn]` table in any
  campaign config — U-turn is smoke-only for now), so it does not enter this
  extrapolation;

a rough order-of-magnitude estimate for the full `campaign_v1` sweep on this
same CPU is **tens of minutes to a few hours**, not the ~15 s the pilot's
raw 23× run-count scaling alone would suggest — this is a wide,
low-confidence range, not a committed number. **A dedicated timing run at
`campaign_v1`'s actual settings (one pair, not all 93 runs) is the correct
next calibration step before launching the full campaign**, not this
pilot's 200-step numbers alone.

## Recommendations for `campaign_v1`

1. Do not change `campaign_v1`'s placeholder settings from this pilot alone
   — 200 steps is far too short to be a scientifically meaningful
   optimization budget at any dimension; the pilot measured engineering
   cost, not sufficiency.
2. Before removing the "placeholder" label from `campaign_v1` configs, run
   one true-scale pair (`n_generate=1000`, `n_true=1000`, `max_steps=2000`,
   `repeats=1`) and confirm: (a) wall-clock is acceptable for the full
   93-run sweep, (b) training loss actually decreases meaningfully over
   2000 steps (no automated convergence criterion exists yet — Phase 4A
   gap, still open), (c) disk usage projects to a manageable total
   (pilot's 2.8 MB analysis output for 4 records suggests analysis storage
   is not a concern; run-artifact storage at 1000 samples/1000 n_true will
   be larger than the pilot's 264–296 KB/run but likely still well under
   1 MB/run for this model size).
3. This pilot's config is committed (`configs/experiments/pilot/`) as the
   calibration record; its generated output (`artifacts/pilot/`) is not —
   `.gitignore` now excludes `artifacts/pilot/` alongside the existing
   `artifacts/smoke-run*/`/`runs/` patterns.
