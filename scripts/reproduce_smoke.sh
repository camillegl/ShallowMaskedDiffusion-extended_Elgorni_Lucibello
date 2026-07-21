#!/usr/bin/env bash
# Tiny CPU smoke run: teacher -> data -> train -> checkpoint -> sample ->
# metrics -> artifact -> validate. Integration check only; never interpret
# scientifically. Does not touch the protected reference notebooks/results.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${1:-artifacts/smoke-run}"
rm -rf "$OUT"

uv run maskeddiffusion-train --config configs/smoke/smoke.toml --output "$OUT" --device cpu
uv run maskeddiffusion-sample --config configs/smoke/smoke.toml --output "$OUT-samples" \
    --checkpoint "$OUT/checkpoints/final.pt" --n-samples 8 --device cpu
uv run maskeddiffusion-evaluate --config configs/smoke/smoke.toml --output "$OUT-eval" \
    --checkpoint "$OUT/checkpoints/final.pt" --teacher "$OUT/teacher.pt" \
    --samples "$OUT-samples" --device cpu
# Tiny U-turn smoke curve against the same checkpoint, stochastic and greedy
# (the sampler config does not affect training, so both U-turn runs reuse
# the one trained checkpoint).
uv run maskeddiffusion-uturn --config configs/smoke/smoke.toml --output "$OUT-uturn-stochastic" \
    --checkpoint "$OUT/checkpoints/final.pt" --teacher "$OUT/teacher.pt" \
    --t-values 0.0 0.25 0.5 0.75 --n-examples 4 --device cpu
uv run maskeddiffusion-uturn --config configs/smoke/smoke_greedy.toml --output "$OUT-uturn-greedy" \
    --checkpoint "$OUT/checkpoints/final.pt" --teacher "$OUT/teacher.pt" \
    --t-values 0.0 0.25 0.5 0.75 --n-examples 4 --device cpu
uv run maskeddiffusion-validate-artifact "$OUT"
uv run maskeddiffusion-validate-artifact "$OUT-samples"
uv run maskeddiffusion-validate-artifact "$OUT-uturn-stochastic"
uv run maskeddiffusion-validate-artifact "$OUT-uturn-greedy"
echo "smoke OK"
