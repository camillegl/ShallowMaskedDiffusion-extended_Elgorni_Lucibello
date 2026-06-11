#!/bin/bash
#SBATCH --job-name=uturn
#SBATCH --account=3261535
#SBATCH --partition=gpunew
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64gb
#SBATCH --ntasks=1
#SBATCH --output=outputs/log/%x_%j.out
#SBATCH --error=outputs/log/%x_%j.err
#SBATCH --mail-type=all
#SBATCH --mail-user=3261535+hpc@phd.unibocconi.it


module load cuda/12.8

# Defaults (can be overridden with sbatch --export=L=...,HEBBIAN=true,...)
: "${L:=8192}"
: "${L2REG:=0.0}"
: "${DATASET:=uniform}"
: "${MODEL:=linear}"
: "${NUM_SAMPLES:=200}"
: "${DECODING:=greedy}"
: "${LOGS_DIR:=logs}"
: "${OUTPUT_DIR:=experiments-analysis}"
: "${DEVICE:=auto}"
: "${HEBBIAN:=false}"
: "${INCLUDE_FMW:=false}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p "${OUTPUT_DIR}" "${LOGS_DIR}" slurm-logs

ARGS=(--L "$L" --dataset "$DATASET" --model "$MODEL" --num-samples "$NUM_SAMPLES" --decoding "$DECODING" --logs-dir "$LOGS_DIR" --output-dir "$OUTPUT_DIR" --l2reg "$L2REG" --device "$DEVICE")
if [ "${HEBBIAN}" = "true" ]; then ARGS+=(--hebbian); fi
if [ "${INCLUDE_FMW}" = "true" ]; then ARGS+=(--include-fmw); fi

srun uv run python experiments-analysis/run_uturn_experiments.py "${ARGS[@]}"
