#!/bin/bash
#SBATCH --job-name="shallowdiff"
#SBATCH --time=240:00:00
#SBATCH --partition=compute
# #SBATCH --qos=debug # only for debug_gpu and debug_cpu partitions
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2      # number of threads per task
#SBATCH --ntasks=1             # SET EQUAL TO gpus IF DOING pytorch's DDP
#SBATCH --output=slurm-logs/%x_%j.out
#SBATCH --error=slurm-logs/%x_%j.err
#SBATCH --mem-per-cpu=16000M  # memory per cpu core, default 8GB

#SBATCH --account=lucibello 
#SBATCH --mail-type=NONE   #notify for NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-user=carlo.lucibello@unibocconi.it

## AVAILABLE PARTITIONS
# defq, timelimit 3 days, Nodes=cnode0[1-4] (CPU)
# compute, timelimit 15 days, Nodes=cnode0[5-8] (CPU)
# long_gpu timelimit 3 days, Nodes=gnode0[1-2] (GPU)
# gpu timelimit 1 day, Nodes=gnode0[1-4] (GPU)
# medium_gpu, timelimit 3 hours, Nodes=gnode0[1-4] (GPU)
# stata, timelimit 3 days, Nodes=cnode08 (CPU)
# debug_cpu, timelimit 15 minutes, Nodes=cnode01 (short test on CPU)
# debug_gpu, timelimit 15 minutes, Nodes=gnode0[1-4] (short test on GPU)

## COMMON SLURM COMMANDS 
# squeue 
# sbatch job.sh
# sinfo -Nel
# scancel <job_id>

start_time=$(date +%s)

L=${1:-2000}     # read from command line arg 1, default 2000
alpha=${2:-0.1}  # read from command line arg 2, default 0.1
l2reg=${3:-0.0}  # read from command line arg 3, default 0.0

dataset="uniform"
model="linear"
epochs=5000
freeze_mask_weights=true
save_dataset=true
pbar=false # set to false if running on slurm
test=false


srun uv run train.py --L=$L --alpha=$alpha --dataset=$dataset --model=$model\
                      --epochs=$epochs --l2reg=$l2reg --test=$test --pbar=$pbar\
                      --freeze-mask-weights=$freeze_mask_weights --save-dataset=$save_dataset

end_time=$(date +%s)
elapsed_seconds=$((end_time - start_time))
elapsed_hours=$(awk "BEGIN {printf \"%.4f\", $elapsed_seconds/3600}")
echo "Elapsed time: $elapsed_hours hours"
