#!/bin/bash
#SBATCH --job-name="shallowdiff"
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
# #SBATCH --qos=debug # only for debug_gpu and debug_cpu partitions
#SBATCH --nodes=1
#SBATCH --gpus=1               # num gpus. If set to 0 change the partition to defq or compute
#SBATCH --cpus-per-task=8      # number of threads per task
#SBATCH --ntasks=1             # SET EQUAL TO gpus IF DOING pytorch's DDP
#SBATCH --output=outputs/log/%x_%j.out
#SBATCH --error=outputs/log/%x_%j.err
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

module load cuda/12.8


start_time=$(date +%s)

L=8192
alpha=0.5
l2reg=0.0
dataset="uniform"
epochs=5000

srun uv run train.py --L=$L --alpha=$alpha --dataset=$dataset --epochs=$epochs --l2reg=$l2reg --test --no-pbar

end_time=$(date +%s)
elapsed_seconds=$((end_time - start_time))
elapsed_hours=$(awk "BEGIN {printf \"%.4f\", $elapsed_seconds/3600}")
echo "Elapsed time: $elapsed_hours hours"

