#!/bin/bash

# Get the directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

L=2000
l2reg=0.0
for alpha in $(seq 1.1 0.1 3.0)
do
    sbatch  $SCRIPT_DIR/train-cpu.sh $L $alpha $l2reg
done

