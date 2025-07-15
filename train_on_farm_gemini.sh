#!/bin/bash

#SBATCH --job-name=heesup_array
#SBATCH --output=code_node%a.out
#SBATCH --account=geminigrp
#SBATCH --partition=gpu-6000_ada-h
#SBATCH --array=0-1                      # 2개 독립 작업
#SBATCH --gres=gpu:4                     # 각 작업당 4개 GPU
#SBATCH --cpus-per-task=32
#SBATCH --time=7-00:00:00               # 7일
#SBATCH --error=code_node%a.err

source /home/lion397/.bashrc
conda activate /home/lion397/codes/Image2PlantArchitecture/.env

export NODE_ID=$SLURM_ARRAY_TASK_ID
export NUM_GPUS=4

echo "Starting array task $SLURM_ARRAY_TASK_ID"
echo "Node ID: $NODE_ID"
echo "Running on actual node: $SLURMD_NODENAME"

./run_experiments.sh