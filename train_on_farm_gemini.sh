#!/bin/bash

#SBATCH --job-name=heesup
#SBATCH --output=code.out
#SBATCH --account=geminigrp  #geminigrp,jmearlesgrp
#SBATCH --partition=gpu-a100-h#gpu-a100-h #gpum,gpu-a100-h
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00 # Change the time accordingly, ex: 36:00:00
#SBATCH --error=code.err
#SBATCH --cpus-per-task=8

source /home/lion397/.bashrc
conda activate /home/lion397/codes/Image2PlantArchitecture/.env
time python src/train.py