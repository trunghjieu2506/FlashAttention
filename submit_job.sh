#!/bin/bash

#SBATCH --job-name=triton-benchmark
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --mem=4GB
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Load modules if needed
# module load gcc
# module load cuda

.venv/bin/python -m cs336_systems.benchmarking \
  --d-model 32 --d-ff 128 --num-layers 2 --num-heads 4 \
  --batch-size 2 --context-length 16 \
  --warmup-steps 1 --measure-steps 2 \
  --mode train --device cpu