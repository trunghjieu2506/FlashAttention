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

set -euo pipefail

gpu_type="${1:-${GPU_TYPE:-}}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  gres="gpu:1"
  if [[ -n "${gpu_type}" ]]; then
    gres="gpu:${gpu_type}:1"
  fi

  exec sbatch --gres="${gres}" "$0"
fi

# Load modules if needed.
# module load gcc
# module load cuda

.venv/bin/python -m cs336_systems.benchmarking \
  --d-model 32 --d-ff 128 --num-layers 2 --num-heads 4 \
  --batch-size 2 --context-length 16 \
  --warmup-steps 5 --measure-steps 2 \
  --mode train --device cuda
