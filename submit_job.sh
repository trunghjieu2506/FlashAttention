#!/bin/bash

#SBATCH --job-name=matrix_multiply
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

# Navigate to working directory
cd $SLURM_SUBMIT_DIR

# Build the program
make clean
make

# Run the program
echo "Running matrix_multiply..."
./matrix-multiply/testbed

echo "Job completed at $(date)"