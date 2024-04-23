#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=deepset     # Name of the job 
#SBATCH --export=ALL                  # Export all environment variables
#SBATCH --output=train-deepset.log # Log-file (important!)
#SBATCH --cpus-per-task=2             # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=4G              # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                  # Number of GPU's
#SBATCH --time=24:00:00               # Max execution time
#

conda activate sbi-astrometry

python generate.py --method "Deepset"
python train.py --method "Deepset" 
