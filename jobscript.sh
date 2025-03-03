#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu
#SBATCH --mem=8000

module purge
module load Python/3.11.3-GCCcore-12.3.0

pip install -r requirements.txt

python3 train.py