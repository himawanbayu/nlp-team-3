#!/bin/bash
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=15GB
#SBATCH --time=10:00:00

# start environments and load  modules
module load Python/3.10.4-GCCcore-11.3.0
source ../env/bin/activate

python3 fine_tuned.py