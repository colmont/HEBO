#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --gpus=quadro_rtx_6000:0
#SBATCH --gres=gpumem:4g
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --output=log.out

source /cluster/home/cdoumont/miniconda3/etc/profile.d/conda.sh
conda activate mcbo_env

bash "./run_experiments.sh"