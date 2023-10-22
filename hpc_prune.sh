#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1

#SBATCH --gres=gpu:1
#SBATCH --mem=88G
#SBATCH --time=1:00:00

#SBATCH --job-name=fal7b_wanda

# Load modules & activate env

module load Anaconda3/2022.10
module load CUDA/11.8.0

# Activate env
source activate seq


# Set CUDA device visibility
# export CUDA_VISIBLE_DEVICES=0,1

# export CUDA_VISIBLE_DEVICES=0,1,2,3
# nvidia-smi
python main.py --model "tiiuae/falcon-40b-instruct" --prune_method "magnitude"