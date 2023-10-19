#!/bin/bash
#SBATCH --partition=gpu 
#SBATCH --qos=gpu 
#SBATCH --nodes=1 
#SBATCH --gpus-per-node=2
#SBATCH --mem=150G 
#SBATCH --time 1:00:00 
#SBATCH --job-name=2gpu_40b

# Load modules & activate env

module load Anaconda3/2022.10
module load CUDA/11.8.0

# Activate env
source activate seq


# Set CUDA device visibility
# export CUDA_VISIBLE_DEVICES=0,1
python main.py --model "tiiuae/falcon-40b-instruct" --prune_method "wanda" 