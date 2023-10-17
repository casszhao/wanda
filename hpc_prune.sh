#!/bin/bash
#SBATCH --comment=prune
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=220G


#SBATCH --job-name=pru-falcon

# Load modules & activate env

module load Anaconda3/2022.10
module load CUDA/11.8.0

# Activate env
source activate seq


# Set CUDA device visibility
# export CUDA_VISIBLE_DEVICES=0,1

export CUDA_VISIBLE_DEVICES=0,1
nvidia-smi
python main.py --model "facebook/opt-iml-30b" --prune_method "sparsegpt" 