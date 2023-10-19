#!/bin/bash
#SBATCH --partition=gpu 
#SBATCH --qos=gpu 
#SBATCH --nodes=1 
#SBATCH --gpus-per-node=1
#SBATCH --mem=82G 
#SBATCH --time 1:00:00 
#SBATCH --job-name=2gpu_40b

# Load modules & activate env

module load Anaconda3/2022.10
module load CUDA/11.8.0

# Activate env
source activate seq


python main.py --model "NousResearch/Nous-Hermes-llama-2-7b" --prune_method "magnitude" 