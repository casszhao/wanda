#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1

#SBATCH --gres=gpu:2
#SBATCH --mem=188G
#SBATCH --time=12:00:00

#SBATCH --job-name=save_attention

# Load modules & activate env

module load Anaconda3/2022.10
module load CUDA/11.8.0

# Activate env
source activate seq


# Set CUDA device visibility
# export CUDA_VISIBLE_DEVICES=0,1

# export CUDA_VISIBLE_DEVICES=0,1,2,3
# nvidia-smi

method="sparsegpt"
model_handle="meta-llama/Llama-2-7b-chat-hf"
dataset="factcc"

for promptID in "A" "B" "C"
do
python demo.py --prompt_id $promptID --prune_method $method --model $model_handle --data $dataset
done