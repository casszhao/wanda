#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1

#SBATCH --gres=gpu:2
#SBATCH --mem=188G
#SBATCH --time=12:00:00

#SBATCH --job-name=7bA_summeval

# Load modules & activate env

module load Anaconda3/2022.10
module load CUDA/11.8.0

# Activate env
source activate seq


# Set CUDA device visibility
# export CUDA_VISIBLE_DEVICES=0,1

# export CUDA_VISIBLE_DEVICES=0,1,2,3
# nvidia-smi


model_handle="/pruned_model/Llama-2-13b-chat-hf_"
dataset="summeval"
promptID="C"


for ratio in "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7"
do 
model_name=$model_handle$ratio
for method in "wanda" "sparsegpt"
do
python demo.py --prompt_id $promptID --prune_method $method --model $model_name --data $dataset
done
done