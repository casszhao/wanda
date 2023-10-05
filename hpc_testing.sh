#!/bin/bash

#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --time=1-00:00:00
#SBATCH --output=joboutput/%x_%a.out

#SBATCH --job-name=pru-gpt2

module load Anaconda3/2022.10
module load CUDA/11.8.0
source activate seq


# Set common variables
model="gpt2-medium"  # "decapoda-research/llama-7b-hf"
model_short_name="gpt2_medium" # "llama_7b"         
sparsity_ratio=0.5
cuda_device=0



# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device


# Define function to run python command
run_python_command () {
    python main.py \
    --model $model \
    --prune_method $1 \
    --sparsity_ratio $sparsity_ratio \
    --sparsity_type $2 \
    --save $3
}

# llama-7b with wanda pruning method
echo "Running with wanda pruning method"
run_python_command "wanda" "unstructured" "results/"$model_short_name"/unstructured/wanda/"
run_python_command "wanda" "2:4" "results/"$model_short_name"/2-4/wanda/"
run_python_command "wanda" "4:8" "results/"$model_short_name"/4-8/wanda/"
echo "Finished wanda pruning method"

# llama-7b with sparsegpt pruning method
echo "Running with sparsegpt pruning method"
run_python_command "sparsegpt" "unstructured" "results/"$model_short_name"/unstructured/sparsegpt/"
run_python_command "sparsegpt" "2:4" "results/"$model_short_name"/2-4/sparsegpt/"
run_python_command "sparsegpt" "4:8" "results/"$model_short_name"/4-8/sparsegpt/"
echo "Finished sparsegpt pruning method"

# llama-7b with magnitude pruning method
echo "Running with magnitude pruning method"
run_python_command "magnitude" "unstructured" "results/"$model_short_name"/unstructured/magnitude/"
run_python_command "magnitude" "2:4" "results/"$model_short_name"/2-4/magnitude/"
run_python_command "magnitude" "4:8" "results/"$model_short_name"/4-8/magnitude/"
echo "Finished magnitude pruning method"



# python main.py \
#     --model $model \
#     --prune_method wanda \
#     --sparsity_ratio $sparsity_ratio \
#     --sparsity_type unstructured \
#     --device cuda