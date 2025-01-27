#!/bin/bash
#SBATCH --comment=seq_rationales
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --output=jobs.out/%j.log
#SBATCH --time=4-00:00:00

#SBATCH --job-name=roberta-gpt2

# Load modules & activate env

module load Anaconda3/2022.10
module load CUDA/11.8.0

# Activate env
source activate seq_rational_pt2   


# Set common variables
model="decapoda-research/llama-7b-hf"
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
run_python_command "wanda" "unstructured" "out/llama_7b/unstructured/wanda/"
run_python_command "wanda" "2:4" "out/llama_7b/2-4/wanda/"
run_python_command "wanda" "4:8" "out/llama_7b/4-8/wanda/"
echo "Finished wanda pruning method"

# llama-7b with sparsegpt pruning method
echo "Running with sparsegpt pruning method"
run_python_command "sparsegpt" "unstructured" "out/llama_7b/unstructured/sparsegpt/"
run_python_command "sparsegpt" "2:4" "out/llama_7b/2-4/sparsegpt/"
run_python_command "sparsegpt" "4:8" "out/llama_7b/4-8/sparsegpt/"
echo "Finished sparsegpt pruning method"

# llama-7b with magnitude pruning method
echo "Running with magnitude pruning method"
run_python_command "magnitude" "unstructured" "out/llama_7b/unstructured/magnitude/"
run_python_command "magnitude" "2:4" "out/llama_7b/2-4/magnitude/"
run_python_command "magnitude" "4:8" "out/llama_7b/4-8/magnitude/"
echo "Finished magnitude pruning method"