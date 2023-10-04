#!/usr/bin/env bash

# model handle must be provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <HF_HANDLE>"
    exit 1
fi

HUGGINGFACE_MODEL_HANDLE=$1
PRUNE_METHOD="wanda"
SPARSITY_TYPE="2:4"
SAVE_PATH="temp/"

python main.py  --model $HUGGINGFACE_MODEL_HANDLE \
                --prune_method $PRUNE_METHOD \
                --sparsity_type $SPARSITY_TYPE \
                --save $SAVE_PATH

# sample python -m pdb main.py --model gpt2 --prune_method wanda --sparsity_type 2:4 --device cpu
