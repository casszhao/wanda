import transformers, json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset,load_from_disk
from rouge import FilesRouge, Rouge
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os, argparse, time


def get_mean_std(list_of_list):

    min_length = min(len(sub_list) for sub_list in list_of_list)
    cuted_pruned_attention_list = [sub_list[:min_length] for sub_list in list_of_list]    
    mean = np.mean(cuted_pruned_attention_list, axis=0)
    std = np.std(cuted_pruned_attention_list, axis=0)

    index = np.arange(1, len(mean) + 1).tolist()
    return index, mean, std


prompt_id = "C"
prune_method = "sparsegpt"
dataset = "factcc"
pruned = np.load(f'./saved_attention/fulldata_{dataset}_prompt{prompt_id}_{prune_method}_model.pkl', allow_pickle=True)
print(f"==>> pruned: {len(pruned)}")
full  = np.load(f"./saved_attention/fulldata_{dataset}_prompt{prompt_id}_full_model.pkl", allow_pickle=True)
print(f"==>> full: {len(full)}")


prund_label = "SparseGPT" if prune_method == "sparsegpt" else "Wanda"
index, mean, std = get_mean_std(pruned)
plt.plot(index, mean, "red", label=prund_label)
plt.fill_between(index, mean - std, mean + std, alpha=0.2, color="red")

index, mean, std = get_mean_std(full)
plt.plot(index, mean, "grey", label="No Pruning")
plt.fill_between(index, mean - std, mean + std, alpha=0.2, color="grey")

plt.legend()
plt.xlabel("Generated new token index")
plt.ylabel("Attention ratio (sum) to source input")
plt.title('Attention distribution (attention to source)', fontsize=18)
plt.savefig(f'./images/{prune_method}_llama-7b-hf_Prompt{prompt_id}_{dataset}.png')