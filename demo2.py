import transformers, json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset,load_from_disk
from rouge import FilesRouge, Rouge
from prompt_functions import generate_prompt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os, argparse, time




def get_attention_to_source_list(model, tokenizer, input):
    input_ids = tokenizer.encode(input, return_tensors="pt").to(model.device)
    input_len = input_ids.size()[-1]
    new_tokens_len=int(input_len*0.25)

    output = model.generate(input_ids, num_return_sequences=1, max_new_tokens=new_tokens_len,
                            output_scores=True, output_attentions=True, output_hidden_states=True,
                            return_dict_in_generate = True) 

    output_text_len = len(output.sequences[-1])
    new_token_len = output_text_len - input_len

    attention = output['attentions']
    tokens_attention_to_source_list = []
    for token_idx in range(1, new_token_len):
        attention_to_source_sum = 0
        for head_idx in range(32):
            attention_to_source = torch.sum(attention[token_idx][31][-1][head_idx][0][:input_len])
            attention_to_source_sum += attention_to_source
        mean_attention_to_source = attention_to_source_sum / 32
        tokens_attention_to_source_list.append(mean_attention_to_source.item())
    return tokens_attention_to_source_list



def get_full_data_attention(model_name, test_num, prompt_id):
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                    cache_dir="llm_weights", 
                                                    trust_remote_code=True, device_map="auto") # torch_dtype=torch.float16, low_cpu_mem_usage=True, 
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)

    list_of_list = []
    for i, key in enumerate(dataset.keys()):
        
        if i <= test_num:
            print(i)
            text = generate_prompt(
                task="summarization",
                model= 'llama',
                prompt_id=prompt_id,
                document=dataset[key]["document"],
                )
            
            list = get_attention_to_source_list(model, tokenizer, text)
            list_of_list.append(list)

    return list_of_list



f = open(f"data/factcc.json")
dataset = json.load(f)




full_data_full_model_attention = get_full_data_attention("meta-llama/Llama-2-7b-chat-hf", 11, "C")
with open(f'./saved_attention/fulldata_factcc_promptC_full_mode.pkl', 'wb') as f:
    pickle.dump(full_data_full_model_attention,f)
    print(' saved at =====> ./saved_attention/fulldata_factcc_promptC_full_model.pkl')





full_data_pruned_model_attention = get_full_data_attention("./pruned_model/llama-7b-hf/sparsegpt/", 11, "C")
with open(f'./saved_attention/fulldata_factcc_promptC_pruned_mode.pkl', 'wb') as f:
    pickle.dump(full_data_pruned_model_attention,f)
    print(' saved at =====> ./saved_attention/fulldata_factcc_promptC_pruned_model.pkl')



def get_mean_std(list_of_list):

    min_length = min(len(sub_list) for sub_list in list_of_list)
    cuted_pruned_attention_list = [sub_list[:min_length] for sub_list in list_of_list]    
    mean = np.mean(cuted_pruned_attention_list, axis=0)
    std = np.std(cuted_pruned_attention_list, axis=0)

    index = np.arange(1, len(mean) + 1).tolist()

    return index, mean, std





index, mean, std = get_mean_std(full_data_full_model_attention)
plt.plot(index, mean, "grey", label="No Pruning")
plt.fill_between(index, mean - std, mean + std, alpha=0.2, color="grey")

index, mean, std = get_mean_std(full_data_pruned_model_attention)
plt.plot(index, mean, "red", label="SparseGPT")
plt.fill_between(index, mean - std, mean + std, alpha=0.2, color="red")

plt.legend()
plt.xlabel("Generated new token index")
plt.ylabel("Attention ratio (sum) to source input")
plt.title('Attention distribution (attention to source)', fontsize=18)
plt.savefig(f'./images/ttt.png')