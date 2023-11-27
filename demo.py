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

parser = argparse.ArgumentParser()
parser.add_argument('--model', default="meta-llama/Llama-2-7b-chat-hf", type=str, help='LLaMA model', 
                        choices=[
                                 "meta-llama/Llama-2-7b-chat-hf",
                                 "meta-llama/Llama-2-13b-chat-hf",
                                 ])
parser.add_argument('--data', default="summeval", type=str, help='select a summarization dataset', 
                    choices=[ #"cogensumm", "frank", 
                             "polytope", "factcc", "summeval", "xsumfaith",
                             ])
parser.add_argument('--prune_method', default="sparsegpt", type=str, help='if using pruned model and which to use', 
                    choices=["fullmodel", "wanda", "sparsegpt", "magnitude"])
parser.add_argument('--prompt_id', default="C", type=str, 
                    choices=["A", "B", "C"],
                    help='pick a prompt template from prompt list, A or B or None')
parser.add_argument('--test_num', default=1111, type=int)
args = parser.parse_args()


current_dir = os.getcwd()
print("Current working directory:", current_dir)


# def get_model_tokenzier(model_name):
#     model = AutoModelForCausalLM.from_pretrained(model_name, 
#                                                  cache_dir="llm_weights", 
#                                                  trust_remote_code=True, device_map="auto") # torch_dtype=torch.float16, low_cpu_mem_usage=True, 
#     tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
#     return model, tokenizer



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



f = open(f"data/{args.data}.json")
dataset = json.load(f)


if "llama" in str(args.model):
    size = str(args.model).split("-")[-3]
    pruned_model_shortname = f"llama-{size}-hf"
else:
    pruned_model_shortname = str(args.model).split("/")[-1]
print(f"==>> pruned_model_shortname: {pruned_model_shortname}")




full_data_pruned_model_attention = get_full_data_attention(f"./pruned_model/{pruned_model_shortname}/{args.prune_method}/", 
                                                           args.test_num, args.prompt_id)
with open(f'./saved_attention/{pruned_model_shortname}/fulldata_{args.data}_prompt{args.prompt_id}_{args.prune_method}.pkl', 'wb') as f:
    pickle.dump(full_data_pruned_model_attention,f)
    print(f' saved pruned_model len ({len(full_data_pruned_model_attention)})')
    
pruned = np.load(f'./saved_attention/{pruned_model_shortname}/fulldata_{args.data}_prompt{args.prompt_id}_{args.prune_method}.pkl', allow_pickle=True)
print(f"==>> pruned: {len(pruned)}")

full_data_full_model_attention = get_full_data_attention(args.model, 
                                                         args.test_num, args.prompt_id)
with open(f'./saved_attention/fulldata_{args.data}_prompt{args.prompt_id}_full_model.pkl', 'wb') as f:
    pickle.dump(full_data_full_model_attention,f)
    print(' saved full_model')





def get_mean_std(list_of_list):
    min_length = min(len(sub_list) for sub_list in list_of_list)
    cuted_pruned_attention_list = [sub_list[:min_length] for sub_list in list_of_list]    
    mean = np.mean(cuted_pruned_attention_list, axis=0)
    std = np.std(cuted_pruned_attention_list, axis=0)
    index = np.arange(1, len(mean) + 1).tolist()

    return index, mean, std



index, mean, std = get_mean_std(full_data_pruned_model_attention)
plt.plot(index, mean, "red", label=str(args.prune_method).capitalize())
plt.fill_between(index, mean - std, mean + std, alpha=0.2, color="red")

index, mean, std = get_mean_std(full_data_full_model_attention)
plt.plot(index, mean, "grey", label="No Pruning")
plt.fill_between(index, mean - std, mean + std, alpha=0.2, color="grey")

plt.legend()
plt.xlabel("Generated new token index")
plt.ylabel("Attention ratio (sum) to source input")
plt.title('Attention distribution (attention to source)', fontsize=18)
plt.savefig(f'./images/{pruned_model_shortname}/{args.prune_method}_{pruned_model_shortname}_Prompt{args.prompt_id}_{args.data}.png')


# model_name = "meta-llama/Llama-2-7b-chat-hf"
# fullmodel_list = get_attention_to_source_list(model_name, text)




# a_index = np.arange(1,len(pruned_list)+1).tolist()
# b_index = np.arange(1,len(fullmodel_list)+1).tolist()



# plt.plot(a_index, pruned_list, "red", label="SparseGPT")
# plt.plot(b_index, fullmodel_list, "grey", label="No Pruning")
# plt.legend()
# plt.xlabel("Generated new token index")
# plt.ylabel("Attention ratio (sum) to source input")
# plt.title('example of high hallucination: Attention distribution (attention to source input) of SparseGPT and No Pruning')
# plt.savefig('high_hallucination.png')




# with open('test.npy', 'rb') as f:
#     a = np.load(f)
#     b = np.load(f)


# a_index = np.arange(1,len(a)+1).tolist()
# print(len(a_index))
# b_index = np.arange(1,len(b)+1).tolist()

# plt.plot(a_index, a, "red", label="SparseGPT")
# plt.fill_between(a_index, a - 0.03, a + 0.1, alpha=0.2, color="red")
# plt.plot(b_index, b, "grey", label="No Pruning")
# plt.fill_between(b_index, b - 0.02, b + 0.05, alpha=0.2, color="grey")
# plt.legend()
# plt.xlabel("Generated new token index")
# plt.ylabel("Attention ratio (sum) to source input")
# plt.title('Attention distribution (attention to source input) of SparseGPT and No Pruning')
# plt.savefig('ttt.png')