import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset,load_from_disk
from rouge import FilesRouge, Rouge


model_name = "./pruned_model/llama-7b-hf/sparsegpt/" # "meta-llama/Llama-2-7b-chat-hf"


def get_model_tokenzier(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                 cache_dir="llm_weights", 
                                                 trust_remote_code=True, device_map="auto") # torch_dtype=torch.float16, low_cpu_mem_usage=True, 
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
    return model, tokenizer


def testing_output(model_name, input):
    model, tokenizer = get_model_tokenzier(model_name)

    input_ids = tokenizer.encode(input, return_tensors="pt").to(model.device)
    input_len = input_ids.size()[-1]
    new_tokens_len=int(input_len*0.25)

    output = model.generate(input_ids, num_return_sequences=1, max_new_tokens=new_tokens_len,
                            output_scores=True, output_attentions=True, output_hidden_states=True,
                            return_dict_in_generate = True) # do_sample=True, top_p=0.95, top_k=60, temperature=1, max_length=max_len, 
    # output_text = tokenizer.decode(output[0], skip_special_tokens=True)

# GreedySearchDecoderOnlyOutput
# sequences: the generated sequences of tokens
# scores (optional): the prediction scores of the language modelling head, for each generation step
# hidden_states (optional): the hidden states of the model, for each generation step
# attentions (optional): the attention weights of the model, for each generation step

    # output len 4, 
    # output[0] 1, 15(all tokens), total output ids 
    # output[1], tuple of len 10(new token), ([1,32000]), scores before softmax

    # output[2], ==> tuple of len 10 / of tuple 32 (of tuple 1)/ [1, 32, 5, 5]) 
    # only for new generateds' s attention 
    #          [2] the second generated token [11] 11 out of 32 blocks [0] [2](2 out of 32 heads) [0]  ==> [7] attention 


    # output[3], tuple of len 10 of tuple 33 of tuple 1, [1, 1, 4097]) hidden_states


    return input_len, output

text = "it is a very simple text, please tell me a simple and funny story about the united kingdom"


input_len, output = testing_output(model_name, text)
print(f"==>> input_len: {input_len}")
output_text_len = len(output.sequences[-1])
new_token_len = output_text_len - input_len
print(f"==>> new_token_len: {new_token_len}")

attention = output['attentions']


tokens_attention_to_source_list = []
for token_idx in range(1, new_token_len):
    print(' ')
    print(f"==>> token_idx: {token_idx}")

    attention_to_source_sum = 0
    for head_idx in range(32):
        #attention_to_source = torch.sum(attention[1][31][-1][0][0][:input_len])
        attention_to_source = torch.sum(attention[token_idx][31][-1][head_idx][0][:input_len])
        attention_to_source_sum += attention_to_source
        if head_idx == 31: print(' finished all heads')

    mean_attention_to_source = attention_to_source_sum / 32
    print(f"==>> mean_attention_to_source: {mean_attention_to_source}")
    tokens_attention_to_source_list.append(mean_attention_to_source)

print(tokens_attention_to_source_list)






# from prompt_functions import convert_to_llama_chat_template, llama_prompt, general_prompt

# #raw_text = "The Cable News Network (CNN) is a multinational news channel and website headquartered in Atlanta, Georgia, U.S.[2][3][4] Founded in 1980 by American media proprietor Ted Turner and Reese Schonfeld as a 24-hour cable news channel, and presently owned by the Manhattan-based media conglomerate Warner Bros. Discovery (WBD),[5] CNN was the first television channel to provide 24-hour news coverage and the first all-news television channel in the United States.[6][7][8][9][10]"
# raw_text = """At the end of 1953, a team of young men placed as a dream and objective，
# """


# text = llama_prompt("A", raw_text)
# output = testing_output(model_name, text)
# print(f"==>> output: {output}")

# print(f"==>> 0 : {output[0]}")
# print(f"==>> 1 : {output[1]}")
# print(f"==>> 2 : {output[2]}")