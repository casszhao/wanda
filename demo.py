import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset,load_from_disk
from rouge import FilesRouge, Rouge


# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", cache_dir="llm_weights") # torch_dtype=torch.float16, low_cpu_mem_usage=True, 
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", cache_dir="llm_weights")



def get_model_tokenzier(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="llm_weights", trust_remote_code=True, device_map="auto") # torch_dtype=torch.float16, low_cpu_mem_usage=True, 
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True, cache_dir = "llm_weights")
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(' model size --->', trainable_params)
    return model, tokenizer


def testing_output(model_name, input):
    model, tokenizer = get_model_tokenzier(model_name)

    input_ids = tokenizer.encode(input, return_tensors="pt").to(model.device)
    input_len = input_ids.size()[-1]
    max_len = input_len+50

    output = model.generate(input_ids, num_return_sequences=1, max_new_tokens=200) # do_sample=True, top_p=0.95, top_k=60, temperature=1, max_length=max_len, 
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(' ')
    print(output_text)

    return output_text




# def demo_print(model_name, example):
#     # print(' ')
#     # print(' ')
#     # print(' =========================== full iput =========================== ')
#     # full_input = example['instruction'] + ' ' + example['context'] + ' '
#     # print(f"{full_input}")
#     # print(' ')


#     output_text2 = testing_output(model_name + 'fullmodel/', example)
#     print(f"==>> fullmodel: {output_text2}")

#     output_text3 = testing_output(model_name + 'random/', example)
#     print(f"==>> random: {output_text3}")
    
model_name = "meta-llama/Llama-2-13b-chat" # "meta-llama/Llama-2-7b-chat-hf"


from prompt_functions import convert_to_llama_chat_template, llama_prompt, general_prompt

#raw_text = "The Cable News Network (CNN) is a multinational news channel and website headquartered in Atlanta, Georgia, U.S.[2][3][4] Founded in 1980 by American media proprietor Ted Turner and Reese Schonfeld as a 24-hour cable news channel, and presently owned by the Manhattan-based media conglomerate Warner Bros. Discovery (WBD),[5] CNN was the first television channel to provide 24-hour news coverage and the first all-news television channel in the United States.[6][7][8][9][10]"
raw_text = """At the end of 1953, a team of young men placed as a dream and objective, the foundation of an association with national and athletic aims based on promoting the education and social skills of its young members. On 14 April 1954, the general assembly of these members with leader Mr Christakis Pavlides proposes the foundation of an athletic association called "APOLLON LIMASSOL".[1] The assembly approved the proposal and thus from that date "APOLLON was born". The first administrative council of the team included: Charalambos Lymbourides (Secretary), Andreas Psyllides (Cashier), Antonakis Fourlas (Adviser), Melis Charalampous (Adviser), Andreas Theoharous (Adviser) Andreas Aggelopoulos (Adviser) and Kostas Panayiotou (Adviser).[citation needed]"""


text = llama_prompt("A", raw_text)
testing_output(model_name, text)
