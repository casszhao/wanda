import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset,load_from_disk
from rouge import FilesRouge, Rouge


files_rouge = FilesRouge()
rouge = Rouge()



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

    output = model.generate(input_ids, num_return_sequences=1) # do_sample=True, top_p=0.95, top_k=60, temperature=1, max_length=max_len, 
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(' ')

    return output_text



def demo_print(example):
    print(' ')
    print(' ')
    print(' =========================== full iput =========================== ')
    full_input = example['instruction'] + ' ' + example['context'] + ' '
    print(f"{full_input}")
    print(' ')

    output_text = testing_output("tiiuae/falcon-7b-instruct", full_input)
    print(f" FULL MODEL OUTPUT: {output_text}")
    print(' ')
    print(rouge.get_scores(output_text, example['response'], avg=True)['rouge-l']['f'])


    model_index_name = "falcon-7b-instruct"

    output_text1 = testing_output(f'pruned_model/{model_index_name}/sparsegpt', full_input)
    print(f"==>> sparsegpt: {output_text1}")
    print(' ')
    print(rouge.get_scores(output_text1, example['response'], avg=True)['rouge-l']['f'])

    output_text2 = testing_output(f'pruned_model/{model_index_name}/magnitude/', full_input)
    print(f"==>> magnitude: {output_text2}")
    print(' ')
    print(rouge.get_scores(output_text2, example['response'], avg=True)['rouge-l']['f'])

    output_text3 = testing_output(f'pruned_model/{model_index_name}/wanda/', full_input)
    print(f"==>> wanda: {output_text3}")
    print(' ')
    print(rouge.get_scores(output_text3, example['response'], avg=True)['rouge-l']['f'])



dataset = load_from_disk("data/dolly/") # features = ['instruction', 'context', 'response', 'category']
example = dataset['train'][1]
demo_print(example)

example = dataset['train'][2]
demo_print(example)

example = dataset['train'][3]
demo_print(example)