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

    output = model.generate(input_ids, num_return_sequences=1) # do_sample=True, top_p=0.95, top_k=60, temperature=1, max_length=max_len, 
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(' ')

    return output_text




def demo_print(model_name, example):
    print(' ')
    print(' ')
    print(' =========================== full iput =========================== ')
    full_input = example['instruction'] + ' ' + example['context'] + ' '
    print(f"{full_input}")
    print(' ')


    output_text2 = testing_output(model_name + 'fullmodel/', full_input)
    print(f"==>> fullmodel: {output_text2}")

    output_text3 = testing_output(model_name + 'random/', full_input)
    print(f"==>> random: {output_text3}")
    
model_name = "meta-llama/Llama-2-7b-chat-hf"



demo_print(model_name, ''' \n 'instruction': 'please summarise this text', 
                        'context': "looking after elderly parents can be difficult at the best of times .\nbut this man takes caring for his alzheimer 's - suffering mother to another level .\na security guard from china has touched hearts across the country because he takes his 84-year-old mother with him to work on the back of his motorbike every single day , reported the people 's daily online .\nlu xincai , who lives in zhejiang province in eastern china , says that he is scared his mother will get lost if he leaves her at home by herself because she suffers from the degenerative disease .\ndevoted : lu xincai takes his 84-year-old mother to work with him on the back of his motorbike every day .\nhe ties a sash around both of their waists to make sure she does n't fall off\nshe would often go up to the mountains to collect firewood and there were a few occasions when she got lost after dark .\nwhen mr lu 's father passed away earlier this year , he decided to take his mother with him to work because there was no one else who could look after her .\nhis wife works in a different city and his son is still in school .\nafter helping his mother to get up at 5 am every morning , he puts her on the back seat of his motorbike and ties a sash around both of their waists to ensure that she does not fall off .\nmr lu said that he rides the four kilometres to work slowly to make sure his mother feels safe and so that they can chat along the way .\nthe whole journey takes an hour .\neven when at work he checks up on his mother , who has been given her own room by his employers , a bank , to make sure that she has not wandered off somewhere .\nhe said that his mother devoted her life to caring for her children , and now he feels like he has a duty to care for her in return .\nvulnerable : his elderly mother suffers from alzheimer 's and used to get lost when she was left alone\nhe said : ` i was an apple in my mum 's eye , and now she 's my apple . '\n` our mother carried us on her back to the fields when she went to work on the farm and collect firewood when we were young . '\nhe added : ` only if i see her will i feel relaxed .\notherwise i would be afraid is she had wandered away . '",
                        the summary is:  \n ''')
