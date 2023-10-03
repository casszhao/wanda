import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


prompt = "The following is a conversation with an AI assistant. The assistant is helpful, knowledagable, clever, and very friendly."

########## dataset from Anthropic/hh-rlhf
text1 = """The Cable News Network (CNN) is a multinational news channel and website headquartered in Atlanta, Georgia, U.S.[2][3][4] \
        Founded in 1980 by American media proprietor Ted Turner and Reese Schonfeld as a 24-hour cable news channel, and presently owned by the Manhattan-based media conglomerate Warner Bros. \
        Discovery (WBD),[5] CNN was the first television channel to provide 24-hour news coverage and the first all-news television channel in the United States.
        Question: Which was the first series CNN aired?
        Assistant:"""

text2 =  """
        The nine mile byway starts south of Morehead, Kentucky and can be accessed by U.S. Highway 60. Morehead is a home rule-class city located along US 60 (the historic Midland Trail) and Interstate 64 in Rowan County, Kentucky, in the United States. 
        Question: What U.S Highway gives access to Zilpo Road, and is also known as Midland Trail?
        Assistant:"""

text3 = """
        Amitriptyline is an antidepressant medicine. It works by increasing levels of a chemical called serotonin in your brain. The most commonly encountered side effects of amitriptyline include weight gain and gastrointestinal symptoms like constipation, xerostomia, dizziness, headache, and somnolence. 
        Question: what pill to take for treating chronic pain?
        Assistant:"""

text4 = """
        Cam ordered a pizza and took it home. He opened the box to take out a slice. Cam discovered that the store did not cut the pizza for him. He looked for his pizza cutter but did not find it. He had to use his chef knife to cut a slice.
        Question: Why did Cam order a pizza?
        Assistant:"""

text5 = """
        I thought I lost my hat at the park today. I spent a lot of time looking for it. I was just about to give up when I saw something far away. It was my hat, stuck in a bush! I'm really glad I found it.
        Question: Why I spent a lot of time?
        Assistant:"""

text6 = """
        The Association for Computational Linguistics (ACL) is a scientific and professional organization for people working on natural language processing. Its namesake conference is one of the primary high impact conferences for natural language processing research, along with EMNLP
        Question: What is the deadline of ACL2024?
        Assistant:"""


def get_model_tokenzier(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="llm_weights", low_cpu_mem_usage=True, device_map="auto") # torch_dtype=torch.float16, 
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, cache_dir = "llm_weights")
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(' model size --->', trainable_params)
    return model, tokenizer


def testing_output(model_name, input):
    model, tokenizer = get_model_tokenzier(model_name)

    input_ids = tokenizer.encode(input, return_tensors="pt").to(model.device)
    input_len = input_ids.size()[-1]
    max_len = input_len+50

    output = model.generate(input_ids, max_length=max_len, do_sample=True, top_p=0.95, top_k=60, temperature=0.8, num_return_sequences=1)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(' ')
    print(output_text)
    print(' ')
    print(' ')

print(' ////////////////////////// ')
full_input = prompt + text1 
testing_output('decapoda-research/llama-7b-hf', full_input)
testing_output('pruned_model/llama-7b-hf/magnitude/', full_input)
testing_output('pruned_model/llama-7b-hf/sparsegpt', full_input)
testing_output('pruned_model/llama-7b-hf/wanda/', full_input)