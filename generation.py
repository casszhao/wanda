from rouge import FilesRouge

files_rouge = FilesRouge()
# scores = files_rouge.get_scores(hyp_path, ref_path)
# or
context = "generation_output/pseudo_model/pseudo_context"
instruction = "generation_output/pseudo_model/pseudo_instruction"
ref_path  = "./generation_output/pseudo_model/pseudo_output_ref"
hyp_path  = "./generation_output/pseudo_model/pseudo_output"

scores = files_rouge.get_scores(hyp_path, ref_path, avg=True)
print(f"==>> scores: {scores}")


quit()


import torch
from instruct_pipeline import InstructionTextGenerationPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from datasets import load_dataset_builder


data_name = "databricks/databricks-dolly-15k"


ds_builder = load_dataset_builder(data_name)



print(ds_builder.info.description)
print(ds_builder.info.features)



generate_text = pipeline(model="databricks/dolly-v2-3b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto", return_full_text=True)



# template for an instrution with no input
prompt = PromptTemplate(
    input_variables=["instruction"],
    template="{instruction}")

# template for an instruction with input
prompt_with_context = PromptTemplate( input_variables=["instruction", "context"], template="{instruction}\n\nInput:\n{context}")

hf_pipeline = HuggingFacePipeline(pipeline=generate_text)

llm_chain = LLMChain(llm=hf_pipeline, prompt=prompt)
llm_context_chain = LLMChain(llm=hf_pipeline, prompt=prompt_with_context)



model_name = "databricks/dolly-v2-3b"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir = "llm_weights") # padding_side="left", 
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir = "llm_weights") # device_map="auto",  torch_dtype=torch.bfloat16, 

generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)
