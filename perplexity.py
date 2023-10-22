import argparse
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from lib.prune import prune_wanda, prune_magnitude, prune_sparsegpt, prune_ablate, check_sparsity, find_layers
from lib.eval import eval_ppl

print('torch', version('torch'))
print('transformers', version('transformers'))
#print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

def get_llm(model_name, cache_dir="llm_weights", device: str = 'auto'):
    # by cass
    assert device in ['auto', 'cuda', 'cpu', 'mps']
    if device == 'cpu':
        torch_dtype=torch.float32
        
    else:
        torch_dtype=torch.float16
    
    print(f"==>> torch_dtype: {torch_dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        device_map=device,
        trust_remote_code=True
    )

    try: model.seqlen = model.config.max_position_embeddings
    except: model.seqlen = 2048 # for falcon-7b-instruct
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="facebook/opt-iml-30b", type=str, help='LLaMA model', 
                        choices=["decapoda-research/llama-7b-hf", 
                                 "decapoda-research/llama-13b-hf",
                                 "decapoda-research/llama-30b-hf",

                                 "tiiuae/falcon-7b-instruct", 
                                 "tiiuae/falcon-40b-instruct",

                                 "facebook/opt-iml-1.3b",
                                 "facebook/opt-iml-30b",

                                 "NousResearch/Nous-Hermes-llama-2-7b",
                                 "NousResearch/Nous-Hermes-Llama2-13b"
                                 ])
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=100, help='Number of calibration samples.') # 128
    parser.add_argument('--sparsity_ratio', type=float, default=0.5, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, default="unstructured", choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str, default="wanda", choices=["magnitude", "wanda", "sparsegpt", "ablate_magnitude", "ablate_wanda"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default="results", help='Path to save results.')
    parser.add_argument('--save_model', type=str, default="saved_model", help='Path to save the pruned model.')
    parser.add_argument('--device', type=str, default="auto", help='device for experiments')

    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # # Handling n:m sparsity
    # prune_n, prune_m = 0, 0
    # if args.sparsity_type != "unstructured":
    #     assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
    #     prune_n, prune_m = map(int, args.sparsity_type.split(":"))
    # original_ppl_train, original_ppl_test = eval_ppl(model, tokenizer, device)
    # print("".center(50, "-"))
    # print(f"===> original model ----> ppl on wikitext_train {original_ppl_train}, wikitext_test {original_ppl_test}")


    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}/{args.prune_method}")
    pruned_model_path = f"pruned_model/{model_name}/{args.prune_method}"

    if torch.cuda.device_count() > 0:
        print(f"\n \n using {torch.cuda.device_count()} GPUs \n ")
        model = get_llm(pruned_model_path, args.cache_dir, device='auto')
    else:
        model = get_llm(pruned_model_path, args.cache_dir, device='auto')
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(pruned_model_path, use_fast=False, cache_dir = args.cache_dir)


    if torch.cuda.device_count() > 1: device = model.hf_device_map["lm_head"]
    else: device = "cuda"




    ################################################################
    print("*"*30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*"*30)
    model_save_path=f"pruned_model/{model_name}/{args.prune_method}"
    os.makedirs(model_save_path, exist_ok=True)


    ################################################################
    ppl_train, ppl_test = eval_ppl(model, tokenizer, device)
    print(f"ppl on wikitext_train {ppl_train}, wikitext_test {ppl_test}")




    ################################################################
    model = get_llm(args.model, args.cache_dir, device='auto')
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, cache_dir = args.cache_dir)

    original_ppl_train, original_ppl_test = eval_ppl(model, tokenizer, device)




    results_save_path = f"results/{model_name}"
    if not os.path.exists(results_save_path):
        os.makedirs(results_save_path)
    save_filepath = os.path.join(results_save_path, f"log_{args.prune_method}.txt")

    with open(save_filepath, "w") as f:
        if "ablate" in args.prune_method:
            print("method\tactual_sparsity\tppl_train\tppl_test", file=f, flush=True)
            print(f"{args.prune_method}\t{sparsity_ratio:.4f}\t{ppl_train:.4f}\t{ppl_test:.4f}", file=f, flush=True)
        else:
            print("method\tactual_sparsity\tppl_test", file=f, flush=True)
            print(f"{args.prune_method}\t{sparsity_ratio:.4f}\t{ppl_test:.4f}", file=f, flush=True)
            print(f"original \t none \t{original_ppl_test:.4f}", file=f, flush=True)

    
if __name__ == '__main__':
    main()