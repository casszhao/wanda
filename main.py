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
    assert device in ['auto', 'cuda', 'cpu', 'mps']
    if device == 'cpu':
        torch_dtype=torch.float32
    else:
        torch_dtype=torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        device_map=device
    )

    model.seqlen = model.config.max_position_embeddings
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="decapoda-research/llama-7b-hf", type=str, help='LLaMA model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0.5, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, default="unstructured", choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str, default="wanda", choices=["magnitude", "wanda", "sparsegpt", "ablate_magnitude", "ablate_wanda"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default="results", help='Path to save results.')
    parser.add_argument('--save_model', type=str, default="saved_model", help='Path to save the pruned model.')
    parser.add_argument('--device', type=str, default="cuda", help='device for experiments')

    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")

    if torch.cuda.device_count() > 0:
        assert args.device == 'cuda', ("You have a gpu why not use it??")
        model = get_llm(args.model, args.cache_dir, device='auto')
    else:
        model = get_llm(args.model, args.cache_dir, device=args.device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, cache_dir = args.cache_dir)

    device = torch.device(args.device)
    print('   model.hf_device_map ===>', model.hf_device_map)
    # if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
    #     device = model.hf_device_map["lm_head"]
    # print("use device ", device)

    # ppl_train, ppl_test = eval_ppl(model, tokenizer, device)
    # print(f"===> original model ----> ppl on wikitext_train {ppl_train}, wikitext_test {ppl_test}")

    if args.sparsity_ratio != 0:
        print("pruning starts")
        if args.prune_method == "wanda":
            prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "magnitude":
            prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "sparsegpt":
            prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif "ablate" in args.prune_method:
            prune_ablate(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

    ################################################################
    print("*"*30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*"*30)
    ################################################################
    ppl_train, ppl_test = eval_ppl(model, tokenizer, device)
    print(f"ppl on wikitext_train {ppl_train}, wikitext_test {ppl_test}")


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

    model_save_path=f"pruned_model/{model_name}/{args.prune_method}"
    os.makedirs(model_save_path, exist_ok=True)

    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

if __name__ == '__main__':
    main()