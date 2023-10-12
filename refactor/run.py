import argparse
import numpy as np
import sys
import torch
from loguru import logger
from model_utils import (
    load_model, load_tokenizer, set_seed
)
from compression.wanda import prune_wanda
from config import settings

def pipeline(args):

    set_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))


    model = load_model(
        model_name = args.model,
        cache_dir=settings.dev.cache_dir,
        device = settings.dev.device
    )

    tokenizer = load_tokenizer(
        model_name= args.model,
        cache_dir=settings.dev.cache_dir
    )

    model.to(settings.dev.device)

    if args.sparsity_ratio != 0:
        logger.info(f"prunning with {args.prune_method}")
        if args.prune_method == "wanda":
            prune_wanda(
                args,
                model,
                tokenizer,
                settings.dev.device,
                prune_n=prune_n,
                prune_m=prune_m
            )

    import pdb; pdb.set_trace()


if __name__ == '__main__':

    logger.add(
        sys.stderr,
        level="ERROR",
        format='{level}: {time} [{name}] {message}',
        serialize=True,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="decapoda-research/llama-7b-hf", type=str, help='LLaMA model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0.5, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, default="unstructured", choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str, default="wanda", choices=["magnitude", "wanda", "sparsegpt", "ablate_magnitude", "ablate_wanda"])

    args = parser.parse_args()

    logger.info(args)

    pipeline(args)