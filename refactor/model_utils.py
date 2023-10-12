from typing import Optional
import torch
import numpy as np
from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set seed for reproducibility
def set_seed(seed: int):
    """sets random seed"""
    np.random.seed(seed)
    torch.random.manual_seed(seed)

def load_model(
        model_name: str,
        cache_dir: Optional[str] = None,
        device: Optional[str] = 'auto'
    ) -> AutoModelForCausalLM:
    """
    returns a pretrained model
    Args:
        model_name (str): huggingface hub handle
        cache_dir (Optional[str], optional):  local cache dir . Defaults to None.
        device (Optional[str], optional): device to load the model on. Defaults to 'auto'.

    Returns:
        AutoModelForCausalLM: the pretrained model
    """
    logger.info(f"Loading {model_name}")
    assert device in ['auto', 'cuda', 'cpu', 'mps']
    if device in ('cpu', 'mps'):
        torch_dtype=torch.float32
    else:
        torch_dtype=torch.float16
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True,
        )
    except Exception as e:
        logger.error(f"Error at loading model step \n {e}")
    model.seqlen = model.config.max_position_embeddings
    logger.success(f"Model {model_name} loaded!")
    return model

def load_tokenizer(
        model_name: str,
        cache_dir: Optional[str] = None
    ) -> AutoTokenizer:
    """
    loads a transformers pretrained tokenizer
    Args:
        model_name (str): huggingface hub handle
        cache_dir (str): local cache dir (optional)

    Returns:
        AutoTokenizer: the pretrained tokenizer
    """

    return AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        cache_dir = cache_dir
    )