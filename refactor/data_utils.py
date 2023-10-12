# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import random
from typing import List, Optional, Tuple
from datasets import load_dataset
import torch
from transformers import AutoTokenizer
from datasets.arrow_dataset import Dataset

def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # Load and process wikitext2 dataset
    import pdb; pdb.set_trace()
    # Load train and test datasets
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

# Load and process c4 dataset
def get_c4(
        nsamples: int,
        seed: int,
        seqlen: int,
        tokenizer: AutoTokenizer
    ) -> Tuple[List[Tuple[torch.Tensor]], torch.Tensor]:
    """
    c4 dataloader
    Args:
        nsamples (int, optional): number of samples for the forward pass. Defaults to 128.
        seed (int, optional): the random seed. Defaults to 0.
        seqlen (int, optional): the max sequence length. Defaults to 2048.
        tokenizer (_type_, optional): tokenizer to . Defaults to None.

    Returns:
        _type_: _description_
    """
    # Load train and validation datasets
    traindata: Dataset = load_dataset(
        'allenai/c4',
        'allenai--c4',
        data_files={'train': 'en/c4-train.00000-of-01024.json.gz'},
        split='train',
        cache_dir="calibration_data/"
    )
    valdata: Dataset = load_dataset(
        'allenai/c4',
        'allenai--c4',
        data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
        split='validation',
        cache_dir="calibration_data/"
    )

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    # # Generate samples from training set
    # random.seed(seed)
    # trainloader = []
    # # get all indxs of traindata
    # indxs: List[int] = list(range(len(traindata)))
    # # shuffle them to get random ids
    # random.shuffle(indxs)
    # # keep the first `nsamples`
    # sample_ids: List[int] = indxs[:nsamples]
    # for indx in sample_ids:
    #     # tokenize the training document
    #     trainenc = tokenizer(
    #         traindata[indx]['text'],
    #         return_tensors='pt',
    #         truncation=True,
    #         max_length=seqlen
    #     )
    for _ in range(nsamples):
        while True:
            random_int = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(
                traindata[random_int]['text'],
                return_tensors='pt',
            )
            if trainenc.input_ids.shape[1] > seqlen:
                break
        random_int = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = random_int + seqlen
        inp = trainenc.input_ids[:, random_int:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    # Prepare validation dataset
    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    return trainloader, valenc

# Function to select the appropriate loader based on dataset name
def get_loaders(
        name: str,
        nsamples: Optional[int]=128,
        seed: Optional[int]=0,
        seqlen: Optional[int]=2048,
        tokenizer: Optional[AutoTokenizer]=None
    ) -> Tuple[List[Tuple[torch.Tensor]], torch.Tensor]:
    """
    selector on the dataloaders to return
    Args:
        name (str): the dataset name
        nsamples (int, optional): number of samples for the forward pass. Defaults to 128.
        seed (int, optional): the random seed. Defaults to 0.
        seqlen (int, optional): the max sequence length. Defaults to 2048.
        tokenizer (_type_, optional): tokenizer to . Defaults to None.

    Returns:
        _type_: _description_
    """
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)