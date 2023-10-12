import torch
from torch import nn

def get_layers_by_model(
        name_or_path: str,
        model
    ) -> torch.nn.modules.container.ModuleList:
    """
    Gets layers by model_name
    Args:
        model_name (str): the model name
        model (_type_): the actual laded model

    Returns:
        torch.nn.modules.container.ModuleList: the layer moduyle list
    """

    if 'opt' in name_or_path:
        return model.base_model.decoder.layers
    elif 'gpt2' in name_or_path:
        return [v for k,v in model._modules.items()][0].h
    elif 'llama' in name_or_path:
        return model.model.layers
    else:
        try:
            return model.base_model.decoder.layers
        except Exception as exc:
            raise NotImplementedError(exc) from exc


def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child,
                layers=layers,
                name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    # layers = model.h
    layers = [v for k,v in model._modules.items()][0].h
    count = 0
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache
    return float(count)/total_params

def prepare_calibration_input(model, dataloader, device):
    """
    preps the calibration data for pruning
    Args:
        model (_type_): _description_
        dataloader (_type_): _description_
        device (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = get_layers_by_model(
        model=model,
        name_or_path=model.config._name_or_path
    )

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            try:
                cache['position_ids'] = kwargs['position_ids']
            except:
                pass
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch_no, batch in enumerate(dataloader):
        try:
            model(batch[0].to(device))
        except ValueError:
            print(batch_no)
            pass
    layers[0] = layers[0].module


    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache
    import pdb; pdb.set_trace()
    return inps, outs, attention_mask, position_ids, layers

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity