import torch
import random
import numpy as np
import os
from typing import List
from torch import Tensor
from torch.nn import functional as F
from omegaconf import OmegaConf, DictConfig, open_dict

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def pad_stack_tensors(matrices: List[Tensor], pad_value=0) -> Tensor:
    """
    Pads list of tensors along every dimension so that they are the same size same size. All tensors in the list must have the same number of dimensions. Uses the size of the largest matrix.
    """
    if len(matrices) == 0:
        return None
    
    max_dims = []
    for dim in range(matrices[0].dim()):
        max_dims.append(max(mat.size(dim) for mat in matrices))
    
    result = []
    for mat in matrices:
        padding_shape = []
        for dim in range(mat.dim()):
            padding_shape.insert(0, max_dims[dim] - mat.size(dim))
            padding_shape.insert(0, 0)
        result.append(F.pad(mat, padding_shape, value=pad_value))

    result = torch.stack(result)

    return result

def tensordict_to_device(d, device):
    if type(d) == dict:
        for k, v in d.items():
            d[k] = tensordict_to_device(v, device)
        return d
    else:
        return d.to(device)

def add_custom_omegaconf_resolvers():
    # currently no custom resolvers
    pass

def assert_and_set(cfg: DictConfig, key: str, value: str):
    """If value is not present in DictConfig, then set it, otherwise assert that the new value is the same as the old value"""
    if key in cfg:
        assert cfg[key] == value
    else:
        OmegaConf.set_struct(cfg, True)
        with open_dict(cfg):
            cfg[key] = value
