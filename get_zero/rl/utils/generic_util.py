from omegaconf import OmegaConf
from omegaconf import DictConfig
from typing import Dict

def remove_base_omegaconf_resolvers():
    OmegaConf.clear_resolver('eq')
    OmegaConf.clear_resolver('contains')
    OmegaConf.clear_resolver('if')
    OmegaConf.clear_resolver('resolve_default')

def register_isaacgym_cfg_resolvers():
    OmegaConf.register_new_resolver('eq', lambda x, y: x.lower()==y.lower())
    OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
    OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
    OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg=='' else arg)

def register_custom_omegaconf_resolvers():
    OmegaConf.register_new_resolver('add_int', lambda x, y: int(x) + int(y))
    OmegaConf.register_new_resolver('and', lambda x, y: x and y)
    OmegaConf.register_new_resolver('mul', lambda x, y: x * y)

def register_all_omegaconf_resolvers():
    register_custom_omegaconf_resolvers()
    register_isaacgym_cfg_resolvers()

def omegaconf_to_dict(d: DictConfig)->Dict:
    """Converts an omegaconf DictConfig to a python Dict, respecting variable interpolation."""
    ret = {}
    for k, v in d.items():
        if isinstance(v, DictConfig):
            ret[k] = omegaconf_to_dict(v)
        else:
            ret[k] = v
    return ret
