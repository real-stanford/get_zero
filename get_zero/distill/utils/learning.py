from omegaconf import DictConfig
from torch.optim import Optimizer
from transformers.optimization import get_inverse_sqrt_schedule
from torch.optim.lr_scheduler import StepLR

def get_lr_scheduler(lr_config: DictConfig, optimizer: Optimizer, total_steps: int, kwargs={}):
    if lr_config['schedule'] == 'ramp':
        warmup_steps = int(total_steps * lr_config.rampup_proportion)
        return get_inverse_sqrt_schedule(optimizer, warmup_steps)
    elif lr_config['schedule'] == 'fixed':
        return StepLR(optimizer, 1, 1, **kwargs)
    else:
        raise NotImplementedError
