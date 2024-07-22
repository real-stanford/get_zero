"""
Given a W&B tag and leap embodiment, finds the local run directory that is associated with the tag and leap embodiment. If there are multiple runs with same embodiment and tag, then returns the best performing run according to the evaluation metric.
"""

from argparse import ArgumentParser
import wandb
import os
from train_leap_runner import get_status_by_embodiment

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--wandb_tag')
    parser.add_argument('--embodiment_name', help='includes name and file extension, but should not include full path')
    return parser.parse_args()

def wandb_get_labeled_experiments(tag, project_path='get_zero'):
    """
    Searches for WandB runs by `project_path` (`<user>/<project>` where `<user>` is optional) and `tag`, then returns list of config files
    """
    api = wandb.Api()
    runs = api.runs(project_path, {"tags": tag})
    experiments = {} # maps from experiment name to run

    for run in runs:
        config_val = run.config
        experiments[run.name] = config_val

    return experiments

if __name__ == '__main__':
    args = parse_args()
    
    data_by_embodiment = get_status_by_embodiment(args.wandb_tag, enable_logging=False)
    data_cur_embodiment = data_by_embodiment[args.embodiment_name]
    best_run = data_cur_embodiment['best_run']

    if best_run:
        print(best_run.name)
    else:
        raise Exception(f'Could not find requested asset {args.asset_name} with requested tag {args.wandb_tag}')
