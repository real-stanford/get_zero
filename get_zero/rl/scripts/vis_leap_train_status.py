"""Plots histogram of training performance for the 236 generated LEAP embodiments"""

from matplotlib import pyplot as plt
from train_leap_runner import get_status_by_embodiment, run_to_metric
import os
import math
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--wandb_proj', default='get_zero', help='project name to use on W&B')
    parser.add_argument('--wandb_tag', default='leap_train_get_zero', help='tag to use on W&B')
    parser.add_argument('--max_metric', type=int, default=30, help='min metric needed to consider training success') # metric is average time to complete full rotation
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    status = get_status_by_embodiment(args.wandb_tag, max_metric=args.max_metric, wandb_proj=args.wandb_proj)
    
    best_scores = []
    embodiment_names_under_max_metric = []
    embodiment_names_missing_best_run = []
    num_embodiments_under_max_metric_connectivity = 0
    num_embodiments_under_max_metric_link_length_001 = 0
    num_embodiments_under_max_metric_link_length_train_val = 0
    for embodiment_name, data in status.items():
        best_run = data['best_run']

        if best_run is None:
            embodiment_names_missing_best_run.append(embodiment_name)
        else:    
            best_metric = run_to_metric(best_run)
            if best_metric != float('inf'):
                best_scores.append(best_metric)
            if best_metric > 0 and best_metric < args.max_metric:
                embodiment_names_under_max_metric.append(embodiment_name)
                if int(embodiment_name) <= 236:
                    num_embodiments_under_max_metric_connectivity += 1
                elif int(embodiment_name) <= 579:
                    num_embodiments_under_max_metric_link_length_001 += 1
                else:
                    num_embodiments_under_max_metric_link_length_train_val += 1

    print(f'Connectivity variation embodiments ({num_embodiments_under_max_metric_connectivity}) with 0 < best_metric < max_metric')
    print(f'001 link length variation embodiments ({num_embodiments_under_max_metric_link_length_001}) with 0 < best_metric < max_metric')
    print(f'Train val link length variation embodiments ({num_embodiments_under_max_metric_link_length_train_val}) with 0 < best_metric < max_metric')

    """Generate histogram"""
    if len(best_scores) > 0:
        bins = [args.max_metric*i for i in range(math.ceil(min(max(best_scores), 400)/args.max_metric))]
        _, _, patches = plt.hist(best_scores, bins)
        plt.bar_label(patches)
        
        plt.xticks(bins)
        out_dir = 'tmp'
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'leap_train_status_histogram.png')
        plt.xlabel('Seconds to complete full 2 Pi rotation')
        plt.ylabel('Number of embodiments')
        plt.title(f'Histogram of best LEAP RL policy for {len(best_scores)} generated embodiments')
        plt.savefig(out_path, dpi=300)
        print(f'Wrote to {out_path}')

    """Log satisfactory embodiments"""
    embodiment_names_under_max_metric.sort()
    print(f'Embodiments ({len(embodiment_names_under_max_metric)}) with 0 < best_metric < max_metric:\n{embodiment_names_under_max_metric}')

    """Log missing best run embodiments"""
    embodiment_names_missing_best_run.sort()
    print(f'Embodiments ({len(embodiment_names_missing_best_run)}) missing best run:\n{embodiment_names_missing_best_run}')
