"""
Prints DoF pose for a specific grasp cache entry. Used in automation scripts.
"""

from argparse import ArgumentParser
import os
import numpy as np

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--config_name', default='original', help='name of the LEAP hand hardware configuration. Can also be `original`')
    parser.add_argument('--cache_name', default='leap_hand_in_palm_cube_grasp_50k_s09.npy', help='name of grasp cache .npy file in cache/leap_hand/<config_name> folder')
    parser.add_argument('--grasp_index', type=int, default=0, help='Index of grasp in the grasp cache to retrieve')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    cache_file_path = os.path.join('..', 'cache', 'leap_hand', args.config_name, args.cache_name)
    cache = np.load(cache_file_path)
    dof_count = cache.shape[1] - 7 # 7 numbers for cube DoF

    print(' '.join([str(x) for x in cache[args.grasp_index, :dof_count]]))
