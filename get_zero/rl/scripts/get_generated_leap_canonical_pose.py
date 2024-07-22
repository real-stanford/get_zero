"""
Prints DoF pose for a the canonical pose for a LEAP hand configuration. Used in automation scripts.
"""

from argparse import ArgumentParser
import os
import yaml

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--config_name', default='001', help='name of the generated LEAP hand hardware configuration')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    embodiment_config_path = os.path.join('..', 'cfg', 'embodiment', 'LeapHand', 'generated', f'{args.config_name}.yaml')
    with open(embodiment_config_path, 'r') as f:
        config = yaml.safe_load(f)
    pose = config['canonical_pose']

    print(' '.join([str(x) for x in pose]))
