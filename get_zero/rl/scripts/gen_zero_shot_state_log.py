"""
When doing zero-shot to a new embodiment, we need a mock state log that contains only the embodiment information when running distill. This script generates a state log for a range of embodiments if there is not an already existing state log. Names the state log in a way that will appear early alphabetically so that if a real state log is generated later that will be used instead of this temporary state log.
"""

import os
import yaml
import torch
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--log_only', action='store_true')
    parser.add_argument('--state_log_dir', default='../state_logs/LeapHandRot')
    return parser.parse_args()

def gen_zero_shot_state_log(state_log_dir=None, log_only=False):
    NUM_EMBODIMENTS = 633

    if log_only:
        print(f'WARNING: in log_only mode so no files will be created (folders will be created)')

    if state_log_dir is None:
        state_log_dir = os.path.join('..', 'state_logs', 'LeapHandRot')
    cfg_dir = os.path.join('..', 'cfg', 'embodiment', 'LeapHand', 'generated')
    asset_dir = os.path.join('..', 'assets')

    for i in range(1, NUM_EMBODIMENTS + 1):
        embodiment_str = f'{i:03}'
        
        cur_state_dir = os.path.join(state_log_dir, embodiment_str)
        os.makedirs(cur_state_dir, exist_ok=True)
        if len(os.listdir(cur_state_dir)) == 0:
            with open(os.path.join(cfg_dir, f'{embodiment_str}.yaml')) as f:
                cfg = yaml.safe_load(f)

            urdf = open(os.path.join(asset_dir, cfg['asset']['handAsset'])).read()

            result = {
                'embodiment_properties': {
                    'name': cfg['embodiment_name'],
                    'asset_file_contents': urdf,
                    'joint_name_to_joint_i': cfg['joint_name_to_joint_i']
                }
            }

            out_path = os.path.join(cur_state_dir, f'aaa_state_log_embodiment_only.pt')

            if log_only:
                print(out_path)
                print(result)

                print('Exiting after first file since in log_only mode')
                exit()
            else:
                torch.save(result, out_path)
                print(f'Wrote to {out_path}')


if __name__ == '__main__':
    args = parse_args()
    gen_zero_shot_state_log(args.state_log_dir, args.log_only)
