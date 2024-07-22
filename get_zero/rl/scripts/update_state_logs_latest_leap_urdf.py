"""
The state logs contain URDF file contents. If the URDF files are updated and you want to update the latest state logs with the new version, run this script.
"""

import os
from glob import glob
import torch
from tqdm import tqdm

fix_dir_prefix = 'zzz_fixed_' # zzz puts the fixed version in the last sorted order so that it will be used as the most recent version

urdf_dir = '../assets/leap/leap_hand/generated/urdf'
log_dir = '../state_logs/LeapHandRot'

embodiment_names = os.listdir(log_dir)
embodiment_names.sort()
for embodiment_name in tqdm(embodiment_names):
    print(f'beginning to process {embodiment_name}')
    cur_fpaths = glob(os.path.join(log_dir, embodiment_name, '*'))
    cur_fpaths.sort(reverse=True)
    dir_i = 0
    while fix_dir_prefix in cur_fpaths[dir_i]: # ignore existing fixed state logs
        dir_i += 1
    cur_fpath = cur_fpaths[dir_i]
    
    data = torch.load(cur_fpath, map_location='cpu')

    new_urdf_file_contents = open(os.path.join(urdf_dir, f'{embodiment_name}.urdf')).read()
    data['embodiment_properties']['asset_file_contents'] = new_urdf_file_contents

    out_dir = os.path.dirname(cur_fpath)
    out_fname = f'{fix_dir_prefix}{os.path.basename(cur_fpath)}'
    out_path = os.path.join(out_dir, out_fname)

    assert out_path != cur_fpath, 'do not want to overwrite original file'

    torch.save(data, out_path)
    print(f'saving to {out_path}')

