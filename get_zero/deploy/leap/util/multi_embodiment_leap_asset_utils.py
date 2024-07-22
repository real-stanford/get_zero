"""
Utilities specfic to this repository for loading LEAP assets.
"""

from typing import Dict
import yaml
from get_zero.distill.utils.embodiment_util import EmbodimentProperties

def get_leap_embodiment_properties(embodiment_name: str, joint_name_to_joint_i: Dict[str, int] = None):
    if joint_name_to_joint_i is None:
        if embodiment_name == 'original':
            config_path = f'../../rl/cfg/embodiment/LeapHand/OrigRepo.yaml'
        else:
            config_path = f'../../rl/cfg/embodiment/LeapHand/generated/{embodiment_name}.yaml'
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        joint_name_to_joint_i = cfg['joint_name_to_joint_i']

    if embodiment_name == 'original':
        asset_path = f'../../rl/assets/leap/leap_hand/original.urdf'
    else:
        asset_path = f'../../rl/assets/leap/leap_hand/generated/urdf/{embodiment_name}.urdf'
    with open(asset_path) as f:
        urdf = f.read()
    
    return EmbodimentProperties(embodiment_name, urdf, joint_name_to_joint_i)
