from yourdfpy import URDF
import pytorch_kinematics as pk
import torch
from torch import Tensor
from typing import Dict
import sys

@torch.inference_mode()
def fk(robot: URDF, thetas: Tensor, joint_name_to_joint_i: Dict[str, int]):
    batch_size = thetas.size(0)
    device = thetas.device
    thetas = thetas.to(torch.float32)

    # pick arbitrary link that is not base link since forward kinematics needs end link specified, but it doesn't seem to matter what link is chosen using pytorch_kinematics, especially considering we are computing FK for every link
    link_names = list(robot.link_map.keys())
    assert hasattr(robot, 'base_link'), 'need to initialize URDF with `build_scene_graph=True` or `build_collision_scene_graph=True` so that base_link is populated'
    link_names.remove(robot.base_link) # any link except the base link works
    arbitrary_link_name = link_names[0]

    # run forward kinematics (supress error output since there are errors about `joint_properties` tag being present)
    save_stderr = sys.stderr
    class DummyFile(object):
        def write(self, x): pass
    sys.stderr = DummyFile()
    chain = pk.build_serial_chain_from_urdf(robot.write_xml_string(), arbitrary_link_name).to(device=device)
    sys.stderr = save_stderr
    
    # run fk
    theta_ordering_pk = chain.get_joint_parameter_names()
    link_name_to_transform = chain.forward_kinematics(thetas, end_only=False) # map from link name -> (batch, 4, 4)

    joint_positions = torch.zeros((batch_size, robot.num_dofs, 3), device=device)
    
    # write resulting positions in the correct sim joint ordering
    for joint_name, joint in robot.joint_map.items():
        joint_i = joint_name_to_joint_i[joint_name]
        assert theta_ordering_pk[joint_i] == joint_name, 'joint_name_to_joint_i does not match joint order that pytorch_kinematics has, so a remapping will need to be implemented in this function'
        child_link_name = joint.child
        joint_transform = link_name_to_transform[child_link_name].get_matrix()
        joint_pos = joint_transform[:, :3, 3] # (batch, 3)
        joint_positions[:, joint_i, :] = joint_pos

    return joint_positions

if __name__ == '__main__':
    file_path = '../rl/assets/leap/leap_hand/generated/urdf/001.urdf'
    thetas = torch.zeros(1, 16, device='cpu')
    robot = URDF.load(file_path, load_meshes=False)
    joint_name_to_joint_i = {'0': 1, '1': 0, '10': 14, '11': 15, '12': 4, '13': 5, '14': 6,
  '15': 7, '2': 2, '3': 3, '4': 9, '5': 8, '6': 10, '7': 11, '8': 13, '9': 12}
    result = fk(robot, thetas, joint_name_to_joint_i)
    print(result)
