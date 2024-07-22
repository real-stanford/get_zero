"""
Multi embodiment replacement for `leap_hand_utils.py` from LEAP_Hand_Sim. Borrows heavily from `leap_hand_utils.py`.
"""

from get_zero.distill.utils.embodiment_util import EmbodimentProperties
import numpy as np
import torch

'''
Embodiments:

LEAPhand: Real LEAP hand (180 for the motor is actual zero)
LEAPsim:  Leap hand in sim (has allegro-like zero positions)
one_range: [-1, 1] for all joints to facilitate RL
allegro:  Allegro hand in real or sim
'''
    
"""compatability layer so we can call util functions either with torch or numpy"""
def maintain_tensor(f):
    def new_f(self, x):
        is_torch = type(x) == torch.Tensor
        if is_torch:
            device = x.device
            x = x.detach().cpu().numpy()
        y = f(self, x)
        if is_torch:
            y = torch.tensor(y, device=device)

        return y
    return new_f
    
class MultiEmbodimentLeapHandUtils:
    """
    Utilities for managing the sim2real dof reordering as well as limit change from -1 to 1 to actual DoF limits.

    sim range (or sim ones) refers to joints in the range -1 to 1
    limit range refers to joitns in the range from lower limit to upper limit (the joint range directly pulled from the URDF file)
    leap range refers to limit range with an additional offset of Pi added (measured in radians; this is the dof range of the real motors)

    All utilities can be called with either torch tensors or numpy arrays.
    """

    def __init__(self, embodiment_properties: EmbodimentProperties):
        self.embodiment_properties = embodiment_properties

        # sim2real ordering
        self.motors_real_ordering = sorted([joint_name for joint_name in embodiment_properties.joint_name_to_joint_i.keys()], key=lambda x: int(x))
        real_to_sim_indices = [None] * self.embodiment_properties.dof_count
        sim_to_real_indices = [None] * self.embodiment_properties.dof_count
        for joint_name, joint_i in self.embodiment_properties.joint_name_to_joint_i.items():
            sim_index = joint_i
            real_index = self.motors_real_ordering.index(joint_name)

            real_to_sim_indices[sim_index] = real_index
            sim_to_real_indices[real_index] = sim_index

        self.sim_to_real_indices = sim_to_real_indices
        self.real_to_sim_indices = real_to_sim_indices
        self.motors_sim_ordering = [self.motors_real_ordering[self.real_to_sim_indices[i]] for i in range(len(self.motors_real_ordering))]

        # limits
        self.lower_limits_sim_ordering = self.embodiment_properties.joint_properties['joint_angle_limits'][:, 0]
        self.upper_limits_sim_ordering = self.embodiment_properties.joint_properties['joint_angle_limits'][:, 1]
        self.lower_limits_real_ordering = self.sim_ordering_to_real_ordering(self.lower_limits_sim_ordering)
        self.upper_limits_real_ordering = self.sim_ordering_to_real_ordering(self.upper_limits_sim_ordering)

    def get_motors(self):
        """
        returns list of motor names for the motors present
        """
        return self.motors_real_ordering

    @maintain_tensor
    def sim_ordering_to_real_ordering(self, values_sim_ordering):
        return values_sim_ordering[self.sim_to_real_indices]

    @maintain_tensor
    def real_ordering_to_sim_ordering(self, values_real_ordering):
        return values_real_ordering[self.real_to_sim_indices]

    @maintain_tensor
    def angle_safety_clip_real_ordering_leap_range(self, joints_real_ordering_leap_range):
        """Can call this right before you send commands to the hand"""
        min_real_ordering = add_real_offset(self.lower_limits_real_ordering)
        max_real_ordering = add_real_offset(self.upper_limits_real_ordering)
        return np.clip(joints_real_ordering_leap_range, min_real_ordering, max_real_ordering)
    
    @maintain_tensor
    def angle_safety_clip_sim_ordering_limit_range(self, joints_sim_ordering_limit_range):
        return np.clip(joints_sim_ordering_limit_range, self.lower_limits_sim_ordering, self.upper_limits_sim_ordering)
    
    @maintain_tensor
    def sim_range_real_ordering_to_leap_range_real_ordering(self, sim_ones_real_ordering):
        joints = scale(sim_ones_real_ordering, self.lower_limits_real_ordering, self.upper_limits_real_ordering)
        joints = add_real_offset(joints)
        return joints
    
    @maintain_tensor
    def leap_range_real_ordering_to_sim_range_real_ordering(self, joints_real_ordering_leap_range):  
        joints = remove_real_offset(joints_real_ordering_leap_range)
        joints = unscale(joints, self.lower_limits_real_ordering, self.upper_limits_real_ordering)
        return joints
    
    def get_default_position_real_ordering_leap_range(self):
        """
        Default position is with hand fully open (all joints at 3.14)
        """
        return np.array([np.pi] * self.embodiment_properties.dof_count)
    
    @maintain_tensor
    def sim_range_sim_ordering_to_limit_range_sim_ordering(self, joint_position_sim_ordering_sim_range):
        return scale(joint_position_sim_ordering_sim_range, self.lower_limits_sim_ordering, self.upper_limits_sim_ordering)
    
    @maintain_tensor
    def limit_range_sim_ordering_to_sim_range_sim_ordering(self, joint_position_sim_ordering_limit_range):
        return unscale(joint_position_sim_ordering_limit_range, self.lower_limits_sim_ordering, self.upper_limits_sim_ordering)
    
    @maintain_tensor
    def clip_limits_range_sim_ordering(self, x):
        return np.clip(x, self.lower_limits_sim_ordering, self.upper_limits_sim_ordering)

    @maintain_tensor
    def clip_limits_range_real_ordering(self, x):
        return np.clip(x, self.lower_limits_real_ordering, self.upper_limits_real_ordering)
    
def add_real_offset(joints_any_ordering):
    joints_any_ordering = np.array(joints_any_ordering)
    ret_joints = joints_any_ordering + 3.14159
    return ret_joints

def remove_real_offset(joints_any_ordering):
    joints_any_ordering = np.array(joints_any_ordering)
    ret_joints = joints_any_ordering - 3.14159
    return ret_joints

#this goes from [-1, 1] to [lower, upper]
def scale(x, lower, upper):
    return (0.5 * (x + 1.0) * (upper - lower) + lower)
#this goes from [lower, upper] to [-1, 1]
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower)/(upper - lower)
