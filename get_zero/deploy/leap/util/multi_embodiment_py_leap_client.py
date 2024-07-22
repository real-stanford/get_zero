from .dynamixel_client import DynamixelClient
import numpy as np
from get_zero.distill.utils.embodiment_util import EmbodimentProperties
from .get_zero_leap_hand_utils import MultiEmbodimentLeapHandUtils

class MultiEmbodimentLeapHand:
    """
    "drop-in" replacement for `LeapHand` in LEAP_HAND_Sim/leapsim/hardware_controller.py, but using the Python LEAP API instead of the ROS LEAP API. Also supports LEAP hand with different configurations of present joints
    Built off of `LeapNode` from leap_hand_utils.py from LEAP_Hand_Sim
    """

    def __init__(self, embodiment_properties: EmbodimentProperties):
        self.embodiment_properties = embodiment_properties
        self.lhu = MultiEmbodimentLeapHandUtils(embodiment_properties)

        ####Some parameters
        self.kP = 600
        self.kI = 0
        self.kD = 200
        self.curr_lim = 350
        self.prev_pos = self.curr_pos = self.lhu.get_default_position_real_ordering_leap_range()
           
        #You can put the correct port here or have the node auto-search for a hand at the first 3 ports.
        self.motors = motors = [int(joint_name) for joint_name in self.lhu.get_motors()]
        try:
            self.dxl_client = DynamixelClient(motors, '/dev/ttyUSB0', 4000000)
            self.dxl_client.connect()
        except Exception:
            try:
                self.dxl_client = DynamixelClient(motors, '/dev/ttyUSB1', 4000000)
                self.dxl_client.connect()
            except Exception:
                self.dxl_client = DynamixelClient(motors, 'COM13', 4000000)
                self.dxl_client.connect()
        #Enables position-current control mode and the default parameters, it commands a position and then caps the current so the motors don't overload
        self.dxl_client.sync_write(motors, np.ones(len(motors))*5, 11, 1)
        self.dxl_client.set_torque_enabled(motors, True)
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kP, 84, 2) # Pgain stiffness     
        self.dxl_client.sync_write([0,4,8], np.ones(3) * (self.kP * 0.75), 84, 2) # Pgain stiffness for side to side should be a bit less
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kI, 82, 2) # Igain
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kD, 80, 2) # Dgain damping
        self.dxl_client.sync_write([0,4,8], np.ones(3) * (self.kD * 0.75), 80, 2) # Dgain damping for side to side should be a bit less
        #Max at current (in unit 1ma) so don't overheat and grip too hard #500 normal or #350 for lite
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.curr_lim, 102, 2)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    def set_leap(self, pose_real_ordering_leap_range):
        """
        Receive LEAP pose and directly control the robot
        """
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose_real_ordering_leap_range)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
    
    def set_ones(self, pose_real_ordering_sim_range):
        """
        Sim compatibility, first read the sim value in range [-1,1] and then convert to leap
        """
        pose_real_ordering_leap_range = self.lhu.sim_range_real_ordering_to_leap_range_real_ordering(np.array(pose_real_ordering_sim_range))
        self.set_leap(pose_real_ordering_leap_range)

    def read_pos(self):
        """
        Returns position in real ordering and in leap range
        """
        return self.dxl_client.read_pos()
    
    def read_vel(self):
        return self.dxl_client.read_vel()
    
    def read_cur(self):
        return self.dxl_client.read_cur()

    def command_joint_position(self, desired_pose_sim_ordering_limit_range):
        # TODO: this has a lot of redundancy in the conversion; clean up implementation
        desired_pose_sim_ordering_sim_range = self.lhu.limit_range_sim_ordering_to_sim_range_sim_ordering(desired_pose_sim_ordering_limit_range)
        desired_pose_real_ordering_sim_range = self.lhu.sim_ordering_to_real_ordering(desired_pose_sim_ordering_sim_range)
        self.set_ones(desired_pose_real_ordering_sim_range)

    def poll_joint_position(self):
        """
        returns position in sim ordering with limit range
        """
        joint_position_real_ordering_leap_range = self.read_pos()
        joint_position_real_ordering_sim_range = self.lhu.leap_range_real_ordering_to_sim_range_real_ordering(joint_position_real_ordering_leap_range)
        joint_position_sim_ordering_sim_range = self.lhu.real_ordering_to_sim_ordering(joint_position_real_ordering_sim_range)
        joint_position_sim_ordering_limit_range = self.lhu.sim_range_sim_ordering_to_limit_range_sim_ordering(joint_position_sim_ordering_sim_range)

        return (joint_position_sim_ordering_limit_range, None)

    def reset_hand_pos(self):
        self.set_leap([np.pi] * self.embodiment_properties.dof_count)
    
    def close_connection_keep_position(self):
        """The idea is to close the serial connection to the hand without disabling the torque control on the motors, which will leave the motors in the last place they were commanded to"""
        self.dxl_client.port_handler.closePort()
