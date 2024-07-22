from isaacgymenvs.tasks.base.vec_task import VecTask
import numpy as np
from get_zero.rl.utils.logging_util import crop_isaacgym_viewer_sidebar
from get_zero.distill.utils.embodiment_util import ObservationTokenizer, StateLogger, EmbodimentProperties
from get_zero.utils.forward_kinematics import fk
from typing import Dict, Any, Tuple
import torch
from isaacgym import gymapi, gymtorch, gymutil
from isaacgym.torch_utils import unscale, torch_rand_float, to_torch
import os
import shutil
from tqdm import tqdm
from yourdfpy import URDF

from get_zero.rl.utils.generic_util import remove_base_omegaconf_resolvers
remove_base_omegaconf_resolvers() # removing since leapsim will add back these resolvers
from leapsim.tasks import LeapHandRot as LeapHandRotOrig

SCREEN_CAPTURE_RESOLUTION = (1027, 768)

class LeapHandRot(LeapHandRotOrig):
    """
    Changes:
    - Base LeapHandRot task doesn't add cameras and doesn't have virtual screen capture support, which is addressed in this class to allow videos to properly be recorded.
    - Updated debug logging to additionally log the entire observation vector
    - Support for tokenizing the observation for use with transformer models
    - Support for loading in different LEAP hand hardware configurations that have varying DoF counts
    """

    # LeapHandRotOrig doesn't inherit from IsaacGymEnvs VecTask, so need to copy in metadata attribute
    metadata = VecTask.metadata

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture=None, force_render=None):
        self.is_test = False # set by self.start() function
        self.run_dir = None # set by self.start() function
        self.enable_camera_sensors = cfg["env"]["enableCameraSensors"]
        self.camera_resolution = cfg["env"]["cameraResolution"]
        self.force_render = force_render
        self.tokenize_observation_for_policy = cfg["env"]["tokenizeObservationForPolicy"]
        self.disable_randomizations_during_test = cfg["disableRandomizationsDuringTest"]
        self.log_state_in_test = cfg["env"]["logStateInTest"]
        self.log_state_suffix = cfg["env"]["logStateSuffix"]
        self.dof_count = cfg["env"]["dofCount"] # TODO: technically this is part of embodiment properties, so it doesn't need to be specified as separate value
        self.debug_logging = "debug" in cfg["env"]
        self.absolute_counter = 0 # increases for every step taken without reset, unlike self.global_counter which resets when `self.start` is called
        self.asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../assets')
        hand_asset_file = cfg['env']['asset']['handAsset']
        self.hand_asset_name = os.path.basename(hand_asset_file).replace('.urdf', '')

        # forward kinematics visualization
        self.fk_estimate = None
        self.enable_fk_vis = cfg['env']['enableForwardKinematicsVis'] if 'enableForwardKinematicsVis' in cfg['env'] else False # backwards compatibility when this parameter wasn't defined
        self.urdf = URDF.load(os.path.join(self.asset_root, hand_asset_file), load_meshes=False)

        # load embodiment properties
        self.joint_name_to_joint_i = cfg['env']['joint_name_to_joint_i']
        with open(os.path.join(self.asset_root, hand_asset_file), 'r') as f:
            urdf = f.read()
        self.embodiment_properties = EmbodimentProperties(self.hand_asset_name, urdf, self.joint_name_to_joint_i)

        # set action count
        cfg["env"]["numActions"] = self.dof_count

        # set observation size (based on whether we are using tokenized policy or original policy)
        if self.tokenize_observation_for_policy:
            self.observation_tokenizer = ObservationTokenizer(cfg["env"]["tokenization"], self.embodiment_properties, rl_device, cfg['env']['numEnvs'])
            cfg["env"]["numObservations"] = self.observation_tokenizer.tokenized_obs_size
        else:
            cfg["env"]["numObservations"] = (self.dof_count + self.dof_count + 2) * 3 # (joint pos, joint target, phase) * history size

        # Set the grasp cache to reference grasps for the correct embodiment used
        cfg["env"]["grasp_cache_name"] = os.path.join('leap_hand', self.hand_asset_name, cfg["env"]["grasp_cache_name"])
        
        # LeapHandRot inherits from VecTaskRot (which doesn't create virutal display) rather than IsaacGymEnvs VecTask which does, so we must manually do so here
        self.virtual_display = None
        if virtual_screen_capture:
            from pyvirtualdisplay.smartdisplay import SmartDisplay
            self.virtual_display = SmartDisplay(size=SCREEN_CAPTURE_RESOLUTION)
            self.virtual_display.start()

        LeapHandRotOrig.__init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

        # reconfigure viewer camera to look from top down and to enable free camera movement mode
        if self.viewer:
            self.default_cam_pos = gymapi.Vec3(1e-4, 0.11, 0.8)
            self.default_cam_target = gymapi.Vec3(1e-3, 0.11, 0)
            self.gym.viewer_camera_look_at(self.viewer, None, self.default_cam_pos, self.default_cam_target)
            self.free_cam = True # default to the free camera mode

        # buffer for keeping track of object angular velocity across entire run
        self.object_angvel_finite_diff_entire_run = torch.zeros((self.num_envs, 3), device=self.device)

        # setup if logging state
        if self.log_state_in_test:
            self.state_logger = StateLogger()

        if self.debug_logging:
            self.joints_unscaled_list = []
            self.joints_scaled_list = []
            self.save_actions_list = []

        # TODO: construct_sim_to_real_transformation works for embodiment 001, but doesn't work for other because joints are missing so we need to determine the best way to either pad or condense where there are mising real joints
        # self.construct_sim_to_real_transformation() # this is not really needed, but has useful assertions that validate the sim2real ordering provided in the config are correct

    def allocate_buffers(self):
        # General buffers
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.at_reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.timeout_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.extras = {}

        if self.tokenize_observation_for_policy:
            self.obs_buf = self.observation_tokenizer.tokenized_obs_buf
        else:
            # Observation buffer for standard observation
            self.standard_obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=torch.float)
            self.standard_obs_buf_lag_history = torch.zeros((
                self.num_envs, 80, self.num_obs // 3
            ), device=self.device, dtype=torch.float)
            self.obs_buf = self.standard_obs_buf

    def _setup_object_info(self, o_cfg):
        self.object_type_list = [o_cfg['type']]
        self.asset_files_dict = {
            'cube': 'leap/obj/cube.urdf'
        }
        self.object_type_prob = o_cfg['sampleProb']

    def _load_assets(self):
        # object file to asset
        self.hand_asset_file = self.cfg['env']['asset']['handAsset']

        # load hand asset
        hand_asset_options = gymapi.AssetOptions()
        hand_asset_options.flip_visual_attachments = False
        hand_asset_options.fix_base_link = True
        hand_asset_options.collapse_fixed_joints = True
        hand_asset_options.disable_gravity = False
        hand_asset_options.thickness = 0.001
        hand_asset_options.angular_damping = 0.01

        # Convex decomposition
        hand_asset_options.vhacd_enabled = True
        hand_asset_options.vhacd_params.resolution = 300000
        # hand_asset_options.vhacd_params.max_convex_hulls = 30
        # hand_asset_options.vhacd_params.max_num_vertices_per_ch = 64

        hand_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        self.hand_asset = self.gym.load_asset(self.sim, self.asset_root, self.hand_asset_file, hand_asset_options)

        # verify dof ordering is the same as in the configuration file
        assert self.gym.get_asset_dof_dict(self.hand_asset) == self.joint_name_to_joint_i
        
        if "leap_hand" in self.hand_asset_file:
            rsp = self.gym.get_asset_rigid_shape_properties(self.hand_asset)   

            for i, (_, body_group) in enumerate(self.cfg["env"]["mask_body_collision"].items()):
                filter_value = 2 ** i

                for body_idx in body_group:
                    start, count = self.body_shape_indices[body_idx]
                    
                    for idx in range(count):
                        rsp[idx + start].filter = rsp[idx + start].filter | filter_value 

            if self.cfg["env"]["disable_self_collision"]: # Disable all collisions
                for i in range(len(rsp)):
                    rsp[i].filter = 1

            self.gym.set_asset_rigid_shape_properties(self.hand_asset, rsp)

        # load object asset
        self.object_asset_list = []
        for object_type in self.object_type_list:
            object_asset_file = self.asset_files_dict[object_type]
            object_asset_options = gymapi.AssetOptions()

            if self.cfg["env"]["disable_gravity"]:
                object_asset_options.disable_gravity = True

            object_asset = self.gym.load_asset(self.sim, self.asset_root, object_asset_file, object_asset_options)
            self.object_asset_list.append(object_asset)

    def _create_envs(self, num_envs, spacing, num_per_row):
        """
        Changes from original:
        - assume only 1 object model
        - create aggregate groups with exactly the right shape/body count
        """
        self._create_ground_plane()
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        self._load_assets()

        # Select the object asset
        object_type_id = np.random.choice(len(self.object_type_list), p=self.object_type_prob)
        object_asset = self.object_asset_list[object_type_id]
        assert object_type_id == 0 and len(self.object_type_list) == 1, 'we assume single object choice'

        # set leap_hand dof properties
        self.num_leap_hand_dofs = self.gym.get_asset_dof_count(self.hand_asset)
        leap_hand_dof_props = self.gym.get_asset_dof_properties(self.hand_asset)

        self.leap_hand_dof_lower_limits = []
        self.leap_hand_dof_upper_limits = []

        for i in range(self.num_leap_hand_dofs):
            self.leap_hand_dof_lower_limits.append(leap_hand_dof_props['lower'][i])
            self.leap_hand_dof_upper_limits.append(leap_hand_dof_props['upper'][i])
            leap_hand_dof_props['effort'][i] = 0.5
            leap_hand_dof_props['stiffness'][i] = self.cfg['env']['controller']['pgain']
            leap_hand_dof_props['damping'][i] = self.cfg['env']['controller']['dgain']
            leap_hand_dof_props['friction'][i] = 0.01
            leap_hand_dof_props['armature'][i] = 0.001

        self.leap_hand_dof_lower_limits = to_torch(self.leap_hand_dof_lower_limits, device=self.device)
        self.leap_hand_dof_upper_limits = to_torch(self.leap_hand_dof_upper_limits, device=self.device)

        self.leap_hand_dof_lower_limits = self.leap_hand_dof_lower_limits.repeat((self.num_envs, 1))  
        self.leap_hand_dof_lower_limits += (2 * torch.rand_like(self.leap_hand_dof_lower_limits) - 1) * self.cfg["env"]["randomization"]["joint_limits"]
        self.leap_hand_dof_upper_limits = self.leap_hand_dof_upper_limits.repeat((self.num_envs, 1))
        self.leap_hand_dof_upper_limits += (2 * torch.rand_like(self.leap_hand_dof_upper_limits) - 1) * self.cfg["env"]["randomization"]["joint_limits"]

        hand_pose, obj_pose = self._init_object_pose()
        self.hand_pose_transform = hand_pose

        # compute aggregate size
        self.num_leap_hand_bodies = self.gym.get_asset_rigid_body_count(self.hand_asset)
        self.num_leap_hand_shapes = self.gym.get_asset_rigid_shape_count(self.hand_asset)
        num_obj_bodies = self.gym.get_asset_rigid_body_count(object_asset)
        num_obj_shapes = self.gym.get_asset_rigid_shape_count(object_asset)
        max_agg_bodies = self.num_leap_hand_bodies + num_obj_bodies
        max_agg_shapes = self.num_leap_hand_shapes + num_obj_shapes

        self.envs = []

        self.object_init_state = []

        self.hand_indices = []
        self.object_indices = []

        self.object_rb_handles = list(range(self.num_leap_hand_bodies, self.num_leap_hand_bodies + num_obj_bodies))
        self.obj_scales = []
        self.object_friction_buf = torch.zeros((self.num_envs), device=self.device, dtype=torch.float)

        # There appears to be a memory allocation issue in Isaac Gym that causes the gym.create_actor function to randomly fail when creating the hand actor with one of mulitple error messages (corrrupted size, freeing invalid pointer, malloc issues) after a couple of the environments have been made. The issue seems to be dependent on the embodiment used and my hypothesis is that it depends on directly on DoF count (since that is one factor that will determine how much memory is allocated). Thus we use a hack that changes the size of the aggregrate group by a multiplier based on the DoF count (in ideal case this multiplier would just be 1), which appears to fix the issue, but is definitely hacky.
        if self.hand_asset_name in ['150', '585', '596', '605', '612', '626', '632']: # TODO: fix this hackiness
            aggregate_multiplier = 2
        elif self.dof_count == 6:
            aggregate_multiplier = 2
        elif self.dof_count == 7:
            aggregate_multiplier = 2
        elif self.dof_count == 10:
            aggregate_multiplier = 2
        elif self.dof_count == 11:
            aggregate_multiplier = 2
        elif self.dof_count == 12:
            aggregate_multiplier = 4
        elif self.dof_count == 14:
            aggregate_multiplier = 3
        elif self.dof_count == 16:
            aggregate_multiplier = 2
        elif self.dof_count >= 10:
            aggregate_multiplier = 9
        else:
            aggregate_multiplier = 4

        for i in tqdm(range(num_envs), desc="Creating envs"):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            if self.aggregate_mode >= 1:
                assert self.gym.begin_aggregate(env_ptr, max_agg_bodies * aggregate_multiplier, max_agg_shapes * aggregate_multiplier, True) # TODO: for some reason you need to multiply by some multiplier otherwise it crashes (and it seems like aggregate_mode has to be enabled otherwise it crashes). Further investigate this.

            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
            hand_actor = self.gym.create_actor(env_ptr, self.hand_asset, hand_pose, 'hand', i, -1, 0)
            self.gym.enable_actor_dof_force_sensors(env_ptr, hand_actor)
            self.gym.set_actor_dof_properties(env_ptr, hand_actor, leap_hand_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)

            # add object
            if self.cfg["env"]["disable_object_collision"]:
                collision_group = -(i+2)
            else:
                collision_group = i

            object_handle = self.gym.create_actor(env_ptr, object_asset, obj_pose, 'object', collision_group, 0, 0)
            self.object_init_state.append([
                obj_pose.p.x, obj_pose.p.y, obj_pose.p.z,
                obj_pose.r.x, obj_pose.r.y, obj_pose.r.z, obj_pose.r.w,
                0, 0, 0, 0, 0, 0
            ])
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)

            obj_scale = self.base_obj_scale
            
            if self.randomize_scale:
                num_scales = len(self.randomize_scale_list)
                obj_scale = np.random.uniform(self.randomize_scale_list[i % num_scales] - 0.025, self.randomize_scale_list[i % num_scales] + 0.025)
                
                if "randomize_scale_factor" in self.cfg["env"]:
                    obj_scale *= np.random.uniform(*self.cfg["env"]["randomize_scale_factor"])
                
                self.obj_scales.append(obj_scale)
            self.gym.set_actor_scale(env_ptr, object_handle, obj_scale)

            obj_com = [0, 0, 0]
            if self.randomize_com:
                prop = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
                assert len(prop) == 1
                obj_com = [np.random.uniform(self.randomize_com_lower, self.randomize_com_upper),
                           np.random.uniform(self.randomize_com_lower, self.randomize_com_upper),
                           np.random.uniform(self.randomize_com_lower, self.randomize_com_upper)]
                prop[0].com.x, prop[0].com.y, prop[0].com.z = obj_com
                self.gym.set_actor_rigid_body_properties(env_ptr, object_handle, prop)

            obj_friction = 1.0
            if self.randomize_friction:
                rand_friction = np.random.uniform(self.randomize_friction_lower, self.randomize_friction_upper)
                hand_props = self.gym.get_actor_rigid_shape_properties(env_ptr, hand_actor)
                for p in hand_props:
                    p.friction = rand_friction
                self.gym.set_actor_rigid_shape_properties(env_ptr, hand_actor, hand_props)

                object_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle)
                for p in object_props:
                    p.friction = rand_friction
                self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle, object_props)
                obj_friction = rand_friction
            self.object_friction_buf[i] = obj_friction

            if self.aggregate_mode > 0:
                assert self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)

        self.obj_scales = torch.tensor(self.obj_scales, device=self.device)
        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.object_rb_handles = to_torch(self.object_rb_handles, dtype=torch.long, device=self.device)
        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)

        # Add camera to the first environment
        if self.enable_camera_sensors:
            env_ptr = self.envs[0]
            camera_props = gymapi.CameraProperties()
            camera_props.horizontal_fov = 35.0
            camera_props.far_plane = 10.0
            camera_props.near_plane = 1e-2
            camera_props.height = self.camera_resolution
            camera_props.width = self.camera_resolution
            
            cam_handle = self.gym.create_camera_sensor(env_ptr, camera_props)
            self.render_camera = cam_handle
            self.render_camera_env = env_ptr
            
            transform = gymapi.Transform()
            transform.p = gymapi.Vec3(0,0,1)
            transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,1,0), np.radians(90.0))
            self.gym.set_camera_transform(cam_handle, env_ptr, transform)

    def render(self, mode="rgb_array"):
        # if requested, draw debug markers for predicted forward kinematics from the model
        if self.viewer and self.enable_fk_vis:
            self.draw_forward_kinematics_estimate()

        super().render()

        if self.virtual_display and mode == "rgb_array":
            img = self.virtual_display.grab(autocrop=False) # disabled autocrop because for some reason 3 pixels were being cropped off the height starting with the second captured frame
            img = np.array(img)
            img = crop_isaacgym_viewer_sidebar(img)
            return img
        
        if self.enable_camera_sensors and mode == "rgb_array":
            # viewer will have already stepped graphics so only do this if no viewer
            if not self.viewer:
                self.gym.fetch_results(self.sim, True)
                self.gym.step_graphics(self.sim)
            
            self.gym.render_all_camera_sensors(self.sim)
            img = self.gym.get_camera_image(self.sim, self.render_camera_env, self.render_camera, gymapi.IMAGE_COLOR)
            img = img.reshape((img.shape[0], img.shape[1] // 4, 4))[:,:,:3]
            
            return img
        
    def reset_idx(self, env_ids):
        if self.randomize_mass:
            lower, upper = self.randomize_mass_lower, self.randomize_mass_upper

            for env_id in env_ids:
                env = self.envs[env_id]
                handle = self.gym.find_actor_handle(env, 'object')
                prop = self.gym.get_actor_rigid_body_properties(env, handle)
                for p in prop:
                    p.mass = np.random.uniform(lower, upper)
                self.gym.set_actor_rigid_body_properties(env, handle, prop)
        else:
            for env_id in env_ids:
                env = self.envs[env_id]
                handle = self.gym.find_actor_handle(env, 'object')
                prop = self.gym.get_actor_rigid_body_properties(env, handle)

        if self.randomize_pd_gains:
            self.p_gain[env_ids] = torch_rand_float(
                self.randomize_p_gain_lower, self.randomize_p_gain_upper, (len(env_ids), self.num_actions),
                device=self.device).squeeze(1)
            self.d_gain[env_ids] = torch_rand_float(
                self.randomize_d_gain_lower, self.randomize_d_gain_upper, (len(env_ids), self.num_actions),
                device=self.device).squeeze(1)

        self.resample_randomizations(env_ids)

        # reset rigid body forces
        self.rb_forces[env_ids, :, :] = 0.0

        num_scales = len(self.randomize_scale_list)
        for n_s in range(num_scales):
            s_ids = env_ids[(env_ids % num_scales == n_s).nonzero(as_tuple=False).squeeze(-1)]
            if len(s_ids) == 0:
                continue
            obj_scale = self.randomize_scale_list[n_s]
            scale_key = str(obj_scale)
            
            if "sampled_pose_idx" in self.cfg["env"]:
                sampled_pose_idx = np.ones(len(s_ids), dtype=np.int32) * self.cfg["env"]["sampled_pose_idx"]
            else:
                sampled_pose_idx = np.random.randint(self.saved_grasping_states[scale_key].shape[0], size=len(s_ids))
            
            sampled_pose = self.saved_grasping_states[scale_key][sampled_pose_idx].clone()
            self.root_state_tensor[self.object_indices[s_ids], :7] = sampled_pose[:, self.dof_count:]
            self.root_state_tensor[self.object_indices[s_ids], 7:13] = 0
            pos = sampled_pose[:, :self.dof_count]
            self.leap_hand_dof_pos[s_ids, :] = pos
            self.leap_hand_dof_vel[s_ids, :] = 0
            self.prev_targets[s_ids, :self.num_leap_hand_dofs] = pos
            self.cur_targets[s_ids, :self.num_leap_hand_dofs] = pos
            self.init_pose_buf[s_ids, :] = pos.clone()
            self.object_init_pose_buf[s_ids, :] = sampled_pose[:, self.dof_count:].clone() 

        object_indices = torch.unique(self.object_indices[env_ids]).to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor), gymtorch.unwrap_tensor(object_indices), len(object_indices))
        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        self.gym.set_dof_position_target_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.prev_targets), gymtorch.unwrap_tensor(hand_indices), len(env_ids))
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        mask = self.progress_buf[env_ids] > 0
        self.object_angvel_finite_diff_ep_buf.extend(list(self.object_angvel_finite_diff_mean[env_ids][mask]))
        self.object_angvel_finite_diff_mean[env_ids] = 0

        if "print_object_angvel" in self.cfg["env"] and len(self.object_angvel_finite_diff_ep_buf) > 0:
            print("mean object angvel: ", sum(self.object_angvel_finite_diff_ep_buf) / len(self.object_angvel_finite_diff_ep_buf))

        self.progress_buf[env_ids] = 0
        # self.obs_buf[env_ids] = 0 # removed from original since this should have no effect since compute_observation should be called right after reset_idx
        self.rb_forces[env_ids] = 0
        self.at_reset_buf[env_ids] = 1

    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        # For some reason, the underlying VecTaskRot step function inherited by LeapHandRot is slightly different than than step from VecTask from IsaacGymEnvs. One difference is that VecTaskRot step always calls the render() function. This is not necessary (and dramatically slows down performance) when force_render == False and during frames in which the video recording is not occuring. Thus, this modification below only renders in cases where it's necessary by temporarily replacing the render call with a dummy function.
        if not self.force_render:
            saved_render_function = self.render
            self.render = lambda *args, **kwargs: None

        result = super().step(actions)

        if not self.force_render:
            self.render = saved_render_function
        
        return result
    
    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)

        # Log observations and actions
        is_logging = self.is_test and self.log_state_in_test
        if is_logging and self.global_counter > 1:
            # TODO: Notably we save the actions after pre_physics_step is called in the parent. The parent clips the actions between -1 and 1, so we are saving clipped actions, rather than the actions directly output by the network. This is a design decision that should be considered to figure out if it makes sense
            # only log after global_counter > 1 since first update to global counter is environment reset, not an actual action
            self.state_logger.record_actions(self.actions)
            self.state_logger.commit_state()  

    def compute_observations(self):
        self._refresh_gym()

        # Store raw observation for logging
        if self.log_state_in_test and self.is_test:
            self.state_logger.record_observation(*self.get_time_varying_raw_obs())

        if self.tokenize_observation_for_policy:
            self.compute_tokenized_observation()
        
        if not self.tokenize_observation_for_policy:
            self.compute_standard_observation()

        self.obs_buf = self.observation_tokenizer.tokenized_obs_buf if self.tokenize_observation_for_policy else self.standard_obs_buf

        if self.debug_logging:
            self.log_obs_debug()

    def get_time_varying_raw_obs(self):
        # For some reason, the dof_pos seems to be "unscaled" so that it is range -1 to 1, but the target pos does not have the same unscaling operation applied (meaning it's in the range self.leap_hand_dof_lower_limits to self.leap_hand_dof_upper_limits). It seems like the scale of these values should match, but this doesn't really impact learning.
        joint_noise_matrix = self.get_joint_noise()
        noisy_dof_pos = unscale(
            joint_noise_matrix.to(self.device) + self.leap_hand_dof_pos, self.leap_hand_dof_lower_limits, self.leap_hand_dof_upper_limits
        )
        raw_global_obs = [self.phase]
        raw_local_obs = [noisy_dof_pos, self.cur_targets]

        return raw_global_obs, raw_local_obs, self.at_reset_buf

    def compute_standard_observation(self):
        """Compute observation in same structure as the original implementation"""
        prev_obs_buf = self.standard_obs_buf_lag_history[:, 1:].clone()
        joint_noise_matrix = self.get_joint_noise()
        cur_obs_buf = unscale(
            joint_noise_matrix.to(self.device) + self.leap_hand_dof_pos, self.leap_hand_dof_lower_limits, self.leap_hand_dof_upper_limits
        ).clone().unsqueeze(1)

        self.cur_obs_buf_noisy = cur_obs_buf.squeeze(1).clone()
        self.cur_obs_buf_clean = unscale(
            self.leap_hand_dof_pos, self.leap_hand_dof_lower_limits, self.leap_hand_dof_upper_limits
        ).clone()            

        cur_tar_buf = self.cur_targets[:, None]
        
        if self.cfg["env"]["include_targets"]:
            cur_obs_buf = torch.cat([cur_obs_buf, cur_tar_buf], dim=-1)

        if self.cfg["env"]["include_obj_pose"]:
            cur_obs_buf = torch.cat([
                cur_obs_buf, 
                self.object_pos.unsqueeze(1), 
                self.object_rpy.unsqueeze(1)
            ], dim=-1)

        if self.cfg["env"]["include_obj_scales"]:
            cur_obs_buf = torch.cat([
                cur_obs_buf, 
                self.obj_scales.unsqueeze(1).unsqueeze(1), 
            ], dim=-1)
        
        if self.cfg["env"]["include_pd_gains"]:
            cur_obs_buf = torch.cat([
                cur_obs_buf, 
                self.p_gain.unsqueeze(1), 
                self.d_gain.unsqueeze(1)
            ], dim=-1)
        
        if self.cfg["env"]["include_friction_coefficient"]:
            cur_obs_buf = torch.cat([
                cur_obs_buf,
                self.object_friction_buf.unsqueeze(1).unsqueeze(1)
            ], dim=-1)

        if "phase_period" in self.cfg["env"]:
            cur_obs_buf = torch.cat([cur_obs_buf, self.phase[:, None]], dim=-1)

        if self.cfg["env"]["include_history"]:
            at_reset_env_ids = self.at_reset_buf.nonzero(as_tuple=False).squeeze(-1)
            self.standard_obs_buf_lag_history[:] = torch.cat([prev_obs_buf, cur_obs_buf], dim=1)

            # refill the initialized buffers
            self.standard_obs_buf_lag_history[at_reset_env_ids, :, 0:self.dof_count] = unscale(
                self.leap_hand_dof_pos[at_reset_env_ids], self.leap_hand_dof_lower_limits[at_reset_env_ids],
                self.leap_hand_dof_upper_limits[at_reset_env_ids]
            ).clone().unsqueeze(1)

            if self.cfg["env"]["include_targets"]:
                self.standard_obs_buf_lag_history[at_reset_env_ids, :, self.dof_count:2*self.dof_count] = self.leap_hand_dof_pos[at_reset_env_ids].unsqueeze(1)
            
            t_buf = (self.standard_obs_buf_lag_history[:, -3:].reshape(self.num_envs, -1)).clone() # attach three timesteps of history

            self.standard_obs_buf[:, :t_buf.shape[1]] = t_buf

            # self.proprio_hist_buf[:] = self.obs_buf_lag_history[:, -self.prop_hist_len:].clone()
            self.at_reset_buf[at_reset_env_ids] = 0
        else:
            self.standard_obs_buf = cur_obs_buf.clone().squeeze(1)

        if self.cfg["env"]["obs_mask"] is not None:
            self.standard_obs_buf = self.standard_obs_buf * torch.tensor(self.cfg["env"]["obs_mask"], device=self.device)[None, :]

    def compute_tokenized_observation(self):
        # We currently have fixed assumptions about the observation structure (matches the default parameters in LEAP repo)
        assert self.cfg["env"]["include_targets"], 'currently assumed to hold'
        assert not self.cfg["env"]["include_obj_pose"], 'not yet implemented'
        assert not self.cfg["env"]["include_obj_scales"], 'not yet implemented'
        assert not self.cfg["env"]["include_pd_gains"], 'not yet implemented'
        assert not self.cfg["env"]["include_friction_coefficient"], 'not yet implemented'
        assert "phase_period" in self.cfg["env"], 'currently assumed to hold'
        assert self.cfg["env"]["include_history"], 'currently assumed to hold'

        # Note: In the original LEAP implementation, on reset, the history buffer is filled with accurate values for the dof pos, but noisy values are used when creating new observations. Not sure why this is the case. My current implementation doesn't exactly match this as it uses the noisy values to reset the old history entries. Turns out this doesn't matter since the joint noise matrix is by default not used (set to 0).
        self.observation_tokenizer.build_tokenized_observation(*self.get_time_varying_raw_obs())
        # TODO: it would be better to not have the joint state normalized between -1 and 1 as it currently is because it's weird to have it normalized, but have the targets not be normalized in the same fashion

        # Reset the reset buffer for the environments that were just reset
        at_reset_env_ids = self.at_reset_buf.nonzero(as_tuple=False).squeeze(-1)
        self.at_reset_buf[at_reset_env_ids] = 0

        if self.cfg["env"]["obs_mask"] is not None:
            self.tokenized_obs_buf = self.tokenized_obs_buf * torch.tensor(self.cfg["env"]["obs_mask"], device=self.device)[None, :]

    def log_obs_debug(self):
        # Log debugging data for deployment to real LEAP hand
        self.joints_unscaled_list.append(self.cur_obs_buf_clean.clone())
        self.target_list.append(self.cur_targets[0].clone().squeeze())
        self.joints_scaled_list.append(self.leap_hand_dof_pos.clone())
        self.obs_list.append(self.obs_buf.clone())
        self.save_actions_list.append(self.actions.clone())

        if self.global_counter == self.record_duration - 1:
            self.joints_unscaled_list = torch.stack(self.joints_unscaled_list, dim=0).cpu().numpy()
            self.target_list = torch.stack(self.target_list, dim=0).cpu().numpy()
            self.joints_scaled_list = torch.stack(self.joints_scaled_list, dim=0).cpu().numpy()
            self.obs_list = torch.stack(self.obs_list, dim=0).cpu().numpy().squeeze(1)
            self.save_actions_list = torch.stack(self.save_actions_list, dim=0).cpu().numpy().squeeze(1)

            if "actions_file" in self.cfg["env"]["debug"]:
                actions_file = os.path.basename(self.cfg["env"]["debug"]["actions_file"])
                suffix = "_".join(actions_file.split("_")[1:]) 
                folder = os.path.dirname(self.cfg["env"]["debug"]["actions_file"])
                folder = os.path.join(folder, suffix)
            else:
                suffix = self.cfg["env"]["debug"]["record"]["suffix"]
                folder = os.path.join('debug', 'leap', 'sim', suffix)
                if os.path.exists(folder):
                    shutil.rmtree(folder)
                os.makedirs(folder)

            np.savez(os.path.join(folder, 'data.npz'),
                        joints_unscaled=self.joints_unscaled_list,
                        target_list=self.target_list,
                        joints_scaled=self.joints_scaled_list,
                        obs_list=self.obs_list,
                        leap_dof_lower=self.leap_hand_dof_lower_limits.cpu().numpy().squeeze(0),
                        leap_dof_upper=self.leap_hand_dof_upper_limits.cpu().numpy().squeeze(0),
                        sim_to_real_indices=np.array(self.cfg["env"]["sim_to_real_indices"]),
                        real_to_sim_indices=np.array(self.cfg["env"]["real_to_sim_indices"]),
                        actions_list=self.save_actions_list
            )
            exit()

    def construct_sim_to_real_transformation(self):
        self.sim_dof_order = self.gym.get_actor_dof_names(self.envs[0], 0)
        self.sim_dof_order = [int(x) for x in self.sim_dof_order]
        self.real_dof_order = list(range(self.dof_count))
        self.sim_to_real_indices = [] # Value at i is the location of ith real index in the sim list

        for x in self.real_dof_order:
            self.sim_to_real_indices.append(self.sim_dof_order.index(x))
        
        self.real_to_sim_indices = []

        for x in self.sim_dof_order:
            self.real_to_sim_indices.append(self.real_dof_order.index(x))
        
        assert(self.sim_to_real_indices == self.cfg["env"]["sim_to_real_indices"])
        assert(self.real_to_sim_indices == self.cfg["env"]["real_to_sim_indices"])

    def start(self, is_train, run_dir):
        self.is_test = not is_train
        self.run_dir = run_dir
        self.object_angvel_finite_diff_entire_run.zero_()
        self.global_counter = 0
        self.absolute_counter -= 1 # each start corresonds to a reset, which counts as an additional step, so remove one here

        # reset buffers
        self.reset_buf.fill_(1) # this needs to be set to nonzero value such that when self.reset() is called by the Runner (soon after this function) all enviroments will be reset
        self.timeout_buf.zero_()
        self.progress_buf.zero_()

        # in test mode always disable random force pertubations
        if self.is_test:
            self.force_scale = 0

        # optionally disable (almost all) randomizations when in test mode
        if self.is_test and self.disable_randomizations_during_test:
            self.max_episode_length = float('inf')

            # set fixed object mass
            self.randomize_mass = False
            for env_id in range(self.num_envs):
                env = self.envs[env_id]
                handle = self.gym.find_actor_handle(env, 'object')
                prop = self.gym.get_actor_rigid_body_properties(env, handle)
                for p in prop:
                    p.mass = (self.randomize_mass_lower + self.randomize_mass_upper) / 2
                self.gym.set_actor_rigid_body_properties(env, handle, prop)

            # It seems like PD gains are randomized, but these values are not actually used

            # reset COM
            for env_id in range(self.num_envs):
                env_ptr = self.envs[env_id]
                object_handle = self.gym.find_actor_handle(env, 'object')
                prop = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
                assert len(prop) == 1
                obj_com = [0,0,0]
                prop[0].com.x, prop[0].com.y, prop[0].com.z = obj_com
                self.gym.set_actor_rigid_body_properties(env_ptr, object_handle, prop)

            # reset friction
            obj_friction = 1.0
            for env_id in range(self.num_envs):
                env_ptr = self.envs[env_id]
                object_handle = self.gym.find_actor_handle(env, 'object')
                hand_actor = self.gym.find_actor_handle(env, 'hand')
                rand_friction = (self.randomize_friction_lower + self.randomize_friction_upper) / 2
                hand_props = self.gym.get_actor_rigid_shape_properties(env_ptr, hand_actor)
                for p in hand_props:
                    p.friction = rand_friction
                self.gym.set_actor_rigid_shape_properties(env_ptr, hand_actor, hand_props)

                object_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle)
                for p in object_props:
                    p.friction = rand_friction
                self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle, object_props)
                obj_friction = rand_friction
                self.object_friction_buf[env_id] = obj_friction

            # reset scale
            self.obj_scales = []
            for env_id in range(self.num_envs):
                env_ptr = self.envs[env_id]
                object_handle = self.gym.find_actor_handle(env, 'object')
                obj_scale = 1
                self.obj_scales.append(obj_scale)
                self.gym.set_actor_scale(env_ptr, object_handle, obj_scale)
            self.obj_scales = torch.tensor(self.obj_scales, device=self.device)

            # joint noise
            assert "joint_noise" not in self.cfg["env"]["randomization"]
    
    def finish(self):
        if self.is_test and self.log_state_in_test:
            log_dir = os.path.join('state_logs', 'LeapHandRot', self.hand_asset_name)
            metadata = {
                'experiment_dir': self.run_dir # perhaps this should be changed to instead be the run ID rather than the path to filesystem path to the run (currently it's in the format f'runs/{run_id}')
            }
            self.embodiment_properties.metadata = metadata
            self.state_logger.save_state_logs(log_dir, os.path.basename(self.run_dir), self.log_state_suffix, self.embodiment_properties)        

    def compute_reward(self, actions):
        super().compute_reward(actions)

        # right after reset, just ignore the yaw_finite_diff value because it's inaccurate due to the reset
        self.object_angvel_finite_diff[torch.where(self.progress_buf < 3)] = 0
        self.extras['yaw_finite_diff'] = self.object_angvel_finite_diff[:, 2].mean()
        self.object_angvel_finite_diff_entire_run += self.object_angvel_finite_diff * self.control_dt # computed as self.object_angvel_finite_diff /= (self.control_dt * delta_counter), but delta_counter appears to always be 1, so safe to not have it included here
        self.extras['yaw_finite_diff_cumulative'] = self.object_angvel_finite_diff_entire_run[:, 2].mean()
        
        self.extras['global_counter'] = self.global_counter - 1 # reset increases by 1 global step, so subtract 1 here
        self.absolute_counter += 1
        self.extras['absolute_counter'] = self.absolute_counter
    
    def update_forward_kinematics_estimate(self, fk_estimate):
        """If the control policy returns estimates of the forward kinematics, this function will be called externally (by CustomPpoPlayerContinuous; thus meaning this only works at test time) to update the estimate for visualization purposes"""
        self.fk_estimate = fk_estimate

    def draw_forward_kinematics_estimate(self):
        """
        Draws green spheres on viewer where the joint positions are in 3D space according to FK.
        Draws red spheres on the viewer where the model controlling the hand predicts that the joints are (if this is suppplied by calling `update_forward_kinematics_estimate`).
        """
        self.gym.clear_lines(self.viewer)

        poses_to_draw = []

        # compute FK on the current joint state
        cur_joint_angles = self.leap_hand_dof_state[:, :, 0] # index 0 is joint angles; ordered based on sim ordering; (num_envs, dof_count)
        joint_pos_gt = fk(self.urdf, cur_joint_angles, self.joint_name_to_joint_i) # (num_envs, dof_count, 3)
        poses_to_draw.append((joint_pos_gt, (0, 1, 0)))
        
        # vis the FK estimate if it is set
        if self.fk_estimate is not None:
            poses_to_draw.append((self.fk_estimate, (1, 0, 0)))
        
        for i in range(self.num_envs):
            env = self.envs[i]

            # offset consists of both offset of environment
            offset = self.gym.get_env_origin(env)

            for j in range(self.dof_count):
                # draw the poses
                for joint_pos, color in poses_to_draw:
                    cur_pose = self.hand_pose_transform.transform_point(gymapi.Vec3(*list(joint_pos[i][j]))) # apply the transform from env frame to hand frame
                    glob_pos = offset + cur_pose

                    # draw poses over the hand and also shifted to the side to make it easier to see
                    for shift in [0, 0.15]:
                        shift = gymapi.Vec3(0, shift,0)
                        target_pose = gymapi.Transform(
                            p=glob_pos + shift,
                            r=gymapi.Quat(1,0,0,0),
                        )
                        axis_geom = gymutil.WireframeSphereGeometry(radius=0.01, color=color)
                        gymutil.draw_lines(axis_geom, self.gym, self.viewer, env, target_pose)
