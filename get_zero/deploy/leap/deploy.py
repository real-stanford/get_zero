# --------------------------------------------------------
# LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning
# https://arxiv.org/abs/2309.06440
# Copyright (c) 2023 Ananye Agarwal
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Based on:
# https://github.com/HaozhiQi/hora/blob/main/hora/algo/deploy/deploy.py
# --------------------------------------------------------

import torch
import os
import hydra
from rl_games.torch_runner import Runner, _override_sigma, _restore
from omegaconf import DictConfig, OmegaConf, open_dict
import numpy as np
from gym import spaces
import math
import random
import shutil
from get_zero.deploy.leap.util.multi_embodiment_py_leap_client import MultiEmbodimentLeapHand
from get_zero.deploy.leap.util.multi_embodiment_leap_hand_utils import MultiEmbodimentLeapHandUtils
from util.rate import Rate
import time
from rl_games.algos_torch.model_builder import register_network
from rl_games.common.algo_observer import AlgoObserver
from get_zero.rl.models.embodiment_transformer import EmbodimentTransformerBuilder
from get_zero.rl.utils.rl_util import CustomPpoPlayerContinuous
from get_zero.distill.utils.embodiment_util import ObservationTokenizer
from get_zero.deploy.leap.util.multi_embodiment_leap_asset_utils import get_leap_embodiment_properties
from get_zero.deploy.leap.util.camera_util import MeasureRotation, RealsenseCamera
from get_zero.deploy.leap.util.ar_tag import ARUCO_DICT, CUBE_ID
from datetime import datetime
import yaml
import wandb
from multiprocessing import Process

from get_zero.rl.utils.generic_util import register_all_omegaconf_resolvers
register_all_omegaconf_resolvers()

def record_video(realsense_video_cam_id, video_path: str, duration):
    video_cam = RealsenseCamera(realsense_video_cam_id)
    video_cam.record_video(duration, video_path)

class HardwarePlayer:
    def __init__(self, cfg):
        # load config from checkpoint if it exists
        if cfg.checkpoint:
            checkpoint_data = torch.load(cfg.checkpoint, map_location='cpu')
            if 'tokenization_config' not in checkpoint_data or 'model_config' not in checkpoint_data:
                print('WARNING: provided checkpoint does not have `tokenization_config` or `model_config` present, so we are assuming that the config used to create the checkpoint matches the config currently specified')
            else:
                print(f'Using tokenization config and model config from the checkpoint at {cfg.checkpoint}')
                cfg.task.env.tokenization = OmegaConf.create(checkpoint_data['tokenization_config'])
                OmegaConf.set_struct(cfg, True)
                with open_dict(cfg):
                    cfg.train.params.network['actor'] = checkpoint_data['model_config']
            del checkpoint_data

        self.config = OmegaConf.to_container(cfg, resolve=True)

        if "include_history" not in self.config["task"]["env"]:
            self.config["task"]["env"]["include_history"] = True

        if "include_targets" not in self.config["task"]["env"]:
            self.config["task"]["env"]["include_targets"] = True
        
        self.action_scale = 1 / 24

        self.device = f"cuda:{self.config['gpu']}" if self.config['gpu'] != -1 else self.config['rl_device']
        self.sim_to_real_indices = self.config["task"]["env"]["sim_to_real_indices"]
        self.real_to_sim_indices = self.config["task"]["env"]["real_to_sim_indices"]

        # Embodiment properties
        joint_name_to_joint_i = self.config['task']['env']['joint_name_to_joint_i']
        hand_asset_file = self.config['task']['env']['asset']['handAsset']
        embodiment_name = os.path.basename(hand_asset_file).replace('.urdf', '')
        self.embodiment_properties = get_leap_embodiment_properties(embodiment_name, joint_name_to_joint_i)
        self.lhu = MultiEmbodimentLeapHandUtils(self.embodiment_properties)

        # tokenization config
        self.tokenize_observation_for_policy = self.config['task']['env']['tokenizeObservationForPolicy']
        if self.tokenize_observation_for_policy:
            self.observation_tokenizer = ObservationTokenizer(self.config["task"]["env"]["tokenization"], self.embodiment_properties, self.device, num_envs=1)
            self.config["task"]["env"]["numObservations"] = self.observation_tokenizer.tokenized_obs_size        

        # setup the cameras for tracking AR tag
        self.just_play = self.config['deploy']['just_play']
        self.ar_cam_id = self.config['deploy']['ar_tag_realsense_id'] if not self.just_play else -1
        self.video_cam_id = self.config['deploy']['video_realsense_id'] if not self.just_play else -1
        self.measure_rotation = MeasureRotation(RealsenseCamera(self.ar_cam_id), CUBE_ID, ARUCO_DICT, reset_threshold=10) if self.ar_cam_id != -1 else None
        self.record_video = self.video_cam_id != -1 and self.config['record_video']

        # generate experiment dir
        experiment = f"_{self.config['experiment']}" if self.config['experiment'] else ''
        self.experiment_name = f"LeapHandRot_deploy{experiment}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')}"
        self.experiment_dir = os.path.join('runs', self.experiment_name)
        os.makedirs(self.experiment_dir)
        print(f'--- LOGGING DEPLOY RUN TO {self.experiment_dir} ---')

        # write config to run dir
        config_out_path = os.path.join(self.experiment_dir, 'config.yaml')
        with open(config_out_path, 'w') as f:
            f.write(OmegaConf.to_yaml(cfg))
        print(f'Wrote config to {config_out_path}')

        # WandB logging
        if self.config['wandb_activate']:
            wandb.init(
                project=self.config['wandb_project'],
                entity=self.config['wandb_entity'],
                group=self.config['wandb_group'],
                tags=self.config['wandb_tags'],
                id=f'uid_{self.experiment_name}',
                name=self.experiment_name,
                config=self.config
            )

    def real_to_sim(self, values):
        return values[:, self.real_to_sim_indices]

    def sim_to_real(self, values):
        return values[:, self.sim_to_real_indices]        

    def fetch_grasp_state(self, s=1.0):
        self.grasp_cache_name = self.config['task']['env']['grasp_cache_name']
        grasping_states = np.load(f'../../rl/cache/leap_hand/{self.embodiment_properties.name}/{self.grasp_cache_name}_grasp_50k_s{str(s).replace(".", "")}.npy')

        if "sampled_pose_idx" in self.config["task"]["env"]:
            idx = self.config["task"]["env"]["sampled_pose_idx"]
        else:
            idx = random.randint(0, grasping_states.shape[0] - 1)

        return grasping_states[idx][:self.embodiment_properties.dof_count] # first dof_count are hand dofs, last 7 is object state
    
    def compute_tokenized_observation(self, dof_pos_unscaled, phase, cur_targets):
        # We currently have fixed assumptions about the observation structure (matches the default parameters in LEAP repo)
        assert self.config["task"]["env"]["include_targets"], 'currently assumed to hold'
        assert "phase_period" in self.config["task"]["env"], 'currently assumed to hold'
        assert self.config["task"]["env"]["include_history"], 'currently assumed to hold'

        # For some reason, the dof_pos seems to be "unscaled" so that it is range -1 to 1, but the target pos does not have the same unscaling operation applied (meaning it's in the range self.leap_hand_dof_lower_limits to self.leap_hand_dof_upper_limits). It seems like the scale of these values should match, but this doesn't really impact learning.
        raw_global_obs = [phase]
        raw_local_obs = [dof_pos_unscaled, cur_targets]

        at_reset_buf = torch.zeros((1,), device=self.device)

        time_varying_raw_obs = (raw_global_obs, raw_local_obs, at_reset_buf)

        self.observation_tokenizer.build_tokenized_observation(*time_varying_raw_obs)

    def deploy(self):        
        # Set up Python interface
        num_obs = 102 
        num_obs_single = num_obs // 3 
        leap = MultiEmbodimentLeapHand(self.embodiment_properties)

        # Setup rate
        hz = 20
        self.control_dt = 1 / hz
        rate = Rate(hz)
        play_steps = self.config['test_steps']

        # Debugging parameters
        actions_from_network = True
        is_recording = False
        single_joint = False
        using_sim_obs = False
        using_sim_targets = False
        if "debug" in self.config["task"]["env"]:
            if "fixed_action_sequence" in self.config["task"]["env"]["debug"]:
                self.fixed_action_sequence = True
                actions_from_network = False
            elif "fixed_target_sequence" in self.config["task"]["env"]["debug"]:
                self.fixed_target_sequence = True
                actions_from_network = False
            elif "single_joint" in self.config["task"]["env"]["debug"]:
                single_joint = True
                single_joint_idx_real = self.config["task"]["env"]["debug"]["single_joint"]["idx"]
                single_joint_idx_sim = self.real_to_sim_indices[single_joint_idx_real]
                single_joint_pos_real = self.config["task"]["env"]["debug"]["single_joint"]["pos"]
                single_joint_pos_sim = single_joint_pos_real - np.pi

                print(f'DEBUG: single_joint mode for joint with real idx {single_joint_idx_real} (sim idx = {single_joint_idx_sim}) to real pos {single_joint_pos_real} (sim pos = {single_joint_pos_sim})')
            elif "use_sim" in self.config["task"]["env"]["debug"]:
                sim_data_path = self.config["task"]["env"]["debug"]["use_sim"]["path"]
                sim_data = np.load(sim_data_path)
                sim_obs = torch.tensor(sim_data['obs_list'], device=self.device)
                sim_targets = torch.tensor(sim_data['target_list'], device=self.device)
                sim_joints_scaled = torch.tensor(sim_data['joints_scaled'], device=self.device)
                use_from_sim = self.config["task"]["env"]["debug"]["use_sim"]["attributes"]
                for attribute in use_from_sim:
                    assert attribute in ['obs', 'targets'], 'only these attributes are currently supported to be used from sim data'
                using_sim_obs = 'obs' in use_from_sim
                using_sim_targets = 'targets' in use_from_sim
                actions_from_network = not using_sim_targets

                play_steps = sim_targets.shape[0]

            if "record" in self.config["task"]["env"]["debug"]:
                is_recording = True
                self.joints_unscaled_list = []
                self.target_list = []
                self.joints_scaled_list = []
                self.obs_list = []
                self.network_targets_list = []
                self.network_actions_list = []

                if "use_sim" in self.config["task"]["env"]["debug"]:
                    self.record_duration = play_steps
                else:
                    self.record_duration = int(self.config["task"]["env"]["debug"]["record"]["duration"] / self.control_dt)

        def goto_initial_hand_position():
            self.init_pose = self.fetch_grasp_state()
            # Move to initial position
            print("command to the initial position")
            for _ in range(hz * 4):
                leap.command_joint_position(self.init_pose)
                joint_obs_sim_ordering_limit_range, _ = leap.poll_joint_position()
                rate.sleep()
            print("reached initial position")

        if not single_joint:
            goto_initial_hand_position()

            input('press enter to confirm you have reset the cube properly:')

            # record initial position
            if self.measure_rotation:
                print("waiting for inital AR tag detection")
                self.measure_rotation.step(block=True)
                print("completed first AR tag detection")

        # start video recording in separate process since the control loop will be running at 20hz and we want to record video at different frequency
        if self.record_video:
            video_out_path = os.path.join(self.experiment_dir, 'video.mp4')
            record_time = min((play_steps / hz) + 3, 30) # record the entire trajectory with a 3 second buffer, but do at most 30 seconds of recording
            video_process = Process(target=record_video, args=(self.video_cam_id, video_out_path, record_time))
            video_process.start()
            print(f'Starting video recording to {video_out_path}')
            time.sleep(2)

        # Construct observation buffer for initial observation
        if using_sim_obs:
            obs_buf = sim_obs[0].unsqueeze(0)
            prev_target_sim_ordering_limit_range = sim_joints_scaled[0].clone()
        else:
            joint_obs_sim_ordering_limit_range = torch.tensor(leap.poll_joint_position()[0], dtype=torch.float32, device=self.device)
            prev_target_sim_ordering_limit_range = joint_obs_sim_ordering_limit_range[None].clone() # since this happens before unscaling, it means that prev_target is in the limit range, while the cur_joint_states will be in sim ones range. This is desired according to the implementation in `leap_hand_rot.py`
            cur_joint_states_sim_ordering_sim_range = self.lhu.limit_range_sim_ordering_to_sim_range_sim_ordering(joint_obs_sim_ordering_limit_range)[None]
            phase = torch.tensor([[0., 0.]], device=self.device)

            if self.tokenize_observation_for_policy:
                self.compute_tokenized_observation(cur_joint_states_sim_ordering_sim_range, phase, prev_target_sim_ordering_limit_range.clone())
                obs_buf = self.observation_tokenizer.tokenized_obs_buf
            else:
                obs_buf = torch.zeros((1,0), dtype=torch.float32, device=self.device)
                if self.config["task"]["env"]["include_history"]:
                    num_append_iters = 3
                else:
                    num_append_iters = 1

                for i in range(num_append_iters):
                    obs_buf = torch.cat([obs_buf, cur_joint_states_sim_ordering_sim_range.clone()], dim=-1)
                    
                    if self.config["task"]["env"]["include_targets"]:
                        obs_buf = torch.cat([obs_buf, prev_target_sim_ordering_limit_range.clone()], dim=-1)

                    if "phase_period" in self.config["task"]["env"]:
                        # The original deploy.py in LEAP_Hand_Sim set the phase for the intial observation. In the simulation the very first observation actually doesn't set the phase when first initializing.  Overall this shouldn't really matter though since this will only be relevant for the first two timesteps of the entire run.
                        obs_buf = torch.cat([obs_buf, phase], dim=-1)

                if "obs_mask" in self.config["task"]["env"]:
                    obs_buf = obs_buf * torch.tensor(self.config["task"]["env"]["obs_mask"], device=self.device)[None, :]

                obs_buf = obs_buf.float()

        using_fixed_action_sequence = hasattr(self, "fixed_action_sequence") and self.fixed_action_sequence
        using_fixed_target_sequence = hasattr(self, "fixed_target_sequence") and self.fixed_target_sequence
        if using_fixed_action_sequence:
            # go to starting position
            pos = np.zeros(16) + target_pos
            pos[[0,4,8]] = 0
            pos[13] = np.pi / 2
            pos = leap.real_to_sim(pos)
            leap.command_joint_position(pos)

            # this is hand closing motion
            fixed_action_motion = torch.ones(self.actions_num, device=self.device) * 0.1
            fixed_action_motion[[0,4,8]] = 0
            fixed_action_motion[13] = torch.pi / 2
            fixed_action_motion = leap.real_to_sim(fixed_action_motion)
        if using_fixed_target_sequence:
            target_pos = 0
            target_dir = 1

        # Setup if RNN (this is true in the case of the original policy architecture)
        if self.player.is_rnn:
            self.player.init_rnn()

        def reset_environment():
            nonlocal joint_obs_sim_ordering_limit_range, prev_target_sim_ordering_limit_range, target_sim_ordering_limit_range

            print('Beginning environment reset')
            
            goto_initial_hand_position()
            # also need to reset to the target to the current pose otherwise the hand will immediately "snap back" to the pose before we stopped being able to detect the cube
            joint_obs_sim_ordering_limit_range = torch.tensor(leap.poll_joint_position()[0], dtype=torch.float32, device=self.device)
            prev_target_sim_ordering_limit_range = joint_obs_sim_ordering_limit_range[None].clone()
            target_sim_ordering_limit_range = prev_target_sim_ordering_limit_range

            # reset the AR tag detection
            if self.measure_rotation:
                self.measure_rotation.reset_env()

            input('Press enter to confirm that you have reset the cube:')

        print('Starting execution loop')
        counter = 0
        last_time = time.time()
        actual_hz_running_average = hz
        start_time = time.time()
        episode_length = self.config['task']['env']['episodeLength']
        while True:
            counter += 1

            if counter > play_steps:
                break

            if (counter - 1) % episode_length == 0 and counter != 1 and not self.just_play:
                # need to perform environment reset at `episode_length` interval
                print(f'{episode_length} steps have passed ({episode_length/hz:.1f} seconds), thus performing an environment reset.')
                reset_environment()

            # Query the network
            if actions_from_network or is_recording:
                network_action = self.forward_network(obs_buf[0])

            # Control commands can come from various places (network or debug target/action trajectories) and can be either targets directly or actions that apply to the previous target
            if using_fixed_target_sequence or using_sim_targets or single_joint: # we are supplying targets directly
                # DEBUG: Fixed open/close hand target trajectory
                if using_fixed_target_sequence:
                    target_pos += self.control_dt * target_dir # move at rate 1 radian/s
                    if target_pos > 1:
                        target_dir = -1
                        target_pos = 1
                    elif target_pos < 0:
                        target_dir = 1
                        target_pos = 0
                        
                    pos = torch.zeros(16, device=self.device) + target_pos
                    pos[[0,4,8]] = 0
                    pos[13] = np.pi / 2
                    pos = leap.real_to_sim(pos)
                    target_sim_ordering_limit_range = pos[None]
                # DEBUG: replay sequence of targets
                elif using_sim_targets:
                    target_sim_ordering_limit_range = sim_targets[counter-1][None, :]
                # DEBUG: single joint
                elif single_joint:
                    pos = torch.zeros(16, device=self.device)
                    pos[single_joint_idx_real] = single_joint_pos_sim
                    pos = leap.real_to_sim(pos) # only changes joint ordering
                    if counter == 1:
                        print(f'target: {pos}')
                    target_sim_ordering_limit_range = pos[None]
            else: # supply actions rather than targets
                # DEBUG: replay sequence of actions
                if hasattr(self, "actions_list"):
                    action = self.actions_list[counter-1][None, :]
                # DEBUG: fixed action sequence (open and close hand)
                elif using_fixed_action_sequence:
                    if counter % hz == 0:
                        fixed_action_motion *= -1
                    action = fixed_action_motion
                # Use actions from policy network
                else:
                    assert actions_from_network
                    action = network_action

                action = torch.clamp(action, -1.0, 1.0)

                if "actions_mask" in self.config["task"]["env"]:
                    action = action * torch.tensor(self.config["task"]["env"]["actions_mask"], device=self.device)[None, :]

                target_sim_ordering_limit_range = prev_target_sim_ordering_limit_range + self.action_scale * action
            
            target_sim_ordering_limit_range = self.lhu.clip_limits_range_sim_ordering(target_sim_ordering_limit_range)
            if actions_from_network or is_recording:
                network_target = self.lhu.clip_limits_range_sim_ordering(prev_target_sim_ordering_limit_range + self.action_scale * network_action)

            prev_target_sim_ordering_limit_range = target_sim_ordering_limit_range.clone()
        
            # Execute command on hardware
            commands = target_sim_ordering_limit_range.cpu().numpy()[0] # TODO: get rid of this extra axis

            if "disable_actions" not in self.config["task"]["env"]:
                leap.command_joint_position(commands)
            
            # check to make sure we are running at the correct rate
            cur_time = time.time()
            actual_hz = 1/(cur_time - last_time)
            actual_hz_running_average = actual_hz_running_average * 0.9 + actual_hz * 0.1
            last_time = cur_time

            if abs(actual_hz_running_average - hz) > 3:
                print(f'Warning: runtime is averaging at {actual_hz_running_average} Hz instead of {hz} Hz desired')

            # print(f'going go sleep for {ros_rate.remaining().nsecs / 1e6} ms')
            rate.sleep()  # keep 20 Hz command
            
            # Compute new joint states
            joint_obs_sim_ordering_limit_range = torch.tensor(leap.poll_joint_position()[0], dtype=torch.float32, device=self.device)
            cur_joint_states_sim_ordering_sim_range = self.lhu.limit_range_sim_ordering_to_sim_range_sim_ordering(joint_obs_sim_ordering_limit_range)[None]

            if using_sim_obs:
                obs_buf = sim_obs[counter - 1].unsqueeze(0)
            else:
                omega = 2 * math.pi / self.config["task"]["env"]["phase_period"]
                phase_angle = (counter - 1) * omega / hz 
                num_envs = obs_buf.shape[0]
                phase = torch.zeros((num_envs, 2), device=obs_buf.device)
                phase[:, 0] = math.sin(phase_angle)
                phase[:, 1] = math.cos(phase_angle)

                if self.tokenize_observation_for_policy:
                    self.compute_tokenized_observation(cur_joint_states_sim_ordering_sim_range, phase, target_sim_ordering_limit_range.clone())
                else:
                    # Add latest observation to history buffer
                    if self.config["task"]["env"]["include_history"]:
                        obs_buf = obs_buf[:, num_obs_single:].clone()
                    else:
                        obs_buf = torch.zeros((1, 0), device=self.device)

                    obs_buf = torch.cat([obs_buf, cur_joint_states_sim_ordering_sim_range.clone()], dim=-1)

                    if self.config["task"]["env"]["include_targets"]:
                        obs_buf = torch.cat([obs_buf, target_sim_ordering_limit_range.clone()], dim=-1)

                    if "phase_period" in self.config["task"]["env"]:
                        obs_buf = torch.cat([obs_buf, phase.clone()], dim=-1)

                    if "obs_mask" in self.config["task"]["env"]:
                        obs_buf = obs_buf * torch.tensor(self.config["task"]["env"]["obs_mask"], device=self.device)[None, :]

                    obs_buf = obs_buf.float()

            # Record observation and joint states into log
            if is_recording:
                self.joints_unscaled_list.append(cur_joint_states_sim_ordering_sim_range[0].clone())
                self.target_list.append(target_sim_ordering_limit_range[0].clone().squeeze())
                self.joints_scaled_list.append(joint_obs_sim_ordering_limit_range.clone())
                self.obs_list.append(obs_buf.clone())
                self.network_targets_list.append(network_target[0].clone())
                self.network_actions_list.append(network_action.clone())

                if counter == self.record_duration:
                    self.joints_unscaled_list = torch.stack(self.joints_unscaled_list, dim=0).cpu().numpy()
                    self.target_list = torch.stack(self.target_list, dim=0).cpu().numpy()
                    self.joints_scaled_list = torch.stack(self.joints_scaled_list, dim=0).cpu().numpy()
                    self.obs_list = torch.stack(self.obs_list, dim=0).cpu().numpy().squeeze(1)
                    self.network_targets_list = torch.stack(self.network_targets_list, dim=0).cpu().numpy()
                    self.network_actions_list = torch.stack(self.network_actions_list, dim=0).cpu().numpy()

                    suffix = self.config["task"]["env"]["debug"]["record"]["suffix"]
                    folder = os.path.join('debug', 'leap', 'real', suffix)
                    if os.path.exists(folder):
                        shutil.rmtree(folder)
                    os.makedirs(folder)
                    
                    out_path = os.path.join(folder, 'data.npz')
                    np.savez(out_path,
                             joints_unscaled=self.joints_unscaled_list,
                             target_list=self.target_list,
                             joints_scaled=self.joints_scaled_list,
                             obs_list=self.obs_list,
                             network_targets_list=self.network_targets_list,
                             network_actions_list=self.network_actions_list,
                             leap_dof_lower=leap.leap_dof_lower,
                             leap_dof_upper=leap.leap_dof_upper,
                             sim_to_real_indices=np.array(self.sim_to_real_indices),
                             real_to_sim_indices=np.array(self.real_to_sim_indices)
                    )
                    print(f'Wrote to {out_path}')
                    exit()

            # record cube position
            if self.measure_rotation:
                cur_rotation, valid_steps, steps_since_last_detection, num_resets, in_reset_mode = self.measure_rotation.step()
                if in_reset_mode: # if the cube has fallen, then pause execution and reset hand to grasp cache pose
                    print(f'Cube AR tag not detected. Resetting to state from grasp cache and pausing execution until the cube is found')
                    reset_environment()
                    
                    print('waiting until AR tag is detected')
                    cur_rotation, valid_steps, steps_since_last_detection, num_resets, in_reset_mode = self.measure_rotation.step(block=True)
                    print('AR tag is detected, resuming')
                cur_rotation_deg = cur_rotation * 180 / np.pi
                completed_cycles, remainder_deg = divmod(cur_rotation_deg, 360)
                print(f'Step {counter}/{play_steps}: Completed {int(completed_cycles)} cycles with additional {int(remainder_deg)} degrees including {num_resets} resets')

                if self.config['wandb_activate']:
                    wandb.log({
                        'cur_rotation_rad': cur_rotation,
                        'cur_rotation_deg': cur_rotation_deg,
                        'steps_since_last_detection': steps_since_last_detection,
                        'num_resets': num_resets,
                        'step': counter
                    })
            else:
                print(f'Step {counter}/{play_steps}')
        
        print(f'Deployment finished! Completed {play_steps} steps corresponding to {play_steps / hz / 60} minutes of behavior in a total of {(time.time() - start_time)/60:.1f} minutes real time')
        
        # log metrics to run directory
        if self.measure_rotation:
            total_rotation_rad, valid_steps, num_resets = self.measure_rotation.get_final_metrics()
            assert valid_steps - 1 == play_steps # -1 to account for initial blocking call at very start to wait for AR tag to be present before acting

            play_time = play_steps / hz
            average_rotational_velocity_rad_per_second = total_rotation_rad / play_time
            results = {
                'final/num_resets': num_resets,
                'final/play_steps': play_steps,
                'final/play_time_sec': play_time,
                'final/total_rotation_rad': total_rotation_rad,
                'final/total_rotation_deg': total_rotation_rad * 180 / np.pi,
                'final/average_rotational_velocity_rad_per_second': average_rotational_velocity_rad_per_second,
                'final/average_rotational_velocity_deg_per_second': average_rotational_velocity_rad_per_second * 180 / np.pi,
            }
            results_path = os.path.join(self.experiment_dir, 'results.yaml')
            with open(results_path, 'w') as f:
                yaml.dump(results, f)
            print(f'Wrote final metrics to {results_path}')

            if self.config['wandb_activate']:
                wandb.log(results)

        if self.record_video:
            video_process.join()
            print(f'Wrote video to {video_out_path}')
            
            if self.config['wandb_activate']:
                wandb.log({f'video': wandb.Video(video_out_path)})

        leap.close_connection_keep_position()

        if self.config['wandb_activate']:
            wandb.finish()


    def forward_network(self, obs):
        return self.player.get_action(obs, True)

    def restore(self):
        rlg_config_dict = self.config['train']
        rlg_config_dict["params"]["config"]["env_info"] = {}
        self.num_obs = self.config["task"]["env"]["numObservations"] if 'numObservations' in self.config["task"]["env"] else 102
        self.num_actions = self.embodiment_properties.dof_count
        observation_space = spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf)
        rlg_config_dict["params"]["config"]["env_info"]["observation_space"] = observation_space
        action_space = spaces.Box(np.ones(self.num_actions) * -1., np.ones(self.num_actions) * 1.)
        rlg_config_dict["params"]["config"]["env_info"]["action_space"] = action_space
        rlg_config_dict["params"]["config"]["env_info"]["agents"] = 1
        rlg_config_dict['params']['config']['device_name'] = self.config['rl_device']

        def build_runner(algo_observer):
            runner = Runner(algo_observer)

            register_network('embodiment_transformer', lambda **kwargs: EmbodimentTransformerBuilder())
            runner.player_factory.register_builder('a2c_continuous', lambda **kwargs : CustomPpoPlayerContinuous(**kwargs))

            return runner
        
        rlg_config_dict['params']['network']['robot_asset_root'] = '../../rl/assets' # used by EmbodimentTransformerBuilder

        runner = build_runner(AlgoObserver())
        runner.load(rlg_config_dict)
        runner.reset()

        args = {
            'train': False,
            'play': True,
            'checkpoint': self.config['checkpoint'],
            'sigma': None
        }

        self.player = runner.create_player()
        _restore(self.player, args)
        _override_sigma(self.player, args)

@hydra.main(config_name='config', config_path='../../rl/cfg')
def main(config: DictConfig):
    agent = HardwarePlayer(config)
    agent.restore()
    agent.deploy()

if __name__ == '__main__':
    main()
