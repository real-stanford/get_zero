# Modified from IsaacGymEnvs

# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import hydra

from omegaconf import DictConfig, OmegaConf

from get_zero.rl.utils.generic_util import register_custom_omegaconf_resolvers
register_custom_omegaconf_resolvers()

@hydra.main(version_base="1.2", config_name="config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):
    rl_runner(cfg)

def rl_runner(cfg: DictConfig, runner_only=False):
    """
    Args:
    - runner_only: if True, then skips the training and testing phases and just returns the runner

    Returns:
    - if runner_only==True, returns a function with signature `def start_runner(is_train)` which can be used to start testing or training runs
    """

    # set the experiment name
    from datetime import datetime
    import os
    import copy
    import random
    experiment = f"_{cfg.experiment}" if cfg.experiment else ''
    train_or_test_str = 'test' if cfg.test else 'train'
    
    experiment_name = f"{cfg.task.name}{experiment}_{train_or_test_str}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')}_{random.randint(0,100000)}"
    experiment_dir = os.path.join('runs', experiment_name)
    os.makedirs(experiment_dir)
    print(f'\n----- LOGGING RL EXPERIMENT TO {os.path.abspath(experiment_dir)} -----')

    # log config to a file
    cfg_initial = copy.deepcopy(cfg)
    with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg_initial))

    """Before starting and loading depedencies (namely before Isaac Gym is imported), setup which CUDA devices we are using, which will impact the Isaac Gym initialization."""
    if cfg.gpu != -1:
        cfg.sim_device = f'cuda:{cfg.gpu}'
        cfg.rl_device = f'cuda:{cfg.gpu}'
        cfg.graphics_device_id = cfg.gpu

    if cfg.smart_device_selection:
        # graphics_device_id is the vulkan device, which may not have the same ordering as the CUDA device. Therefore the graphics device should be set to match the cuda device so that it's easy to specify the right device
        # it seems like Isaac Gym has a bug where it assumes CUDA ordering matches graphics device ordering so if you set graphics device to x, but graphics device x corresponds to cuda:y, then it will put the camera image tensor on cuda:x (also completely disregarding CUDA_VISIBLE_DEVICES), when it is expected that it would go on cuda:y. This means that you should expect some memory allocation on the cuda:x device
        used_cuda_devices = set([int(cfg.rl_device.split(':')[1]), int(cfg.sim_device.split(':')[1])])
        if 'CUDA_ID_TO_VULKAN_ID' in os.environ and cfg.graphics_device_id != -1:
            cuda_id_to_vulkan_id = [int(x) for x in os.environ['CUDA_ID_TO_VULKAN_ID'].split(" ")]
            cuda_graphics_device_id = cfg.graphics_device_id
            cfg.graphics_device_id = cuda_id_to_vulkan_id[cuda_graphics_device_id]
            print(f'Remapping graphics device from {cuda_graphics_device_id} (CUDA ordering) to {cfg.graphics_device_id} (Vulkan ordering)')
            used_cuda_devices.add(cuda_graphics_device_id)
        else:
            used_cuda_devices.add(cfg.graphics_device_id)

        # we are going to set CUDA_VISIBLE_DEVICES so that when isaac gym gym.create_sim is called, it only sees devices that we are actually using (to avoid issues where Isaac Gym would allocate memory on devices that weren't actually be used)
        if not cfg.task.get('uses_gpu_camera_pipeline', True):
            assert 'CUDA_VISIBLE_DEVICES' not in os.environ, 'when using `smart_device_selection` and the task has `uses_gpu_camera_pipeline` as True, you should not specify CUDA_VISIBLE_DEVICES'
        
            # if we use CPU camera pipeline, then it's ok to remap the ordering of CUDA devices on only make the ones we are using visible to get around the issue of Isaac Gym allocating memory on devices we aren't using
            # with the GPU camera pipeline the image tensor ends up cuda:graphics_device_id, so we can't safely change the ordering of CUDA devices with CUDA_VISIBLE_DEVICES
            used_cuda_devices = list(used_cuda_devices)
            used_cuda_devices.sort()
            cuda_visible_devices_str = ','.join([str(x) for x in used_cuda_devices])
            os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices_str
            print(f'setting CUDA_VISIBLE_DEVICES={cuda_visible_devices_str}')

            # based on values set in CUDA_VISIBLE_DEVICES we need to remap the index of each device
            # note we do not need to do this for the graphics device becuase it does not necessarily use CUDA ordering
            prior_to_new_gpu_index = {}
            remap_counter = 0
            for gpu_id in used_cuda_devices:
                prior_to_new_gpu_index[gpu_id] = remap_counter
                remap_counter += 1

            cfg.rl_device = f'cuda:{prior_to_new_gpu_index[int(cfg.rl_device.split(":")[1])]}'
            cfg.sim_device = f'cuda:{prior_to_new_gpu_index[int(cfg.sim_device.split(":")[1])]}'

    # Now that devices are properly selected, we can import dependencies
    import isaacgym
    import torch
    from get_zero.rl.tasks import isaacgym_task_map
    import gym
    from isaacgymenvs.utils.reformat import omegaconf_to_dict
    from isaacgymenvs.utils.utils import set_np_formatting, set_seed
    from isaacgymenvs.train import preprocess_train_config
    from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, MultiObserver, RLGPUAlgoObserver, ComplexObsRLGPUEnv
    from get_zero.rl.utils.logging_util import WandbAlgoObserver
    import isaacgymenvs.tasks.base.vec_task as vec_task
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner
    from rl_games.algos_torch.model_builder import register_network
    from rl_games.algos_torch.a2c_continuous import A2CAgent
    from get_zero.rl.models.embodiment_transformer import EmbodimentTransformerBuilder
    from get_zero.rl.utils.rl_util import CustomPpoPlayerContinuous
    import get_zero.rl
    from get_zero.rl.utils.logging_util import WandBVideoRecorder
    from gym.wrappers import RecordVideo
    from sys import maxsize
    from get_zero.rl.tasks import leap_hand_rot
    from isaacgym import gymapi
    import wandb

    # set numpy formatting for printing only
    set_np_formatting()

    # global rank of the GPU
    global_rank = int(os.getenv("RANK", "0"))

    # sets seed. if seed is -1 will pick a random one
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=global_rank)

    # update video capture resolution for virtual display
    virtual_screen_resolution = (gymapi.DEFAULT_VIEWER_WIDTH, gymapi.DEFAULT_VIEWER_HEIGHT)
    vec_task.SCREEN_CAPTURE_RESOLUTION = virtual_screen_resolution
    leap_hand_rot.SCREEN_CAPTURE_RESOLUTION = virtual_screen_resolution

    assert not (not cfg.headless and cfg.record_video_source == 'viewer') or not cfg.record_video, 'Having viewer visible and also recording from the viewer is not supported'

    envs = None
    def create_isaacgym_env(is_test, **kwargs):
        nonlocal envs

        # Only create a single environment across both training and testing
        # TODO: when testing after training it's inefficient to run a lot of environments in parallel when often the video recording will only capture a small subset of those environments. One option is to create a new sim with much fewer environments and then run testing.
        if envs is None:
            print('Initializing Isaac Gym environment')
            envs = get_zero.rl.make(
                cfg.seed, 
                cfg.task_name, 
                cfg.task.env.numEnvs, 
                cfg.sim_device,
                cfg.rl_device,
                cfg.graphics_device_id,
                cfg.headless and not (cfg.record_video_source == 'viewer' and cfg.record_video),
                cfg.multi_gpu,
                cfg.record_video_source == 'viewer' and cfg.record_video,
                not cfg.headless,
                cfg,
                **kwargs,
            )

        if hasattr(envs, 'start'):
            envs.start(not is_test, experiment_dir)
        
        if cfg.record_video:
            envs.is_vector_env = True

            # For test, capture video of the entire episode. For train, capture video according to capture_train_video_freq and capture_train_video_len in the config
            if is_test or cfg.record_entire_train_video:
                cfg.capture_train_video_freq = 1
                cfg.capture_train_video_len = maxsize

            video_prefix = 'test' if is_test else 'train'
            video_recorder_class = WandBVideoRecorder if cfg.wandb_activate else RecordVideo
            envs_wrapped = video_recorder_class(
                envs,
                f"{experiment_dir}/videos",
                step_trigger=lambda step: step % cfg.capture_train_video_freq == 0,
                video_length=cfg.capture_train_video_len,
                name_prefix=video_prefix
            )
        else:
            envs_wrapped = envs
        
        return envs_wrapped

    ige_env_cls = isaacgym_task_map[cfg.task_name]
    dict_cls = ige_env_cls.dict_obs_cls if hasattr(ige_env_cls, 'dict_obs_cls') and ige_env_cls.dict_obs_cls else False

    if dict_cls:         
        obs_spec = {}
        actor_net_cfg = cfg.train.params.network
        obs_spec['obs'] = {'names': list(actor_net_cfg.inputs.keys()), 'concat': not actor_net_cfg.name == "complex_net", 'space_name': 'observation_space'}
        if "central_value_config" in cfg.train.params.config:
            critic_net_cfg = cfg.train.params.config.central_value_config.network
            obs_spec['states'] = {'names': list(critic_net_cfg.inputs.keys()), 'concat': not critic_net_cfg.name == "complex_net", 'space_name': 'state_space'}
        
        vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: ComplexObsRLGPUEnv(config_name, num_actors, obs_spec, **kwargs))
    else:

        vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))

    # if embodiment aware policy provided, then we need to load in model and tokenization configuration that is stored in the checkpoint file unless it has already been specified
    if cfg.checkpoint:
        checkpoint_data = torch.load(cfg.checkpoint, map_location='cpu')
        if 'tokenization_config' in checkpoint_data and 'model_config' in checkpoint_data:
            if cfg.train.params.network.actor is None:
                print(f'Using tokenization config and model config from the checkpoint at {cfg.checkpoint}')
                cfg.task.env.tokenization = OmegaConf.create(checkpoint_data['tokenization_config'])
                cfg.train.params.network.actor = OmegaConf.create(checkpoint_data['model_config'])
            else:
                print('Provided checkpoint has tokenization and model configuration, but since it was already specified, not going to load any configuration from the checkpoint')

    rlg_config_dict = omegaconf_to_dict(cfg.train)
    rlg_config_dict = preprocess_train_config(cfg, rlg_config_dict) # TODO: this has extra print statements that could be avoided for cleaner output
    rlg_config_dict['params']['config']['device_name'] = rlg_config_dict['params']['config']['device'] # For some reason rl_games looks for 'device' parameter during training process to figure out where to put the network (for example see A2CBase in rl_games), but looks for the 'device_name' in the testing process (for example see BasePlayer in rl_games). This line makes sure both parameters are set, so that the network is put on the right device (rl_device)
    rlg_config_dict['params']['config']['full_experiment_name'] = experiment_name

    rl_gpu_algo_observer = RLGPUAlgoObserver()
    observers = [rl_gpu_algo_observer]

    if cfg.wandb_activate:
        cfg.seed += global_rank
        if global_rank == 0:
            # initialize wandb only once per multi-gpu run
            wandb_observer = WandbAlgoObserver(cfg_initial)
            observers.append(wandb_observer)

    # register custom network builder and agent
    def build_runner(algo_observer):
        runner = Runner(algo_observer)

        # Additional networks
        register_network('embodiment_transformer', lambda **kwargs: EmbodimentTransformerBuilder())

        # Overwrite default a2c_continuous implementations with custom modifications
        runner.player_factory.register_builder('a2c_continuous', lambda **kwargs : CustomPpoPlayerContinuous(**kwargs))

        return runner
    
    # convert CLI arguments into dictionary
    # create runner and set the settings
    runner = build_runner(MultiObserver(observers))
    runner.load(rlg_config_dict)
    runner.reset()

    algo_observer_has_init = False
    def start_runner(is_train, is_last_run=False):
        nonlocal algo_observer_has_init
        if is_train:
            algo_observer_has_init = True
        elif not algo_observer_has_init:
            # The learning algorithm in RL Games initializes the observers, but only if in train mode, so enable here for test mode
            runner.algo_observer.before_init(experiment_name, cfg, experiment_name)
            algo_observer_has_init = True

        env_configurations.register('rlgpu', {
            'vecenv_type': 'RLGPU',
            'env_creator': lambda **kwargs: create_isaacgym_env(not is_train, **kwargs),
        })

        mode_str = 'train' if is_train else 'test'
        print(f'Starting {mode_str} runner')
        runner.run({
            'train': is_train,
            'play': not is_train,
            'checkpoint': cfg.checkpoint,
            'sigma': cfg.sigma if cfg.sigma != '' else None
        })
        
        if hasattr(envs, 'finish'):
            envs.finish()

        rl_gpu_algo_observer.writer.close() # force tensorboard logs to close and flush after experiment runs; a new tensorboard SummaryWriter will be created for each call to this runner, so it should be closed each time
        
        if is_last_run:
            if cfg.wandb_activate:
                wandb.finish()

        if is_train:
            cfg.checkpoint = os.path.join('runs', experiment_name, 'nn', f'{cfg.train.params.config.name}.pth')

        return experiment_dir

    """Either execute the runner to optionally train and then always test or return it to be executed elsewhere"""
    if not runner_only:
        # Train if requested
        if not cfg.test:
            start_runner(True)
        
        # Always test
        start_runner(False, is_last_run=True)
    
    if runner_only:
        return start_runner

if __name__ == "__main__":
    launch_rlg_hydra()
