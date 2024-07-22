import os
from gym.wrappers import RecordVideo
import wandb
import torch

from rl_games.common.algo_observer import AlgoObserver
from get_zero.rl.utils.generic_util import omegaconf_to_dict
from isaacgymenvs.utils.utils import flatten_dict

class WandBVideoRecorder(RecordVideo):
    def close_video_recorder(self) -> None:
        recording = self.recording
        recorded_frames = self.recorded_frames
        super().close_video_recorder()

        if recording:
            if os.path.exists(self.video_recorder.path):
                video_start_step = self.step_id + 1 - recorded_frames
                video_end_step = self.step_id
                wandb.log({
                    f'videos/video_{self.name_prefix}': wandb.Video(self.video_recorder.path, caption=f'steps {video_start_step} to {video_end_step}'),
                    f'videos/video_step_{self.name_prefix}': video_start_step}) 
            else:
                print('Warning: video recorder closed, but no video was found to upload to W&B')  

def crop_isaacgym_viewer_sidebar(img):
    # the IsaacGym viewer has a sidebar with some options overlaid, so crop it out here
    return img[:, 400:, :] # TODO: setting directly to 400 is a hack; see if we can get gym.get_viewer_camera_handle working to directly access camera feed. This function was causing crash, but it seems that viewer may just be the camera with id 0, so we can just access this directly

class WandbAlgoObserver(AlgoObserver):
    # Adapted from implementation of RLGPUAlgoObserver and WandbAlgoObserver in IsaacGymEnvs

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def before_init(self, base_name, config, experiment_name):
        wandb_unique_id = f"uid_{experiment_name}"
        wandb.require("service") # needed for running wandb in subprocess (https://docs.wandb.ai/guides/track/log/distributed-training)
        wandb.init(
            project=self.cfg.wandb_project,
            entity=self.cfg.wandb_entity,
            group=self.cfg.wandb_group,
            tags=self.cfg.wandb_tags,
            id=wandb_unique_id,
            name=experiment_name
        )

        if isinstance(self.cfg, dict):
            wandb.config.update(self.cfg, allow_val_change=True)
        else:
            wandb.config.update(omegaconf_to_dict(self.cfg), allow_val_change=True)

    def process_infos(self, infos, done_indices):
        # turn nested infos into summary keys (i.e. infos['scalars']['lr'] -> infos['scalars/lr']
        if len(infos) > 0 and isinstance(infos, dict):  # allow direct logging from env
            infos_flat = flatten_dict(infos, prefix='', separator='/')
            self.scalar_info = {}
            for k, v in infos_flat.items():
                # only log scalars
                if isinstance(v, float) or isinstance(v, int) or (isinstance(v, torch.Tensor) and len(v.shape) == 0):
                    self.scalar_info[k] = v

    def after_print_stats(self, frame, epoch_num, total_time):
        if frame % self.cfg.wandb_log_freq == 0:
            wandb.log(self.scalar_info)
    