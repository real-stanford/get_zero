from rl_games.algos_torch.a2c_continuous import A2CAgent
from rl_games.algos_torch.players import PpoPlayerContinuous
import time
import os
import torch
from tensorboardX import SummaryWriter
from tqdm import trange

class DummyLoggingAlgo:
    """
    The AlgoObserver classes assume they have an algorithm class instance (such as A2CBase from rl_games) that has specific properties that are used in the logging process. This class mimics those same properties that are accessed by the AlgoObserver, so that during test time we can still log stats out.
    """
    def __init__(self, games_to_track, summaries_dir, device):
        self.games_to_track = games_to_track
        self.writer = SummaryWriter(summaries_dir)
        self.device = device

class CustomPpoPlayerContinuous(PpoPlayerContinuous):
    """
    Changes:
      - runs the environment for a fixed number of steps rather than a fixed number of "games" (enviornment resets).
      - for `embodiment_transformer` network, support loading checkpoint that comes from trained network outside of rl_games
      - support logging of stats produced by the task with algo observers
    """
    def __init__(self, params):
        self.network_name = params['network']['name']
        super().__init__(params)

        self.play_steps = self.config['player']['play_steps']
        self.algo_observer = self.config['features']['observer']

        # Experiment dir for logging
        self.experiment_name = self.config['full_experiment_name']
        self.experiment_dir = os.path.join('runs', self.experiment_name)
        self.summaries_dir = os.path.join(self.experiment_dir, 'summaries')

        # Init algo observer
        self.algo_observer.before_init('BaseNamePlaceHolder', params, self.experiment_name)
        self.algo_observer.after_init(DummyLoggingAlgo(1, self.summaries_dir, self.device))

    def run(self):
        # Adapted from player.py in rl_games
        render = self.render_env
        is_deterministic = self.is_deterministic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        games_played = 1
        n_game_life = 1
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None
        start_time = time.time()

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        self.wait_for_checkpoint()

        need_init_rnn = self.is_rnn

        obses = self.env_reset(self.env)
        batch_size = 1
        batch_size = self.get_batch_size(obses, batch_size)

        if need_init_rnn:
            self.init_rnn()
            need_init_rnn = False

        for n in trange(self.play_steps, desc='RL Play Steps'):
            if self.evaluation and n % self.update_checkpoint_freq == 0:
                self.maybe_load_new_checkpoint()

            if has_masks:
                masks = self.env.get_action_mask()
                action = self.get_masked_action(
                    obses, masks, is_deterministic)
            else:
                action = self.get_action(obses, is_deterministic)

            # This is definitely hacky, but the idea is the model can produce a forward kinematics estimate and we want to pass this information to the task for visualization.
            # However, there is no easy way to have this information returned, so we use the running rnn_state as a storage location for the additional model outputs.
            # We check if forward kinematics is present because potentially the rnn_state is actually used as an RNN state and we want that functionality to still work
            if hasattr(self.env, 'update_forward_kinematics_estimate') and type(self.states) == dict and 'policy_forward_kinematics' in self.states:
                self.env.update_forward_kinematics_estimate(self.states['policy_forward_kinematics'])

            obses, r, done, info = self.env_step(self.env, action)
            sum_rewards += r.sum().item()
            sum_steps += 1

            if render:
                self.env.render(mode='human')
                time.sleep(self.render_sleep)

            # logging
            all_done_indices = done.nonzero(as_tuple=False)
            self.algo_observer.process_infos(info, all_done_indices)
            self.algo_observer.after_print_stats(n, 0, time.time() - start_time)

        print('av reward:', sum_rewards / games_played * n_game_life,
              'av steps:', sum_steps / games_played * n_game_life)

    def restore(self, fn):
        """
        Changes from original:
        - load checkpoint on the RL device rather than the device stored in the file (which may not even exist)
        - support loading embodiment transformer trained policies (which are trained externally rather than within RL games)
        """
        print("=> loading checkpoint '{}'".format(fn))
        checkpoint = torch.load(fn, map_location=self.device)
        # for the embodiment_transformer, the checkpoint was generated outside rl_games, so we need to load it into the right place
        if self.network_name == 'embodiment_transformer':
            self.model.a2c_network.embodiment_transformer.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint['model'])
        
        if self.normalize_input and 'running_mean_std' in checkpoint:
            self.model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

        env_state = checkpoint.get('env_state', None)
        if self.env is not None and env_state is not None:
            self.env.set_env_state(env_state)
