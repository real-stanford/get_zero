import os
import torch

from rl_games.algos_torch.network_builder import NetworkBuilder
from get_zero.distill.models.embodiment_transformer import EmbodimentTransformer
from get_zero.distill.utils.embodiment_util import EmbodimentProperties

class EmbodimentTransformerBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        net = EmbodimentTransformerBuilder.Network(self.params, **kwargs)
        return net
    
    class Network(NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            NetworkBuilder.BaseNetwork.__init__(self)

            # Load params
            model_cfg = params['actor']
            tokenization_cfg = params['tokenization']

            # From kwargs
            _input_shape = kwargs.pop('input_shape')[0] # corresponds to observation size: (observation_dim, ); not used
            self.value_size = kwargs.pop('value_size', 1) # size of critic output
            self.max_actions_num = kwargs.pop('actions_num') # corresponds max DoF
            _num_seqs = kwargs.pop('num_seqs', 1) # number of environments in sim; not used

            # Setup embodiment properties
            robot_asset = params['robot_asset']
            robot_asset_root = params['robot_asset_root'] if 'robot_asset_root' in params else 'assets'
            with open(os.path.join(robot_asset_root, robot_asset)) as f:
                asset_file_contents = f.read()
            joint_name_to_joint_i = params['joint_name_to_joint_i']
            name = os.path.basename(robot_asset).replace('.urdf', '').replace('.mjcf', '')
            embodiment_properties = [EmbodimentProperties(name, asset_file_contents, joint_name_to_joint_i)]

            # Actor
            self.embodiment_transformer = EmbodimentTransformer(model_cfg, tokenization_cfg, embodiment_properties)
            self.embodiment_transformer.eval()

            # Enable `policy` head to get actions and, if present, `policy_forward_kinematics` head to get FK predictions
            self.heads_to_use = ['policy']
            head_names = self.embodiment_transformer.get_head_names()
            if 'policy_forward_kinematics' in head_names:
                self.heads_to_use.append('policy_forward_kinematics')

        @torch.inference_mode()
        def forward(self, obs_dict):
            obs = obs_dict['obs']
            num_envs = obs.size(0)

            embodiment_ids = torch.zeros((num_envs, ), dtype=torch.int, device=obs.device) # only one embodiment -> id 0
            results_by_head = self.embodiment_transformer(obs, embodiment_ids, head_names=self.heads_to_use) # only compute outputs desired heads
            mu = results_by_head.pop('policy') # pop to remove policy head and just leave the remaining heads in `results_by_head`

            # TODO: don't have to re-initialize value and log sigma during every forward pass

            # we only use this network at test time, so we can just set value to all zero
            value = torch.zeros((num_envs, ), device=obs.device)

            # our model doesn't produce sigma estimates, so set to all -inf
            log_sigma = torch.full(mu.shape, -torch.inf, device=mu.device)

            # the return format is (mean, log_sigma, value, rnn_states). We are abusing the return type of rnn_states to return the outputs of the other heads from the model, such that this information can be accessed by the RL task later on for visualization purposes (see PpoPlayerContinuous)
            return mu, log_sigma, value, results_by_head 
