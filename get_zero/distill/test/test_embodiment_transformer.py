import os
import yaml
import torch
import numpy as np
import hydra
from omegaconf import OmegaConf

import unittest
from unittest.mock import Mock

from get_zero.distill.models.embodiment_transformer import EmbodimentTransformer
from get_zero.distill.utils.embodiment_util import EmbodimentProperties
from get_zero.distill.utils.generic import add_custom_omegaconf_resolvers

class TestEmbodimentTransformer(unittest.TestCase):
    def get_sample_config(self):
        add_custom_omegaconf_resolvers()
        cfg_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../cfg')
        with hydra.initialize_config_dir(version_base="1.2", config_dir=cfg_path):
            cfg = hydra.compose(config_name="config")
        cfg = OmegaConf.to_container(cfg, resolve=True)

        # prepare model config
        model_cfg = cfg['model']
        cfg['model']['heads']['policy'] = {'output_dim': 1, 'squeeze_output_dim': True}

        # prepare tokenization config
        tokenization_cfg = cfg['tokenization']
        tokenization_cfg['globalObsSize'] = 4
        tokenization_cfg['localObsSize'] = 5
        tokenization_cfg['enableGeometryEncoding'] = False

        return model_cfg, tokenization_cfg
    
    def load_embodiment(self, name, joint_name_to_joint_i='file', **kwargs):
        """Either provide joint_name_to_joint_i or it is read from a yaml file in the `cfg` dir."""
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(dir_path, 'assets', f'{name}.urdf')) as f:
            urdf = f.read()
        
        if joint_name_to_joint_i == 'file':
            with open(os.path.join(dir_path, 'cfg', f'{name}.yaml')) as f:
                yaml_contents = yaml.safe_load(f)
            
            joint_name_to_joint_i = yaml_contents['joint_name_to_joint_i']
        
        embodiment_properties = EmbodimentProperties(name, urdf, joint_name_to_joint_i, **kwargs)
        return embodiment_properties
    
    def get_sample_input(self, tokenization_config, dof_count, batch_size=5):
        global_obs_size = tokenization_config['globalObsSize']
        local_obs_size = tokenization_config['localObsSize']

        obs_size = global_obs_size + local_obs_size * dof_count
        inp = torch.rand((batch_size, obs_size))
        return inp
    
    def permute_local_tokens(self, inp, tokenization_config, new_ordering):
        """Reorders the local tokens in a given model input according to the specified ordering"""
        global_obs_size = tokenization_config['globalObsSize']
        local_obs_size = tokenization_config['localObsSize']

        global_obs = inp[:, :global_obs_size]
        local_obs = inp[:, global_obs_size:]

        num_local_tokens = local_obs.size(1) // local_obs_size
        local_tokens_chunked = torch.chunk(local_obs, num_local_tokens, dim=1)
        local_tokens_chunked_permuted = [local_tokens_chunked[i] for i in new_ordering]
        local_tokens_permuted = torch.cat(local_tokens_chunked_permuted, dim=1)
        
        return torch.cat((global_obs, local_tokens_permuted), dim=1)

    def pad_input(self, tokenization_config, inp, padding_count):
        local_obs_size = tokenization_config['localObsSize']
        batch_size = inp.size(0)
        padding_size = local_obs_size * padding_count
        padded_inp = torch.cat((inp, torch.rand((batch_size, padding_size))), dim=1) # dof_count valid tokens with padding_count tokens of padding
        return padded_inp

    def test_mask(self):
        """Ensures that padding tokens don't impact output of the model. Specifically we take an input (and a padded version) and pass them through the model to ensure they have the same outputs regardless of the padding."""
        padding_tokens = 6
        
        # load embodiment
        embodiment_properties = self.load_embodiment('001')
        dof_count = embodiment_properties.dof_count

        # make a mock embodiment, so that we have some embodiment with dof count enough to support the padding
        padded_dof_count = dof_count + padding_tokens
        pad_embodiment_properties = Mock()
        pad_embodiment_properties.dof_count = padded_dof_count
        pad_embodiment_properties.adjacency_matrix = 1 - torch.diag(torch.tensor([1] * padded_dof_count))
        pad_embodiment_properties.compute_parent_and_child_matrix = Mock(return_value=(torch.zeros((padded_dof_count, padded_dof_count), dtype=torch.int), torch.zeros((padded_dof_count, padded_dof_count), dtype=torch.int)))
        pad_embodiment_properties.compute_edge_matrix = Mock(return_value=(torch.zeros((padded_dof_count, padded_dof_count), dtype=torch.int)))

        # create model
        model_cfg, tokenization_config = self.get_sample_config()
        model_cfg['max_dof_count'] = max(model_cfg['max_dof_count'], dof_count + padding_tokens)
        model_cfg['transformer']['graphormer_attention_embedding_init'] = 'normal' # need to ensure embedding weights are not zero init to make this test valid
        model = EmbodimentTransformer(model_cfg, tokenization_config, [embodiment_properties, pad_embodiment_properties])
        model.eval()

        # get an input, then construct a padded version
        batch_size = 10
        input1 = self.get_sample_input(tokenization_config, dof_count, batch_size)
        input2 = self.pad_input(tokenization_config, input1, padding_tokens)

        embodiment_ids = torch.full((batch_size,), 0, dtype=torch.long)

        with torch.inference_mode():
            actions1 = model(input1, embodiment_ids)['policy']
            actions2 = model(input2, embodiment_ids)['policy']

        # check that outputs from model are the same regardless of padding
        actions2 = actions2[:, :dof_count]

        self.assertTrue(input1.shape != input2.shape)
        self.assertTrue(torch.allclose(input1, input2[:, :input1.size(1)]))
        self.assertTrue(torch.allclose(actions1, actions2))

    def test_graphormer_embedding(self):
        self._test_graphormer_embedding('none')
        self._test_graphormer_embedding('centrality')

    def _test_graphormer_embedding(self, positional_encoding):
        """Ensures that the graphormer embedding means that we get the exact same output (permuted) when was pass a permuted version of the input that has the same graph connectivity."""
        # load embodiment with two different joint orderings
        ordering1 = {'0': 0, '1': 1, '2': 2, '3': 3}
        ordering2 = {'0': 2, '1': 0, '2': 3, '3': 1}
        embodiment_properties_ordering1 = self.load_embodiment('toy_example', joint_name_to_joint_i=ordering1, skip_geo_load=True)
        embodiment_properties_ordering2 = self.load_embodiment('toy_example', joint_name_to_joint_i=ordering2, skip_geo_load=True)
        dof_count = embodiment_properties_ordering1.dof_count

        reordering = [None] * len(ordering1.keys()) # index i in reordering indicates which piece of input1 goes into spot i of input2
        for joint_name_o1, joint_i_o1 in ordering1.items():
            joint_i_o2 = ordering2[joint_name_o1]
            reordering[joint_i_o2] = joint_i_o1
        
        # create model
        model_cfg, tokenization_config = self.get_sample_config()
        model_cfg['graphormer']['positional_encoding'] = positional_encoding
        model_cfg['graphormer']['attention']['enable_edge_encoding'] = False # we are using skip_geo_load when loading the embodiment, so we can use edge encoding
        model = EmbodimentTransformer(model_cfg, tokenization_config, [embodiment_properties_ordering1, embodiment_properties_ordering2])
        model.eval()

        # get input, then another with a permuted input
        batch_size = 10
        input1 = self.get_sample_input(tokenization_config, dof_count, batch_size)
        input2 = self.permute_local_tokens(input1, tokenization_config, reordering)

        # run model
        with torch.inference_mode():
            actions1 = model(input1, torch.full((batch_size,), 0, dtype=torch.long))['policy']
            actions2 = model(input2, torch.full((batch_size,), 1, dtype=torch.long))['policy']

        # reorder the output tokens
        actions1 = actions1[:, reordering]

        # validate model outputs are exactly the same despite different ordering
        actions_close = torch.allclose(actions1, actions2, atol=1e-6)
        if not actions_close:
            print(f'actions1: {actions1}\nactions2: {actions2}')
            print('diff:', torch.abs(actions1 - actions2).sum())

        self.assertTrue(actions_close)

    def test_gpu_inference(self):
        """validates that inference works when the model is on the GPU"""
        device = torch.device('cuda:0')

        # load embodiment
        embodiment_properties = self.load_embodiment('001')
        dof_count = embodiment_properties.dof_count

        # create model
        model_cfg, tokenization_config = self.get_sample_config()
        model = EmbodimentTransformer(model_cfg, tokenization_config, [embodiment_properties]).to(device)
        model.eval()

        # get an input, then construct a padded version
        input1 = self.get_sample_input(tokenization_config, dof_count).to(device)
        embodiment_ids = torch.full((input1.size(0),), 0, dtype=torch.long, device=device)

        with torch.inference_mode():
            actions1 = model(input1, embodiment_ids)['policy']

        self.assertEqual(actions1.device, device)

if __name__ == '__main__':
    unittest.main()

# TODO: add test for disjoint set of joints and check Graphormer attention weights
