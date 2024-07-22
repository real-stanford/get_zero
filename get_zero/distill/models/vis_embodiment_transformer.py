usage="""
Visualizes key learned parameters from a trained embodiment transformer.

Runs in one of three modes:
- provide (-c, -m and -t)
- (-r -c) (run dir assumed to have folders `checkpoints/{-c}`, `configs/Model.yaml` and `configs/Tokenization.yaml`)
- (-r -c) (run dir assumed to have folders `checkpoints/{-c}` and `cfg.yaml`. `cfg.yaml` should have keys `model` and `tokenization`)
"""

from argparse import ArgumentParser
import torch
import os
import yaml
import matplotlib.pyplot as plt
import numpy as np

from get_zero.distill.models.embodiment_transformer import EmbodimentTransformer, GraphormerPositionalEmbedding
from get_zero.distill.models.embodiment_attention import SpatialEncodingTransformerEncoderLayer

def parse_args():
    parser = ArgumentParser(usage=usage)
    parser.add_argument('--checkpoint', '-c', help='path to learned embodiment transformer model checkpoint if -r not provided or name of checkpoint if -r provided')
    parser.add_argument('--model_config_path', '-m', default=None, help='path to model config to initialize embodiment transformer')
    parser.add_argument('--tokenization_config_path', '-t', default=None, help='path to tokenization config to initialize embodiment transformer')
    parser.add_argument('--run_path', '-r', default=None, help='path to run to pull config and checkpoint from')
    return parser.parse_args()

def vis_embodiment_transformer(out_dir, checkpoint_path=None, model_config=None, tokenization_config=None, model=None):
    """Provide either 1) `model` or 2) `checkpoint_path`, `model_config` and `tokenization_config`"""
    assert model is not None or (checkpoint_path is not None and model_config is not None and tokenization_config is not None)
    plt.ioff()
    generated_figures = {}

    if model is None:
        # create model and load checkpoint
        model = EmbodimentTransformer(model_config, tokenization_config, [])
        model.load_state_dict(torch.load(checkpoint_path)['model'])
        
    # output dir
    os.makedirs(out_dir, exist_ok=True)

    def plot_embedding(embedding_name: str, x_label: str, start_x: int = 0):
        embedding_lst = []
        for layer in model.encoder.layers:
            if not hasattr(layer, embedding_name):
                return
            embedding = getattr(layer, embedding_name).weight.detach().cpu() # (max dof count, num attention heads)
            embedding = embedding.mean(dim=1) # (dof count, ); average over attention heads
            embedding_lst.append(embedding)
        embedding = torch.stack(embedding_lst) # (num layers, dof count)

        path = os.path.join(out_dir, f'graphormer_{embedding_name}.png')
        fig, axes = plt.subplots()
        cax = axes.matshow(embedding.numpy())
        axes.set_xticks(np.arange(0, embedding.size(1)), labels=np.arange(start_x, embedding.size(1) + start_x))
        axes.set_xlabel(x_label)
        axes.set_yticks(np.arange(0, embedding.size(0)), labels=np.arange(1, embedding.size(0)+1))
        axes.set_ylabel('Encoder layer')
        axes.set_title(f'Graphormer {embedding_name} (averaged over heads)')
        fig.colorbar(cax)
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'Wrote to {path}')
        generated_figures[embedding_name] = path

    # visualize learned attention reweighting
    if type(model.encoder.layers[0]) == SpatialEncodingTransformerEncoderLayer:
        plot_embedding('spatial_embedding', 'Shortest path distance (-1 indicates not connected)', start_x=-1)
        plot_embedding('parent_embedding', 'Distance from joint (0 indicates not a parent)')
        plot_embedding('child_embedding', 'Distance from joint (0 indicates not a child)')

    return generated_figures

if __name__ == '__main__':
    args = parse_args()

    """Prepare the model from the config and checkpoint"""
    # determine paths
    if args.run_path:
        checkpoint_path = os.path.join(args.run_path, 'checkpoints', args.checkpoint)
        model_config_path = os.path.join(args.run_path, 'configs', 'Model.yaml')
        tokenization_config_path = os.path.join(args.run_path, 'configs', 'Tokenization.yaml')
        run_name = os.path.basename(args.run_path)
    else:
        checkpoint_path = args.checkpoint
        model_config_path = args.model_config_path
        tokenization_config_path = args.tokenization_config_path
        run_name = os.path.basename(checkpoint_path)

    # load configs
    if os.path.exists(model_config_path):
        with open(model_config_path, 'r') as f:
            model_config = yaml.safe_load(f)
        with open(tokenization_config_path, 'r') as f:
            tokenization_config = yaml.safe_load(f)
    else:
        # load from `{run_path}/config.yaml`
        cfg_path = os.path.join(args.run_path, 'config.yaml')
        with open(cfg_path, 'r') as f:
            config = yaml.safe_load(f)
        model_config = config['model']
        tokenization_config = config['tokenization']

    out_dir = os.path.join('tmp', run_name)
    vis_embodiment_transformer(out_dir, checkpoint_path, model_config, tokenization_config)
