import torch
from torch import nn, Tensor
import math
from typing import Dict, List
from get_zero.distill.utils.embodiment_util import EmbodimentProperties
from torch.nn import TransformerEncoderLayer
from torch.nn import functional as F
from get_zero.distill.models.embodiment_attention import EmbodimentTransformerEncoder, SpatialEncodingTransformerEncoderLayer

class EmbodimentTransformer(nn.Module):
    def __init__(self, model_cfg: Dict, tokenization_cfg: Dict, embodiment_properties_by_id: List[EmbodimentProperties]):
        """
        model_cfg: see distill/cfg/model/Transformer.yaml for required values
        tokenization_cfg: matching format of tokenization config from distill
        embodiment_properties_by_id: list of EmbodimentProperties index by embodiment ID
        """
        super().__init__()

        if len(embodiment_properties_by_id) == 0:
            print('WARNING: no embodiments provided when initializing EmbodimentTransformer. You will be able to initalize the model, but the forward pass may error depending on whether the model config includes components that rely on embodiment information')

        # prepare embodiment properties
        self.num_embodiment_types = len(embodiment_properties_by_id)
        dof_counts_by_id = []
        self.max_dof_count = model_cfg.get('max_dof_count', None) # optional, as some configurations won't require the use of this value
        self.max_degree_count = model_cfg.get('max_degree_count', None) # optional, as some configurations won't require the use of this value
        self.provided_embodiment_max_dof_count = 0 # The max DoF count of the embodiments in `embodiment_properties_by_id`. Some layers don't require knowledge of the max dof count possible (since they scale with DoF count, such as GraphormerPositionalEmbedding), while others require a max expected dof count (such as FixedPositionalEncoding)
        for embodiment_properties in embodiment_properties_by_id:
            dof_count = embodiment_properties.dof_count
            dof_counts_by_id.append(dof_count)
            self.provided_embodiment_max_dof_count = max(self.provided_embodiment_max_dof_count, dof_count)
        dof_counts_by_id = torch.tensor(dof_counts_by_id, dtype=torch.long)
        self.register_buffer('dof_counts_by_id', dof_counts_by_id, persistent=False)

        # compute token size
        self.global_obs_size = tokenization_cfg['globalObsSize']
        self.local_obs_size = tokenization_cfg['localObsSize']
        self.token_feature_dim = self.local_obs_size + self.global_obs_size

        # fixed local observation values that require embedding table (they will be formatted as integers in the observation)
        # TODO: this is a little hacky to have the embedding sizes defined here since they are also defined in embodiment_util.py and dataset.py
        embedding_properties = [ # (tokenization property name, number of metric values); assume that these values start at 0 and step by 1
            ('includeDegreeCount', self.max_dof_count),
            ('includeParentCount', self.max_dof_count),
            ('includeChildCount', self.max_dof_count),
            ('includeChildLinkId', self.max_dof_count + 1)
        ]
        self.obs_property_embeddings = nn.ModuleList()
        if tokenization_cfg['enableGeometryEncoding']:
            for obs_property_embedding_name, property_size in embedding_properties:
                if tokenization_cfg[obs_property_embedding_name]:
                    property_embed_dim = model_cfg['transformer']['obs_property_embed_dim']
                    self.token_feature_dim += property_embed_dim - 1 # going from 1 entry to `property_embed_dim` entries
                    self.obs_property_embeddings.append(nn.Embedding(property_size + 1, property_embed_dim)) # add 1 to account for padding tokens which will have value set to -1

        # input token layer norm
        self.enable_input_layer_norm = model_cfg['transformer']['enable_input_layer_norm']
        if self.enable_input_layer_norm:
            self.input_layer_norm = nn.LayerNorm([self.token_feature_dim]) # layer norm over the feature dimension of the token

        # token embedding
        self.token_embed_dim = model_cfg['transformer']['token_embed_dim']
        self.token_embedding = nn.Linear(self.token_feature_dim, self.token_embed_dim)

        # positional encoding
        self.pe_dropout = model_cfg['positional_encoding']['dropout']
        self.embodiment_encoding_type = model_cfg['embodiment_encoding']

        if self.embodiment_encoding_type == 'graphormer':
            self.positional_encoding_type = model_cfg['graphormer']['positional_encoding']
        else:
            self.positional_encoding_type = self.embodiment_encoding_type

        if self.positional_encoding_type == 'fixed':
            self.positional_encoding = FixedPositionalEncoding(self.token_embed_dim, self.max_dof_count)
        elif self.positional_encoding_type == 'learned':
            self.positional_encoding = LearnedPositionalEncoding(self.token_embed_dim, self.max_dof_count)
        elif self.positional_encoding_type == 'centrality':
            degree_counts_by_id = [embodiment_properties.adjacency_matrix.sum(0) for embodiment_properties in embodiment_properties_by_id]
            self.positional_encoding = GraphormerPositionalEmbedding(self.token_embed_dim, self.provided_embodiment_max_dof_count, self.max_degree_count, degree_counts_by_id)
        elif self.positional_encoding_type == 'none':
            self.positional_encoding = nn.Identity()
        else:
            raise NotImplementedError
        self.positional_encoding_dropout = nn.Dropout(p=self.pe_dropout)

        # transformer
        transformer_params = model_cfg['transformer']
        self.feedforward_dim = transformer_params['feedforward_dim']
        self.num_attention_heads = transformer_params['num_attention_heads']
        self.num_layers = transformer_params['num_layers']

        encoder_layer_kwargs = {
            'd_model': self.token_embed_dim,
            'nhead': self.num_attention_heads,
            'dim_feedforward': self.feedforward_dim,
            'norm_first': True, # found to have much more stable training performance in recent literature,
            'dropout': model_cfg['transformer']['dropout']
        }
        encoder_kwargs = {
            'num_layers': self.num_layers
        }
        if self.embodiment_encoding_type == 'graphormer':
            encoder_layer = SpatialEncodingTransformerEncoderLayer(embodiment_properties_by_id, self.max_dof_count, model_cfg['graphormer']['attention'], **encoder_layer_kwargs)
            self.encoder = EmbodimentTransformerEncoder(encoder_layer, **encoder_kwargs)
        else:
            encoder_layer = TransformerEncoderLayer(**encoder_layer_kwargs)
            self.encoder = nn.TransformerEncoder(encoder_layer, **encoder_kwargs)

        # initialize (potentially multiple) output heads
        self.head_configs = {}
        self.head_modules = nn.ModuleDict()
        for head_name, head_cfg in model_cfg['heads'].items():
            self.head_configs[head_name] = head_cfg = {**model_cfg['head_defaults'], **head_cfg}
            
            linear_head = self._build_mlp(self.token_embed_dim, head_cfg['units'], head_cfg['output_dim'], self._activation_name_to_module(head_cfg['activation'])) # maps from self.token_embed_dim to output_dim
            self.head_modules[head_name] = linear_head

    def _activation_name_to_module(self, name):
        if name == 'relu':
            return nn.ReLU
        elif name == 'elu':
            return nn.ELU
        else:
            raise NotImplementedError

    def _build_mlp(self, in_size, intermediate_sizes, out_size, activation):
        modules = []
        if len(intermediate_sizes) > 0:
            sizes = [in_size] + intermediate_sizes
            for i in range(len(sizes) - 1):
                modules.append(nn.Linear(sizes[i], sizes[i+1]))
                modules.append(activation())
            modules.append(nn.Linear(sizes[-1], out_size))
        else:
            modules.append(nn.Linear(in_size, out_size))

        return nn.Sequential(*modules)

    def get_head_names(self):
        """
        Returns a list of heads names that this model has. These names can then be provided to the forward function if you wish to enable specific heads.
        """
        return list(self.head_modules.keys())

    def forward(self, obs, embodiment_ids, head_names=None):
        """
        obs: (batch_size, (global_obs_size + local_obs_size * batch_max_dof_count)
        embodiment_ids: (batch_size, )
        if `head_names` is specified as a list of string, then only the heads with the given names will have outputs computed. Otherwise output for all heads.

        `obs` format:
        Global observation - global_obs_size
        Local observation joint 0 - local_obs_size
        Local observation joint 1 - local_obs_size
        ...

        Token format that is input into transformer:
        global observation - global_obs_size
        local observation - local_obs_size

        Not that when there is an embodiment with DoF count less than the max dof count in the observation, the observation must be padded (the length of this padding must be in increments of local_obs_size). You can pad with any value except for torch.inf or -torch.inf (these values result in NaNs even though padding is ignored).

        If a child link ID is included in the tokens, it is assumed that the last entry of the local observation corresponds to the child link ID.
        """

        """Format observation into tokens"""
        batch_size, obs_feature_size = obs.shape
        largest_dof_count_this_batch = (obs_feature_size - self.global_obs_size) // self.local_obs_size # based on the size of the observation vector provided, we can infer the max number of tokens provided, which is equal to the DoF count since 1 token per DoF
        dof_counts = self.dof_counts_by_id[embodiment_ids]

        src_key_padding_mask = (torch.arange(largest_dof_count_this_batch, device=obs.device)+1).unsqueeze(0).repeat(batch_size, 1) > dof_counts.unsqueeze(1) # (batch_size, largest_dof_count_this_batch), contains True where pad token locations are for each entry in batch; we need this since there will be different embodiments that have different dof counts and we only want to attend to non-padding tokens

        # Global observation
        global_obs = obs[:, :self.global_obs_size] # dim: (batch_size, global_observation_size)

        # Local tokens
        local_tokens = obs[:, self.global_obs_size:] # dim: (batch_size, max # tokens * token_size)
        local_tokens = local_tokens.reshape((batch_size, largest_dof_count_this_batch, self.local_obs_size)) # dim: (batch_size, #tokens (joint count), local token size) aka (batch, seq, feature)
        local_tokens = local_tokens.permute(1, 0, 2) # dim: (#tokens (joint count), batch_size, local tokens size) aka (seq, batch, feature)

        # apply embeddings to required properties
        if len(self.obs_property_embeddings) > 0:
            embedded_property_lst = []
            for i, property_embedding in enumerate(self.obs_property_embeddings):
                # note we make the assumption that the properties that require embeddings are the last properties in local_tokens and that they are in the same ordered as defined by `embedding_properties`
                property_values = local_tokens[:, :, -(len(self.obs_property_embeddings) - i)].to(torch.int) # (tokens, batch_size)

                # set all padding locations to -1
                padding_locs = src_key_padding_mask.permute(1, 0) # (tokens, batch)
                property_values[padding_locs] = -1

                embedded_property = property_embedding(property_values + 1) # +1 since invalid padding value is -1, so need to shift up by 1; (tokens, batch, link_id_embed_dim)
                embedded_property_lst.append(embedded_property)

            local_tokens_without_embedded_properties = local_tokens[:, :, :-len(self.obs_property_embeddings)]
            embedded_property_cat = torch.cat(embedded_property_lst, dim=2)
            local_tokens = torch.cat((local_tokens_without_embedded_properties, embedded_property_cat), dim=2) # (seq, batch, feature)

        # Concat local and global information
        global_tokens = global_obs.unsqueeze(0).repeat(largest_dof_count_this_batch, 1, 1) # dim: (num_joints aka #tokens, batch_size, global_observation_size)
        tokens = torch.cat((global_tokens, local_tokens), dim=2) # dim: (joint_count, batch_size, token_feature_dim)
        assert tokens.size(2) == self.token_feature_dim

        # Layer norm over feature dim since values in token could be on very different scales
        if self.enable_input_layer_norm:
            tokens = self.input_layer_norm(tokens) # (seq, batch, token_feature_dim)

        """Pass tokens through the network"""
        tokens = self.token_embedding(tokens) # dim: (joint_count, batch_size, self.token_embed_dim)
        if self.positional_encoding_type == 'centrality':
            tokens = self.positional_encoding(tokens, embodiment_ids) # dim: (joint_count, batch_size, self.token_embed_dim)
        else:
            tokens = self.positional_encoding(tokens) # dim: (joint_count, batch_size, self.token_embed_dim)
        tokens = self.positional_encoding_dropout(tokens) # dim: (joint_count, batch_size, self.token_embed_dim)
        
        encoder_kwargs = {
            'src': tokens,
            'src_key_padding_mask': src_key_padding_mask
        }
        if self.embodiment_encoding_type == 'graphormer':
            encoder_kwargs['embodiment_ids'] = embodiment_ids

        tokens = self.encoder(**encoder_kwargs) # dim: (joint_count, batch_size, self.token_embed_dim)
        
        """Decode head outputs from output tokens"""
        results_by_head = {}
        head_names_to_use = head_names if head_names is not None else self.head_configs.keys()
        for head_name in head_names_to_use:
            head_config = self.head_configs[head_name]
            head_module = self.head_modules[head_name]

            head_output = head_module(tokens) # dim: (joint_count, batch_size, output_dim)
            head_output = head_output.permute(1, 0, 2) # dim: (batch_size, joint_count, output_dim)
            if head_config['squeeze_output_dim'] and head_output.size(2) == 1:
                head_output = head_output.squeeze(2) # dim: (batch_size, joint_count)

            # set invalid outputs (dof counts smaller than max dof count) to infinity for sanity checks
            head_output[torch.where(src_key_padding_mask)] = torch.inf

            results_by_head[head_name] = head_output

        return results_by_head

# Adapted from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class FixedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len):
        super().__init__()
        self.seq_len = max_seq_len

        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: Tensor, shape (seq_len, batch_size, d_model)
        """
        return x + self.pe[:x.size(0)]

# Adapted from https://github.com/agrimgupta92/metamorph/blob/d49826475d4d22d8e63df694785833dc127c4e8d/metamorph/algos/ppo/model.py#L111C1-L111C1
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.seq_len = max_seq_len
        self.pe = nn.Parameter(torch.randn(max_seq_len, 1, d_model))

    def forward(self, x):
        """
        x: Tensor, shape (seq_len, batch_size, d_model)
        """
        return x + self.pe[:x.size(0)]

class GraphormerPositionalEmbedding(nn.Module):
    """
    Each token is embedded based on the connectivity degree of the associated joint. This embedding is invariant to the token ordering.
    For example, if the joint is connected to two other joints, then we add the embedding corresponding to degree==2 to the associated token.
    Based on description in https://arxiv.org/pdf/2106.05234.pdf
    """

    def __init__(self, d_model: int, max_seq_len: int, max_degree_count: int, degree_counts_by_id: List[Tensor]):
        """
        max_seq_len - only needs to be max sequence length of the embodiment you plan to test on. The parameters learned by this model are computed based on degree count for each token and are independent of max sequence length.
        max_degree_count - should be the max degree count across all embodiment configurations you ever expect to run on with this model
        """
        super().__init__()

        num_embodiments = len(degree_counts_by_id)

        degree_counts_padded = torch.zeros((num_embodiments, max_seq_len), dtype=torch.long)
        for i, degree_counts in enumerate(degree_counts_by_id):
            degree_counts_padded[i, :len(degree_counts)] = degree_counts

        self.register_buffer('degree_counts_by_id', degree_counts_padded, persistent=False)
        self.embedding = nn.Embedding(max_degree_count + 1, d_model) # + 1 to account for degree count of 0
    
    def forward(self, tokens: Tensor, embodiment_ids: Tensor):
        """
        tokens: Tensor, shape (seq_len, batch_size, d_model)
        embodiment_ids: Tensor, shape (batch_size, )
        """
        seq_len, batch_size, d_model = tokens.shape
        degree_counts = self.degree_counts_by_id[embodiment_ids, :seq_len] # (batch_size, seq_len)
        degree_embeddings = self.embedding(degree_counts) # (batch_size, seq_len, d_model)
        pe = degree_embeddings.permute(1, 0, 2) # (seq_len, batch_size, d_model)

        return tokens + pe
