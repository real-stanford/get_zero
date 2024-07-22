"""
Modified implementations of MultiHeadAttention to support custom weights in attention computation.
Adapted from implementations in PyTorch
"""

import torch
from torch import nn, Tensor
import math
from typing import List, Dict
from get_zero.distill.utils.embodiment_util import compute_spd_matrix, EmbodimentProperties, compute_shortest_path_edge_matrix
from get_zero.distill.utils.generic import pad_stack_tensors
from torch.nn import TransformerEncoderLayer, MultiheadAttention, Linear, Dropout, LayerNorm, Module
from torch.nn import functional as F
from torch.nn.modules.transformer import _get_activation_fn, _get_clones
from typing import Union, Callable, Optional, Tuple
from torch.overrides import handle_torch_function, has_torch_function
from torch.nn.functional import _in_projection_packed, _mha_shape_check, _canonical_mask, _none_or_dtype, _in_projection, pad, softmax, dropout, linear
import numpy as np

class EmbodimentTransformerEncoder(Module):
    """
    Modified version of nn.TransformerEncoder to support custom attention weights according to embodiment parameters
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None, enable_nested_tensor=True, mask_check=True):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.enable_nested_tensor = enable_nested_tensor
        self.mask_check = mask_check

    def forward(
            self,
            embodiment_ids: Tensor,
            src: Tensor,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: Optional[bool] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            embodiment_ids: ids of the embodiments for each entry in the batch
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            is_causal: If specified, applies a causal mask as mask (optional)
                and ignores attn_mask for computing scaled dot product attention.
                Default: ``False``.
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(mask),
            other_name="mask",
            target_type=src.dtype
        )

        mask = F._canonical_mask(
            mask=mask,
            mask_name="mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        output = src
        convert_to_nested = False
        first_layer = self.layers[0]
        src_key_padding_mask_for_layers = src_key_padding_mask
        why_not_sparsity_fast_path = ''
        str_first_layer = "self.layers[0]"
        if not isinstance(first_layer, torch.nn.TransformerEncoderLayer):
            why_not_sparsity_fast_path = f"{str_first_layer} was not TransformerEncoderLayer"
        elif first_layer.norm_first :
            why_not_sparsity_fast_path = f"{str_first_layer}.norm_first was True"
        elif first_layer.training:
            why_not_sparsity_fast_path = f"{str_first_layer} was in training mode"
        elif not first_layer.self_attn.batch_first:
            why_not_sparsity_fast_path = f" {str_first_layer}.self_attn.batch_first was not True"
        elif not first_layer.self_attn._qkv_same_embed_dim:
            why_not_sparsity_fast_path = f"{str_first_layer}.self_attn._qkv_same_embed_dim was not True"
        elif not first_layer.activation_relu_or_gelu:
            why_not_sparsity_fast_path = f" {str_first_layer}.activation_relu_or_gelu was not True"
        elif not (first_layer.norm1.eps == first_layer.norm2.eps) :
            why_not_sparsity_fast_path = f"{str_first_layer}.norm1.eps was not equal to {str_first_layer}.norm2.eps"
        elif not src.dim() == 3:
            why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        elif not self.enable_nested_tensor:
            why_not_sparsity_fast_path = "enable_nested_tensor was not True"
        elif src_key_padding_mask is None:
            why_not_sparsity_fast_path = "src_key_padding_mask was None"
        elif (((not hasattr(self, "mask_check")) or self.mask_check)
                and not torch._nested_tensor_from_mask_left_aligned(src, src_key_padding_mask.logical_not())):
            why_not_sparsity_fast_path = "mask_check enabled, and src and src_key_padding_mask was not left aligned"
        elif output.is_nested:
            why_not_sparsity_fast_path = "NestedTensor input is not supported"
        elif mask is not None:
            why_not_sparsity_fast_path = "src_key_padding_mask and mask were both supplied"
        elif first_layer.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = "num_head is odd"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"

        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                first_layer.self_attn.in_proj_weight,
                first_layer.self_attn.in_proj_bias,
                first_layer.self_attn.out_proj.weight,
                first_layer.self_attn.out_proj.bias,
                first_layer.norm1.weight,
                first_layer.norm1.bias,
                first_layer.norm2.weight,
                first_layer.norm2.bias,
                first_layer.linear1.weight,
                first_layer.linear1.bias,
                first_layer.linear2.weight,
                first_layer.linear2.bias,
            )

            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif not (src.is_cuda or 'cpu' in str(src.device)):
                why_not_sparsity_fast_path = "src is neither CUDA nor CPU"
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
                                              "input/output projection weights or biases requires_grad")

            if (not why_not_sparsity_fast_path) and (src_key_padding_mask is not None):
                convert_to_nested = True
                output = torch._nested_tensor_from_mask(output, src_key_padding_mask.logical_not(), mask_check=False)
                src_key_padding_mask_for_layers = None

        # Prevent type refinement
        make_causal = (is_causal is True)

        if is_causal is None:
            if mask is not None:
                sz = mask.size(0)
                causal_comparison = torch.triu(
                    torch.ones(sz, sz, device=mask.device) * float('-inf'), diagonal=1
                ).to(mask.dtype)

                if torch.equal(mask, causal_comparison):
                    make_causal = True

        is_causal = make_causal

        for mod in self.layers:
            output = mod(embodiment_ids, output, src_mask=mask, is_causal=is_causal, src_key_padding_mask=src_key_padding_mask_for_layers)

        if convert_to_nested:
            output = output.to_padded_tensor(0.)

        if self.norm is not None:
            output = self.norm(output)

        return output

class SpatialEncodingTransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, embodiment_properties_by_id: List[EmbodimentProperties], max_dof_count: int, attention_config: Dict, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        Module.__init__(self)
        self.batch_first = batch_first
        self.self_attn = CustomAttentionScoresMultiHeadAttention(attention_config['bias_before_softmax'], d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

        """
        Above is original from `TransformerEncoderLayer`, below is custom changes for spatial encoding.
        For notes on spatial encoding, see section 3.1.2 of https://arxiv.org/pdf/2106.05234.pdf
        """

        self.nhead = nhead
        self.enable_spatial_encoding = attention_config['enable_spatial_encoding']
        self.enable_parent_encoding = attention_config['enable_parent_encoding']
        self.enable_child_encoding = attention_config['enable_child_encoding']
        self.enable_edge_encoding = attention_config['enable_edge_encoding']
        self.edge_encoding_size = attention_config['edge_encoding_size']
        self.num_edge_types = len(attention_config['edge_name_to_id'])
        self.longest_edge_path = attention_config['longest_edge_path']

        """Prepare embodiment properties"""
        
        # DoF counts
        dof_counts_by_id = []
        for embodiment_properties in embodiment_properties_by_id:
            dof_counts_by_id.append(embodiment_properties.dof_count)
        dof_counts_by_id = torch.tensor(dof_counts_by_id, dtype=torch.long)
        self.register_buffer('dof_counts_by_id', dof_counts_by_id, persistent=False)

        # Adjacency matrix
        adjacency_matrices_by_id = pad_stack_tensors([ep.adjacency_matrix for ep in embodiment_properties_by_id]) # (B, max_provided_dof_count, max_provided_dof_count)

        if adjacency_matrices_by_id is not None:
            # Shortest path distance (SPD) matrices
            if self.enable_spatial_encoding:
                spd_matrices_by_id = []
                for i in range(len(adjacency_matrices_by_id)):
                    cur_adjacency_matrix = adjacency_matrices_by_id[i]
                    cur_dof_count = dof_counts_by_id[i].item()
                    spd_matrices_by_id.append(compute_spd_matrix(cur_adjacency_matrix.cpu().numpy(), cur_dof_count))
                self.register_buffer('spd_matrices_by_id', torch.from_numpy(np.stack(spd_matrices_by_id)), persistent=False) # shape: (num_embodiments, max_seq_len, max_seq_len); will have -1 in invalid locations

            # Parent and child matrices
            parent_matrices = []
            child_matrices = []
            for embodiment_properties in embodiment_properties_by_id:
                parent_mat, child_mat = embodiment_properties.compute_parent_and_child_matrix()
                parent_matrices.append(parent_mat)
                child_matrices.append(child_mat)
            
            if self.enable_parent_encoding:
                self.register_buffer('parent_matrices_by_id', pad_stack_tensors(parent_matrices, pad_value=0), persistent=False)
            if self.enable_child_encoding:
                self.register_buffer('child_matrices_by_id', pad_stack_tensors(child_matrices, pad_value=0), persistent=False)

            # Edge feature matrices
            if self.enable_edge_encoding:
                edge_matrices = []
                edge_path_length_matrices = []
                for embodiment_properties in embodiment_properties_by_id:
                    A = embodiment_properties.adjacency_matrix.numpy()
                    E = embodiment_properties.compute_edge_matrix(attention_config['edge_name_to_id'])
                    SPEM = torch.from_numpy(compute_shortest_path_edge_matrix(A, E)) # (dof_count, dof_count, dof_count)
                    SPEM = SPEM[:,:,:self.longest_edge_path] # cut unnecessary padding; (dof_count, dof_count, longest_edge_path)
                    edge_matrices.append(SPEM)
                    edge_path_length_matrix = (SPEM != -1).sum(dim=-1) # length is number of non -1 entries
                    edge_path_length_matrix += torch.eye(A.shape[0], dtype=torch.long) * -1 # length from a node from itself should be invalid value (this value has to be nonzero because we divide by it, but it otherwise doesn't matter what the value is since the embeddings at these locations will be all zeros)
                    edge_path_length_matrices.append(edge_path_length_matrix)
                self.register_buffer('edge_matrices_by_id', pad_stack_tensors(edge_matrices, pad_value=-1), persistent=False) # (num_embodiments, dof_count, dof_count, longest_edge_path)
                self.register_buffer('edge_path_length_by_id', pad_stack_tensors(edge_path_length_matrices, pad_value=-1), persistent=False) # (num_embodiments, dof_count, dof_count)
        else:
            print(f'WARNING: SpatialEncodingTransformerEncoderLayer not properly initialized since adjacency_matrices_by_id is empty, meaning you will not be able to run a forward pass')

        """Initialize layers"""
        max_node_to_node_distance = max_dof_count - 1 # decreased by 1 due to the longest possible chain length being one less than num_dof (imagine worst case where dofs are connected in a serial chain, then farthest dof are at the ends at a distance of dof_count - 1)
        if self.enable_spatial_encoding:
            # Spatial Embedding (node to node distance)
            self.spatial_embedding = nn.Embedding(max_node_to_node_distance + 1 + 1, nhead, padding_idx=0) # +1 to account for padding value of -1; +1 for distance 0 (attention between node and itself)
        if self.enable_parent_encoding:
            # Parent child Embedding (node to parent distance)
            self.parent_embedding = nn.Embedding(max_node_to_node_distance + 1, nhead, padding_idx=0) # +1 for 0, which corresponds to padding tokens, not parent tokens, and self tokens
        if self.enable_child_encoding:
            # Parent child Embedding (node to child distance)
            self.child_embedding = nn.Embedding(max_node_to_node_distance + 1, nhead, padding_idx=0) # same as for parent_embedding

        if self.enable_edge_encoding:
            # edge features are shared across all heads, but a custom weight based on distance along path is learned per head
            self.edge_embedding = nn.Embedding(self.num_edge_types + 1, self.edge_encoding_size, padding_idx=0) # additional +1 to handle -1 case which corresponds to padding and the case of no more entries in the path (which can also just be considered as padding since the value isn't used)
            self.edge_path_weight = nn.Parameter(torch.empty((nhead, self.longest_edge_path, self.edge_encoding_size))) # (nhead, longest_edge_path, edge_encoding_size); based on the distance along the path and the current head there is a separate learned edge weight encoding
            nn.init.xavier_normal_(self.edge_path_weight)

        # Embedding layer initialization
        embedding_layers = []
        embedding_layers += [self.spatial_embedding] if self.enable_spatial_encoding else []
        embedding_layers += [self.parent_embedding] if self.enable_parent_encoding else []
        embedding_layers += [self.child_embedding] if self.enable_child_encoding else []
        embedding_layers += [self.edge_embedding] if self.enable_edge_encoding else []

        for embedding_layer in embedding_layers:
            if attention_config['embedding_init']['type'] == 'zero':
                nn.init.zeros_(embedding_layer.weight)
            elif attention_config['embedding_init']['type'] == 'normal':
                nn.init.normal_(embedding_layer.weight, std=attention_config['embedding_init']['sigma'])
            else:
                raise NotImplementedError
            
            # All embedings that have padding_idx should have all zeros at padding location since weight initialization will alter padding weight which was initially set to all zeros
            if embedding_layer.padding_idx is not None:
                embedding_layer.weight.data[embedding_layer.padding_idx] = 0

    def forward(
            self,
            embodiment_ids: Tensor,
            src: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: bool = False) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            embodiment_ids: ids of each of the embodiments in the batch
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            is_causal: If specified, applies a causal mask as src_mask.
              Default: ``False``.
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype
        )

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        x = src
        if self.norm_first:
            x = x + self._sa_block(embodiment_ids, self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(embodiment_ids, x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, embodiment_ids: Tensor, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        if self.batch_first:
            B, S, D = x.shape
        else:
            S, B, D = x.shape
        
        assert embodiment_ids.size(0) == B, 'number of embodiment IDs should match batch dimension of input'


        # note we add +1 when computing embedding because each embedding starts at -1 which is the invalid entry embedding

        unique_embodiment_ids, embodiment_ids_inverse = embodiment_ids.unique(return_inverse=True)
        b = len(unique_embodiment_ids)

        attention_bias = torch.zeros((b, S, S, self.nhead), device=x.device) # (b, S, S, n_head)

        # Spatial bias (undirected)
        if self.enable_spatial_encoding:
            spd_matrices = self.spd_matrices_by_id[unique_embodiment_ids] # (b, max S, max S)
            spd_matrices = spd_matrices[:, :S, :S] # (b, S, S)
            spatial_bias = self.spatial_embedding(spd_matrices + 1) # (b, S, S, n_head)
            assert torch.all(spatial_bias.permute(0, 2, 1, 3) == spatial_bias), 'spatial bias should be symmetric'
            attention_bias += spatial_bias

        # Parent bias (directed)
        if self.enable_parent_encoding:
            parent_matrices = self.parent_matrices_by_id[unique_embodiment_ids] # (b, max S, max S)
            parent_matrices = parent_matrices[:, :S, :S] # (b, S, S)
            parent_bias = self.parent_embedding(parent_matrices) # (b, S, S, n_head)
            attention_bias += parent_bias

            with torch.inference_mode():
                assert parent_bias[:,0,0,:].abs().sum() == 0, 'across every embodiment and attention head for the (0->0 parent relationship), the parent_bias should be 0'

        # Child bias (directed)
        if self.enable_child_encoding:
            child_matrices = self.child_matrices_by_id[unique_embodiment_ids] # (b, max S, max S)
            child_matrices = child_matrices[:, :S, :S] # (b, S, S)
            child_bias = self.child_embedding(child_matrices) # (b, S, S, n_head)
            attention_bias += child_bias

            with torch.inference_mode():
                assert child_bias[:,0,0,:].abs().sum() == 0, 'across every embodiment and attention head for (0->0 child relationship), the child_bias should be 0'            

        # Edge bias (undirected, but encoding from i->j != encoding from j->i) (follows Graphormer edge encoding; see their paper for details)
        if self.enable_edge_encoding:
            spe_matrices = self.edge_matrices_by_id[unique_embodiment_ids, :S, :S] # (b, S, S, longest_edge_path) = (b, from node i, to node j, path nodes from i to j)
            path_lengths = self.edge_path_length_by_id[unique_embodiment_ids, :S, :S] # (b, S, S)
            embedded_edge_paths = self.edge_embedding(spe_matrices + 1) # (b, S, S, longest_edge_path, edge_encoding_size); add 1 to make -1 padding become 0. padding will map to all zero embedding, so it won't impact output
            embedded_edge_paths = embedded_edge_paths.permute(3,0,1,2,4) # (longest_edge_path, b, S, S, edge_encoding_size)
            embedded_edge_paths = embedded_edge_paths.unsqueeze(0).repeat(self.nhead, 1, 1, 1, 1, 1) # (nhead, longest_edge_path, b, S, S, edge_encoding_size)
            embedded_edge_paths = embedded_edge_paths.reshape(self.nhead*self.longest_edge_path, b, S, S, self.edge_encoding_size) # (nhead*longest_edge_path, b, S, S, edge_encoding_size)
            edge_path_weight = self.edge_path_weight.flatten(0,1) # (nhead*longest_edge_path, edge_encoding_size)
            edge_path_weight = edge_path_weight[:, None, None, :, None] # (nhead*longest_edge_path, 1, 1, edge_encoding_size, 1)
            embedded_edge_paths = torch.matmul(embedded_edge_paths, edge_path_weight) # (nhead*longest_edge_path, b, S, S, edge_encoding_size) multiply (nhead*longest_edge_path, 1, 1, edge_encoding_size, 1) = (nhead*longest_edge_path, b, S, S, 1)
            embedded_edge_paths = embedded_edge_paths.squeeze(-1) # (nhead*longest_edge_path, b, S, S)
            embedded_edge_paths = embedded_edge_paths.reshape(self.nhead, self.longest_edge_path, b, S, S) # (nhead, longest_edge_path, b, S, S)
            embedded_edge_paths = embedded_edge_paths.permute(2,3,4,1,0) # (b, S, S, longest_edge_path, nhead)
            embedded_edge_paths = embedded_edge_paths.sum(dim=3) # (b, S, S, nhead)
            embedded_edge_paths = embedded_edge_paths / path_lengths.unsqueeze(-1) # (b, S, S, nhead) / (b, S, S, 1) = (b, S, S, nhead)
            edge_bias = embedded_edge_paths # (b, S, S, nhead)

            with torch.inference_mode():
                assert edge_bias[:,0,0,:].abs().sum() == 0, 'across every embodiment and attention head for the node 0 to node 0 path, the edge_bias should be 0'
            
            attention_bias += edge_bias

        attention_bias = attention_bias[embodiment_ids_inverse] # (b, S, S, nhead) -> # (B, S, S, nhead)

        x = self.self_attn(attention_bias, x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

class CustomAttentionScoresMultiHeadAttention(MultiheadAttention):
    """
    Modified version of MultiheadAttention that let's us pass an additional attention score reweight that is added to the attention score matrix that is normally computed
    """

    def __init__(self, bias_before_softmax: bool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias_before_softmax = bias_before_softmax

    def forward(
            self,
            attention_reweighting: Tensor,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[Tensor] = None,
            average_attn_weights: bool = True,
            is_causal : bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
    Args:
        attention_reweighting: Tensor of shape (N, L, S, H) to add to the normally computed attention scores
        query: Query embeddings of shape :math:`(L, E_q)` for unbatched input, :math:`(L, N, E_q)` when ``batch_first=False``
            or :math:`(N, L, E_q)` when ``batch_first=True``, where :math:`L` is the target sequence length,
            :math:`N` is the batch size, and :math:`E_q` is the query embedding dimension ``embed_dim``.
            Queries are compared against key-value pairs to produce the output.
            See "Attention Is All You Need" for more details.
        key: Key embeddings of shape :math:`(S, E_k)` for unbatched input, :math:`(S, N, E_k)` when ``batch_first=False``
            or :math:`(N, S, E_k)` when ``batch_first=True``, where :math:`S` is the source sequence length,
            :math:`N` is the batch size, and :math:`E_k` is the key embedding dimension ``kdim``.
            See "Attention Is All You Need" for more details.
        value: Value embeddings of shape :math:`(S, E_v)` for unbatched input, :math:`(S, N, E_v)` when
            ``batch_first=False`` or :math:`(N, S, E_v)` when ``batch_first=True``, where :math:`S` is the source
            sequence length, :math:`N` is the batch size, and :math:`E_v` is the value embedding dimension ``vdim``.
            See "Attention Is All You Need" for more details.
        key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
            to ignore for the purpose of attention (i.e. treat as "padding"). For unbatched `query`, shape should be :math:`(S)`.
            Binary and float masks are supported.
            For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
            the purpose of attention. For a float mask, it will be directly added to the corresponding ``key`` value.
        need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
            Default: ``True``.
        attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
            :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,
            :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
            broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
            Binary and float masks are supported. For a binary mask, a ``True`` value indicates that the
            corresponding position is not allowed to attend. For a float mask, the mask values will be added to
            the attention weight.
            If both attn_mask and key_padding_mask are supplied, their types should match.
        is_causal: If specified, applies a causal mask as attention mask. Mutually exclusive with providing attn_mask.
            Default: ``False``.
        average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across
            heads. Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an
            effect when ``need_weights=True``. Default: ``True`` (i.e. average weights across heads)

    Outputs:
        - **attn_output** - Attention outputs of shape :math:`(L, E)` when input is unbatched,
          :math:`(L, N, E)` when ``batch_first=False`` or :math:`(N, L, E)` when ``batch_first=True``,
          where :math:`L` is the target sequence length, :math:`N` is the batch size, and :math:`E` is the
          embedding dimension ``embed_dim``.
        - **attn_output_weights** - Only returned when ``need_weights=True``. If ``average_attn_weights=True``,
          returns attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
          :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
          head of shape :math:`(\text{num\_heads}, L, S)` when input is unbatched or :math:`(N, \text{num\_heads}, L, S)`.

        .. note::
            `batch_first` argument is ignored for unbatched inputs.
        """
        if attn_mask is not None and is_causal:
            raise AssertionError("Only allow causal mask or attn_mask")

        is_batched = query.dim() == 3

        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype
        )

        any_nested = query.is_nested or key.is_nested or value.is_nested
        assert not any_nested, ("MultiheadAttention does not support NestedTensor outside of its fast path. ")

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = custom_attention_weights_multi_head_attention_forward(
                attention_reweighting,
                self.bias_before_softmax,
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal)
        else:
            attn_output, attn_output_weights = custom_attention_weights_multi_head_attention_forward(
                attention_reweighting,
                self.bias_before_softmax,
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal)
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights

def custom_attention_weights_multi_head_attention_forward(
    attention_reweighting: Tensor,
    bias_before_softmax,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Optional[Tensor],
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    average_attn_weights: bool = True,
    is_causal: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        is_causal: If specified, applies a causal mask as attention mask, and ignores
            attn_mask for computing scaled dot product attention.
            Default: ``False``.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
        average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across heads.
            Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an effect
            when ``need_weights=True.``. Default: True


    Shape:
        Inputs:
        - query: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, E)` or :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, E)` or :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(S)` or :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a FloatTensor is provided, it will be directly added to the value.
          If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

        Outputs:
        - attn_output: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: Only returned when ``need_weights=True``. If ``average_attn_weights=True``, returns
          attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
          :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
          head of shape :math:`(num_heads, L, S)` when input is unbatched or :math:`(N, num_heads, L, S)`.
    """
    tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias)
    if has_torch_function(tens_ops):
        return handle_torch_function(
            custom_attention_weights_multi_head_attention_forward,
            tens_ops,
            attention_reweighting,
            bias_before_softmax,
            query,
            key,
            value,
            embed_dim_to_check,
            num_heads,
            in_proj_weight,
            in_proj_bias,
            bias_k,
            bias_v,
            add_zero_attn,
            dropout_p,
            out_proj_weight,
            out_proj_bias,
            training=training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            is_causal=is_causal,
            use_separate_proj_weight=use_separate_proj_weight,
            q_proj_weight=q_proj_weight,
            k_proj_weight=k_proj_weight,
            v_proj_weight=v_proj_weight,
            static_k=static_k,
            static_v=static_v,
            average_attn_weights=average_attn_weights,
        )

    is_batched = _mha_shape_check(query, key, value, key_padding_mask, attn_mask, num_heads)

    # For unbatched input, we unsqueeze at the expected batch-dim to pretend that the input
    # is batched, run the computation and before returning squeeze the
    # batch dimension so that the output doesn't carry this temporary batch dimension.
    if not is_batched:
        # unsqueeze if the input is unbatched
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0)

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape

    key_padding_mask = _canonical_mask(
        mask=key_padding_mask,
        mask_name="key_padding_mask",
        other_type=_none_or_dtype(attn_mask),
        other_name="attn_mask",
        target_type=query.dtype
    )

    if is_causal:
        attn_mask = None

    assert embed_dim == embed_dim_to_check, \
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    if use_separate_proj_weight:
        # allow MHA to have different embedding dimensions when separate projection weights are used
        assert key.shape[:2] == value.shape[:2], \
            f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

    #
    # compute in-projection
    #
    if not use_separate_proj_weight:
        assert in_proj_weight is not None, "use_separate_proj_weight is False but in_proj_weight is None"
        q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    else:
        assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
        assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
        assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)
        q, k, v = _in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)

    # prep attention mask

    attn_mask = _canonical_mask(
        mask=attn_mask,
        mask_name="attn_mask",
        other_type=None,
        other_name="",
        target_type=q.dtype,
        check_other=False,
    )

    if attn_mask is not None:
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

    # add bias along batch dimension (currently second)
    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if static_k is None:
        k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_k.size(0) == bsz * num_heads, \
            f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
        assert static_k.size(2) == head_dim, \
            f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
        k = static_k
    if static_v is None:
        v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_v.size(0) == bsz * num_heads, \
            f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
        assert static_v.size(2) == head_dim, \
            f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
        v = static_v

    # add zero attention along batch dimension (now first)
    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    # update source sequence length after adjustments
    src_len = k.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (bsz, src_len), \
            f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
            expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        else:
            attn_mask = attn_mask + key_padding_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #

    B, Nt, E = q.shape
    q_scaled = q / math.sqrt(E)
    if attn_mask is not None:
        attn_output_weights = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))
    else:
        attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))

    # attn_output_weights shape: (batch_size*num_heads, seq_len, seq_len)
    # attention_reweighting: (batch_size, seq_len, seq_len, num_heads)
    # each head shares the same attention reweighting
    assert is_batched

    attention_reweighting = attention_reweighting.permute(0, 3, 1, 2) # (batch_size, num_heads, seq_len, seq_len)
    attention_reweighting = attention_reweighting.reshape(bsz * num_heads, src_len, src_len) # (batch_size*num_heads, seq_len, seq_len)
    assert attn_output_weights.shape == attention_reweighting.shape

    if bias_before_softmax:
        attn_output_weights = attn_output_weights + attention_reweighting # implicitly applies padding mask to the reweight since there will be -inf entries in `attn_output_weights` that won't change even if something is added to it
    attn_output_weights = softmax(attn_output_weights, dim=-1)

    if not bias_before_softmax:
        pre_softmax_attn_mask = torch.ones((B, 1, Nt), device=attention_reweighting.device)
        pre_softmax_attn_mask[attn_mask == -torch.inf] = 0
        attention_reweighting = attention_reweighting * pre_softmax_attn_mask # masked regions should also mask attention_reweight, but we set to 0 instead of infinity since we add attention_reweighting after the Softmax
        attn_output_weights = attn_output_weights + attention_reweighting

    if dropout_p > 0.0:
        attn_output_weights = dropout(attn_output_weights, p=dropout_p)
    attn_output = torch.bmm(attn_output_weights, v)

    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
    attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

    # optionally average attention weights over heads
    attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
    if average_attn_weights:
        attn_output_weights = attn_output_weights.mean(dim=1)

    if not is_batched:
        # squeeze the output if input was unbatched
        attn_output = attn_output.squeeze(1)
        attn_output_weights = attn_output_weights.squeeze(0)
    return attn_output, attn_output_weights
