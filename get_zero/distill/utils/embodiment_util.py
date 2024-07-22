import os
from datetime import datetime
from typing import Tuple, List, Dict, Optional
import torch
from torch import Tensor
import xml.etree.ElementTree as ET
from treelib import Tree
import tempfile
from yourdfpy import URDF
import numpy as np
import trimesh.transformations as tra
from collections import OrderedDict
from multiprocessing.pool import AsyncResult
from omegaconf import DictConfig
from sknetwork.path import get_distances
from scipy.sparse.csgraph import shortest_path

# TODO: a lot of these functions can be rewritten using `urdfpy` or `yourdfpy` which would be much simpler

class EmbodimentProperties:
    def __init__(self, name: str, asset_file_contents: str, joint_name_to_joint_i: Optional[Dict[str, int]]=None, metadata: Optional[Dict]=None, skip_geo_load=False):
        self.name = name
        self.asset_file_contents = asset_file_contents
        self.joint_name_to_joint_i = joint_name_to_joint_i
        self.metadata = metadata

        # properties that can be inferred from the provided params
        self.dof_count = get_dof_count_from_asset_file(asset_file_contents)
        adjacency_marix_returns = get_adjacency_matrix_from_asset_file(asset_file_contents, joint_name_to_joint_i)
        if joint_name_to_joint_i is None:
            self.adjacency_matrix = adjacency_marix_returns[0]
            self.joint_name_to_joint_i = adjacency_marix_returns[1]
        else:
            self.adjacency_matrix = adjacency_marix_returns

        self.joint_properties = get_geometry_properties(asset_file_contents, self.joint_name_to_joint_i) if not skip_geo_load else None
    
    def to_dict(self):
        return {
            'name': self.name,
            'asset_file_contents': self.asset_file_contents,
            'joint_name_to_joint_i': self.joint_name_to_joint_i,
            'metadata': self.metadata
        }

    @staticmethod
    def from_dict(dict):
        return EmbodimentProperties(
            name=dict['name'],
            asset_file_contents=dict['asset_file_contents'],
            joint_name_to_joint_i=dict['joint_name_to_joint_i'],
            metadata=dict.get('metadata', None)
        )
    
    def get_connectivity_tree(self, root_link_name):
        return get_embodiment_tree_str(self.asset_file_contents, root_link_name)
    
    def compute_parent_and_child_matrix(self) -> Tuple[Tensor, Tensor]:
        """
        Comptues a parent matrix `P`, which is `NxN` where `N=dof_count` and entry `P[i, j]` is:
        - distance (# of joints away) from joint i to j if j is a parent of i.
        - 0 otherwise

        Comptues a child matrix `C`, which is `NxN` where `N=dof_count` and entry `P[i, j]` is:
        - distance (# of joints away) from joint i to j if j is a child of i.
        - 0 otherwise

        Returns (`P`, `C`)
        """
        assert get_asset_format(self.asset_file_contents) == 'urdf'
        urdf = get_urdf(self.asset_file_contents)
        
        P = torch.zeros((self.dof_count, self.dof_count), dtype=torch.int)
        C = torch.zeros((self.dof_count, self.dof_count), dtype=torch.int)
        for joint_name, i in self.joint_name_to_joint_i.items():
            parent_joints = get_joint_family(urdf, joint_name, 'parents')
            child_joints = get_joint_family(urdf, joint_name, 'children')
            
            for (parent_joint_name, distance) in parent_joints:
                j = self.joint_name_to_joint_i[parent_joint_name]
                P[i, j] = distance

            for (child_joint_name, distance) in child_joints:
                j = self.joint_name_to_joint_i[child_joint_name]
                C[i, j] = distance

        return P, C
    
    def compute_edge_matrix(self, link_name_to_id: Dict[str, int]) -> Tensor:
        """
        entry i,j:
        -2 along diagonal for self (i == j)
        -1 for no link (i and j do not have direct connection)
        link ID (starting from 1 and increasing) (i and j have direct link between them)

        returns matrix of shape (dof_count, dof_count)
        """
        assert get_asset_format(self.asset_file_contents) == 'urdf'
        urdf = get_urdf(self.asset_file_contents)

        E = torch.full((self.dof_count, self.dof_count, ), -1, dtype=torch.int)
        for joint_name_i, i in self.joint_name_to_joint_i.items():
            E[i, i] = -2
            child_joint_name = self.joint_properties['child_joint_names'][i]
            child_link_name = self.joint_properties['child_link_names'][i]

            if child_joint_name is not None:
                # this means there is a directed edge joint_name_i---(child_link_name)--->child_joint_name
                j = self.joint_name_to_joint_i[child_joint_name]
                e = link_name_to_id[child_link_name]
                E[i, j] = E[j, i] = e

            parent_joint_name = self.joint_properties['parent_joint_names'][i]
            parent_link_name = self.joint_properties['parent_link_names'][i]

            if parent_joint_name is None:
                # there is a parent link, but no parent joint. Add in the edge between the current joint and each child joint (that's not the current joint) of this parent link
                for joint_name_j, j in self.joint_name_to_joint_i.items():
                    if i == j:
                        continue
                    parent_link_name_j = self.joint_properties['parent_link_names'][j]
                    if parent_link_name_j == parent_link_name:
                        e = link_name_to_id[parent_link_name]
                        E[i, j] = E[j, i] = e

            # TODO: add equivalent child for shared child link

        return E

class StateLogger:
    """
    Handles logging of state (observations and actions) during test time. Observations are notably kept in a "raw" format in order to experiment with different tokenization methods later in the case of training the embodiment transformer.
    """
    def __init__(self):
        self.obs_log = []
        self.actions_log = []
        self.cur_log_state_obs = None
        self.cur_log_state_actions = None

    def record_observation(self, global_obs_entries: List[Tensor], local_obs_entries: List[Tensor], at_reset_buf: Tensor):
        # unsqueeze to add time dimension at start
        self.cur_log_state_obs = {
            'global': [t.to('cpu', copy=True).unsqueeze(0) for t in global_obs_entries],
            'local': [t.to('cpu', copy=True).unsqueeze(0) for t in local_obs_entries],
            'resets': at_reset_buf.to('cpu', copy=True).unsqueeze(0)
        }

    def record_actions(self, actions):
        self.cur_log_state_actions = actions.to('cpu', copy=True)

    def commit_state(self):
        self.obs_log.append(self.cur_log_state_obs)
        self.actions_log.append(self.cur_log_state_actions)

    def save_state_logs(self, dir: str, run_name: str, fname_suffix: str, embodiment_properties: EmbodimentProperties): # TODO: move this embodiment_properties param to __init__
        os.makedirs(dir, exist_ok=True)
        fname_suffix = '' if len(fname_suffix) == 0 else f'__{fname_suffix}'

        fname = os.path.join(dir, f"statelog__time-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')}__runname-{run_name}{fname_suffix}.pt")

        obs_joined = {k: None for k in self.obs_log[0]}
        for new_obs in self.obs_log:
            for key in obs_joined:
                if obs_joined[key] is None:
                    obs_joined[key] = new_obs[key]
                else:
                    if key == 'resets':
                        # resets is single tensor not list of tensors
                        obs_joined[key] = torch.cat((obs_joined[key], new_obs[key]))
                    else:
                        # other entries are lists of tensors (global and local)
                        for obs_i in range(len(new_obs[key])):
                            obs_joined[key][obs_i] = torch.cat((obs_joined[key][obs_i], new_obs[key][obs_i]))
        
        actions_tensor = torch.stack(self.actions_log)
        torch.save({'obs': obs_joined,
                    'actions': actions_tensor,
                    'embodiment_properties': embodiment_properties.to_dict()
                   }, fname)
        print(f'Saved tokenized observation/action/embodiment to {fname}')

class ObservationTokenizer: # TODO: rename to EmbodimentTokenizer
    """
    Provides logic needed to tokenize the observation. Specifically, it helps compute observations that are tokenzied according to the parameters of the specific embodiment.

    The tokenization observations for local and global obs both consist of a fixed component that doesn't vary over time (like static embodiment properties) and a variable component that changes over time (like joint angles).
    Local refers to properties that are specific to each DoF and global refers to properties that are shared across all DoF.
    """
    
    def __init__(self, tokenization_cfg: Dict, embodiment_properties: EmbodimentProperties, device, num_envs: int):
        self.tokenization_cfg = tokenization_cfg
        self.include_variable_global_obs = tokenization_cfg['includeVariableGlobalObs']
        self.include_variable_local_obs = tokenization_cfg['includeVariableLocalObs']
        self.variable_global_obs_size = tokenization_cfg['variableGlobalObsSize'] if self.include_variable_global_obs else 0
        self.variable_local_obs_size = tokenization_cfg['variableLocalObsSize'] if self.include_variable_local_obs else 0
        self.obs_history_size = tokenization_cfg['obsHistorySize']
        assert self.obs_history_size > 0 or (self.variable_global_obs_size == 0 and self.variable_local_obs_size == 0), 'need at least 1 observation step to be put in the tokenized observation unless there is no variable observation'
        self.flush_variable_global_obs_on_reset = tokenization_cfg['flushVariableGlobalObsOnReset']
        self.flush_variable_local_obs_on_reset = tokenization_cfg['flushVariableLocalObsOnReset']
        self.embodiment_properties = embodiment_properties
        self.dof_count = embodiment_properties.dof_count
        self.device = device
        self.num_envs = num_envs
        
        self._initialize_fixed_observations()

        self.global_obs_size = self.fixed_global_obs_size + self.variable_global_obs_size * self.obs_history_size
        self.local_obs_size = self.fixed_local_obs_size + self.variable_local_obs_size * self.obs_history_size
        self.tokenized_obs_size = self.global_obs_size + self.local_obs_size * self.dof_count

        # Obsevation buffer contained local and global versions of both variable observations (including history) and fixed observations
        self.tokenized_obs_buf = torch.zeros((self.num_envs, self.tokenized_obs_size), device=self.device)

        # History buffer that contains only the variable observations
        self.variable_global_obs_history_buf = torch.zeros((
            self.num_envs, self.obs_history_size, self.variable_global_obs_size
        ), device=self.device)
        self.variable_local_obs_history_buf = torch.zeros((
            self.num_envs, self.obs_history_size, self.dof_count, self.variable_local_obs_size
        ), device=self.device)

    def _initialize_fixed_observations(self):
        """Local and global fixed obs"""
        # dict maps from key from `get_geometry_properties` -> tuple: (yaml key name, True if for local obs and False if for global obs)
        joint_property_name_to_yaml_property = OrderedDict([
            ('joint_axis', ('includeJointAxis', True)),
            ('joint_origin', ('includeJointOrigin', True)),
            ('joint_rot', ('includeJointRot', True)),
            ('joint_angle_limits', ('includeJointAngleLimits', True)),
            ('parent_link_origin', ('includeParentLinkPose', True)),
            ('parent_link_rot', ('includeParentLinkPose', True)),
            ('child_link_origin', ('includeChildLinkPose', True)),
            ('child_link_rot', ('includeChildLinkPose', True)),
            ('parent_joint_axis', ('includeParentJointAxis', True)),
            ('parent_joint_origin', ('includeParentJointPose', True)),
            ('parent_joint_rot', ('includeParentJointPose', True)),
            ('child_joint_axis', ('includeChildJointAxis', True)),
            ('child_joint_origin', ('includeChildJointPose', True)),
            ('child_joint_rot', ('includeChildJointPose', True)),
            ('num_serial_chains', ('includeNumSerialChains', False)),
            ('degree_count', ('includeDegreeCount', True)), # needs to be at end; see below
            ('parent_count', ('includeParentCount', True)), # needs to be at end; see below
            ('child_count', ('includeChildCount', True)) # needs to be at end; see below
            # `includeChildLinkId` and `includeChildLinkLength` are special cases handled below
        ])

        # note that it's critical that 'degree_count', 'parent_count', 'child_count' and child link ID (in this order) are the the very end of the fixed local obs so that the embodiment transformer model can properly convert the integer values into embedding values

        # add requested embodiment properties
        fixed_local_properties = []
        fixed_global_properties = []

        if self.tokenization_cfg['includeDebugBlankLocalObs']:
            fixed_local_properties.append(np.zeros((self.dof_count, 1)))
        if self.tokenization_cfg['includeDebugBlankGlobalObs']:
            fixed_global_properties.append(np.zeros((1,)))

        if self.tokenization_cfg['enableGeometryEncoding']:
            # child link length is special case
            if self.tokenization_cfg['includeChildLinkLength']:
                link_name_to_length = self.tokenization_cfg['linkNameToLength']
                child_link_names = self.embodiment_properties.joint_properties['child_link_names']
                child_link_lengths = np.array([link_name_to_length[link_name] for link_name in child_link_names]).reshape((self.dof_count, 1))
                fixed_local_properties.append(child_link_lengths)

            # most properties are handled generically this way by pulling from joint_properties from the EmbodimentProperties
            for joint_property_name, (yaml_property_name, is_local) in joint_property_name_to_yaml_property.items():
                if self.tokenization_cfg[yaml_property_name]:
                    prop_value = self.embodiment_properties.joint_properties[joint_property_name]
                    if is_local:
                        fixed_local_properties.append(prop_value)
                    else:
                        fixed_global_properties.append(prop_value)

        # child link ID is special case
        if self.tokenization_cfg['enableGeometryEncoding'] and self.tokenization_cfg['includeChildLinkId']:
            # note EmbodimentTransformer assumes that the child link ID, if present, is the last entry in the local observation, so this must be done last
            link_name_to_id = self.tokenization_cfg['linkNameToId']
            child_link_names = self.embodiment_properties.joint_properties['child_link_names']
            child_link_ids = np.array([link_name_to_id[link_name] for link_name in child_link_names]).reshape((self.dof_count, 1))
            fixed_local_properties.append(child_link_ids)
        
        # Concatentate information across all local properties
        self.fixed_local_obs_size = sum([prop.shape[1] for prop in fixed_local_properties]) # prop has shape (dof_count, feature_size)
        self.fixed_local_obs = torch.zeros((self.dof_count, self.fixed_local_obs_size), device=self.device)
        obs_i = 0
        for local_obs_entry in fixed_local_properties:
            local_obs_entry = torch.tensor(local_obs_entry, device=self.device) # (dof_count, feature_dim)
            feature_dim = local_obs_entry.size(1)
            self.fixed_local_obs[:, obs_i:obs_i+feature_dim] = local_obs_entry; obs_i += feature_dim
        assert obs_i == self.fixed_local_obs_size, 'provided fixed local observations do not match expected size'
        self.fixed_local_obs = self.fixed_local_obs.unsqueeze(0).expand((self.num_envs, -1, -1)) # (num_envs, dof_count, fixed_local_obs_size)

        # Concatentate information across all global properties
        self.fixed_global_obs_size = sum([prop.shape[0] for prop in fixed_global_properties]) # prop has shape (feature_size,)
        self.fixed_global_obs = torch.zeros(self.fixed_global_obs_size, device=self.device)
        obs_i = 0
        for global_obs_entry in fixed_global_properties:
            global_obs_entry = torch.tensor(global_obs_entry, device=self.device) # (feature_dim,)
            feature_dim = global_obs_entry.size(0)
            self.fixed_global_obs[obs_i:obs_i+feature_dim] = global_obs_entry; obs_i += feature_dim
        assert obs_i == self.fixed_global_obs_size, 'provided fixed global observations do not match expected size'
        self.fixed_global_obs = self.fixed_global_obs.unsqueeze(0).expand((self.num_envs, -1)) # (num_envs, fixed_global_obs_size)

        # Sanity check if fixed obs sizes are provided
        if 'fixedGlobalObsSize' in self.tokenization_cfg:
            assert self.tokenization_cfg['fixedGlobalObsSize'] == self.fixed_global_obs_size
        if 'fixedLocalObsSize' in self.tokenization_cfg:
            assert self.tokenization_cfg['fixedLocalObsSize'] == self.fixed_local_obs_size

    def build_tokenized_observation(self, variable_global_obs_entries: List[Tensor], variable_local_obs_entries: List[Tensor], at_reset_buf: Tensor):
        """
        Formats local and global information into tokenized observation.

        One fundamental assumption here is that the global information does not depend on the degree of freedom of the specific robot and that the local observations are all based on the number of degrees of freedom.

        Observation layout sent to policy (given N = max_num_dof; this is tokenized format, where tokens are ordered sequentially one after another):
        variable global observation - variable_global_obs_size * obs_history_size
        fixed global observation - fixed_global_obs_size
        variable local observation for dof 0 - variable_local_obs_size * obs_history_size
        fixed local observation for dof 0 - fixed_local_obs_size * dof_count
        ... variable local for dof 1
        ... fixed local for dof 2
        ...

        ARGS:
        variable_global_obs_entries: List containing Tensor of shape (num_envs, *) such that sum of * across all entries in list equals variable_global_obs_size and where each entry is a separate attribute of the global observation. If * is omitted, it is assumed * is 1.

        variable_local_obs_entries: List containing Tensor of shape (num_envs, num_dof, *) such that sum of * across entries across all Tensors in the list equals variable_local_obs_size. If * dim is ommitted, it is assumed that * is 1

        at_rest_buf: Bool Tensor of shape (num_envs,) indicating which environments are going to be reset.
        """
        at_reset_env_ids = at_reset_buf.nonzero(as_tuple=False).squeeze(-1)

        # unset global and/or local obs if they shouldn't be included
        if not self.include_variable_global_obs:
            variable_global_obs_entries = []
        if not self.include_variable_local_obs:
            variable_local_obs_entries = []

        """Compute current variable observations"""
        # Compute current variable global observation
        cur_variable_global_obs = torch.zeros((self.num_envs, self.variable_global_obs_size), device=self.device)
        obs_i = 0
        for global_obs_entry in variable_global_obs_entries:
            if global_obs_entry.dim() == 1:
                global_obs_entry = global_obs_entry.unsqueeze(1)
            feature_dim = global_obs_entry.size(1)
            cur_variable_global_obs[:, obs_i:obs_i+feature_dim] = global_obs_entry; obs_i += feature_dim
        assert obs_i == self.variable_global_obs_size, 'provided variable global observations do not match expected size'
        
        # Compute current variable local observation
        cur_variable_local_obs = torch.zeros((self.num_envs, self.dof_count, self.variable_local_obs_size), device=self.device)
        obs_i = 0
        for local_obs_entry in variable_local_obs_entries:
            if local_obs_entry.dim() == 2: # in this case feature dim was ommitted
                local_obs_entry = local_obs_entry.unsqueeze(2)
            feature_dim = local_obs_entry.size(2)
            cur_variable_local_obs[:, :, obs_i:obs_i+feature_dim] = local_obs_entry; obs_i += feature_dim
        assert obs_i == self.variable_local_obs_size, 'provided variable local observations do not match expected size'

        """Add current variable observations to the history buffer"""
        # Add variable global obs
        cur_variable_global_obs = cur_variable_global_obs.unsqueeze(1) # add time dimension; shape (num_envs, time, variable_global_obs_size)
        prev_global_obs_history_buf = self.variable_global_obs_history_buf[:, 1:] # remove oldest entry in history buffer
        self.variable_global_obs_history_buf[:] = torch.cat([prev_global_obs_history_buf, cur_variable_global_obs], dim=1) # store the newest entry in the history buffer

        # Add variable local obs
        cur_variable_local_obs = cur_variable_local_obs.unsqueeze(1) # add time dimension; shape (num_envs, time, dof_count, variable_local_obs_size)
        prev_local_obs_history_buf = self.variable_local_obs_history_buf[:, 1:] # remove oldest entry in history buffer
        self.variable_local_obs_history_buf[:] = torch.cat([prev_local_obs_history_buf, cur_variable_local_obs], dim=1) # store the newest entry in the history buffer

        """Flush history buffer on reset"""
        # This entails potentially resetting local and global observations for the older entries of the history buffer. This rewritting of history makes sense to do in order to prevent the observation from containing state information that is very different than it currently is (before vs after reset).
        if self.flush_variable_global_obs_on_reset:
            self.variable_global_obs_history_buf[at_reset_env_ids] = cur_variable_global_obs[at_reset_env_ids] # use global observation from current state to overwrite history; this broadcasts across the time dimension
        if self.flush_variable_local_obs_on_reset:
            self.variable_local_obs_history_buf[at_reset_env_ids] = cur_variable_local_obs[at_reset_env_ids] # use local observation from current state to overwrite history; this broadcasts across the time dimension

        """Add in fixed local and global observations"""
        # Add fixed global observations
        variable_global_obs = self.variable_global_obs_history_buf.permute(0, 2, 1) # (num_envs, variable_global_obs_size, obs_history_size)
        variable_global_obs = variable_global_obs.reshape(self.num_envs, -1) # (num_envs, obs_history_size*variable_global_obs_size); corresponding feature entries from all timesteps will be put sequentially next to each other
        all_global_obs = torch.cat([variable_global_obs, self.fixed_global_obs], dim=1) # (num_envs, obs_history_size*variable_global_obs_size + fixed_global_obs_size); fixed_global_obs has shape (num_envs, fixed_global_obs_size)

        # Add fixed local obs
        variable_local_obs = self.variable_local_obs_history_buf # (num_envs, obs_history_size, dof_count, variable_local_obs_size)
        variable_local_obs = self.variable_local_obs_history_buf.permute(0, 2, 3, 1) # (num_envs, dof_count, variable_local_obs_size, obs_history_size)
        variable_local_obs = torch.flatten(variable_local_obs, start_dim=2, end_dim=3) # (num_envs, dof_count, variable_local_obs_size*obs_history_size); corresponding feature entries from all timesteps will be put sequentially next to each other
        all_local_obs = torch.cat([variable_local_obs, self.fixed_local_obs], dim=2) # (num_envs, dof_count, variable_local_obs_size*obs_history_size+fixed_local_obs_size); fixed_local_obs has shape (num_envs, dof_count, fixed_local_obs_size)
        all_local_obs = all_local_obs.reshape(self.num_envs, -1) # (num_envs, (variable_local_obs_size*obs_history_size+fixed_local_obs_size)*dof_count); ordering: all features from one DoF, then all features from next DoF, etc. (this is required for the embodiment transformer to separate observations by DoF)
        
        """Combine global and local information into the observation buffer"""
        self.tokenized_obs_buf[:] = torch.cat([all_global_obs, all_local_obs], dim=1) # (num_envs, tokenized_observation_size)
        return self.tokenized_obs_buf

def get_dof_count_from_asset_file(asset_file_contents):
    file_format = get_asset_format(asset_file_contents)

    if file_format == 'urdf':
        dof_count = 0
        root = ET.fromstring(asset_file_contents)
        for child_xml in root:
            if child_xml.tag == 'joint':
                dof_count += 1
        return dof_count
    else:
        raise NotImplementedError


def get_adjacency_matrix_from_asset_file(asset_file_contents, joint_name_to_joint_i=None) -> Tuple[Tensor, Optional[Dict[str, int]]]:
    """
    Returns an adjacency matrix for the given asset.
    `asset_file_contents` is the file contents of the asset file (either in URDF or MCJF format)

    `joint_name_to_joint_i` maps from joint name in URDF file to the dof_index for the adjacency matrix. If not provided, then the DoF ordering will be based on the alphabetical ordering of the joint names in the URDF file in ascending order.

    If `joint_name_to_joint_i` is None, then the computed one is returned in addition to the adjacency matrix
    """
    file_format = get_asset_format(asset_file_contents)
    if file_format == 'urdf':
        return get_adjacency_matrix_from_urdf(asset_file_contents, joint_name_to_joint_i)
    else:
        raise NotImplementedError
    
def get_urdf_connectivity(urdf):
    root = ET.fromstring(urdf)
    joint_to_child_link = {}
    joint_to_parent_link = {}
    parent_link_to_child_joints = {}

    for child_xml in root:
        if child_xml.tag == 'joint':
            joint_name = child_xml.attrib["name"]
            parent_link_name = None
            child_link_name = None
            for joint_child in child_xml:
                if joint_child.tag == 'parent':
                    parent_link_name = joint_child.attrib['link']
                elif joint_child.tag == 'child':
                    child_link_name = joint_child.attrib['link']

            assert parent_link_name and child_link_name
            joint_to_parent_link[joint_name] = parent_link_name
            if parent_link_name not in parent_link_to_child_joints:
                parent_link_to_child_joints[parent_link_name] = []
            if child_link_name not in parent_link_to_child_joints:
                parent_link_to_child_joints[child_link_name] = []
            parent_link_to_child_joints[parent_link_name].append(joint_name)
            joint_to_child_link[joint_name] = child_link_name
    
    return joint_to_child_link, joint_to_parent_link, parent_link_to_child_joints

def get_adjacency_matrix_from_urdf(urdf, joint_name_to_joint_i=None) -> Tuple[Tensor, Optional[Dict[str, int]]]:
    """Extract connectivity information from URDF"""
    joint_to_child_link, joint_to_parent_link, parent_link_to_child_joints = get_urdf_connectivity(urdf)

    """create joint indices in alphabetical order if dof ordering not provided"""
    return_joint_name_to_joint_i = joint_name_to_joint_i is None
    if joint_name_to_joint_i is None:
        joint_name_to_joint_i = {}
        joint_names = list(joint_to_parent_link.keys())

        try:
            # try to sort by int if joint names are all ints
            joint_names = [int(x) for x in joint_names]
            joint_names.sort()
            joint_names = [str(x) for x in joint_names]
        except:
            # otherwise sort by string name
            joint_names.sort()
            pass

        for i, joint_name in enumerate(joint_names):
            joint_name_to_joint_i[joint_name] = i
    joint_i_to_joint_name = {v: k for k, v in joint_name_to_joint_i.items()} # reverse map since bidirectional

    """Build adjacency matrix"""
    dof_count = len(joint_name_to_joint_i.keys())
    A = torch.zeros((dof_count, dof_count), dtype=torch.int)
    for i in range(dof_count):
        joint_name = joint_i_to_joint_name[i]
        child_link_name = joint_to_child_link[joint_name]
        parent_link_name = joint_to_parent_link[joint_name]
        # we know that joint i is connected to the child joints of its parent link
        for child_joint in parent_link_to_child_joints[parent_link_name]:
            j = joint_name_to_joint_i[child_joint]
            if i == j: # need to skip self
                continue
            A[i, j] = 1 # only need to assign one way since other direction will be handled one that index is reached

        # we know that joint i is connected to the child joints of its child link (and vice-versa)
        for child_joint in parent_link_to_child_joints[child_link_name]:
            j = joint_name_to_joint_i[child_joint]
            A[i, j] = 1
            A[j, i] = 1

    if return_joint_name_to_joint_i:
        return A, joint_name_to_joint_i
    else:
        return A

def get_asset_format(asset_file_contents):
    """
    Determines if the given file contents correspond to a URDF file or a MCJF file.
    Returns either 'urdf' or 'mjcf'.
    Raises NotImplementedError if couldn't determine the file type
    """
    root = ET.fromstring(asset_file_contents)
    if root.tag == 'robot':
        return 'urdf'
    elif root.tag == 'mujoco':
        return 'mcjf'
    else:
        raise NotImplementedError


def get_embodiment_tree_str(asset_file_contents, root_link_name):
    """Generates a string representation of a tree that represents the embodiment's connectivity. The tree starts with `root_link_name` which must match a link name in the asset file."""
    file_format = get_asset_format(asset_file_contents)

    if file_format == 'urdf':
        joint_to_child_link, _, parent_link_to_child_joints = get_urdf_connectivity(asset_file_contents)

        def build_tree(tree, root_link_name, parent_node_name=None):
            tree.create_node(root_link_name, root_link_name, parent=parent_node_name)
            for child_joint in parent_link_to_child_joints.get(root_link_name, []):
                tree.create_node(child_joint, child_joint, parent=root_link_name)
                child_link = joint_to_child_link[child_joint]
                build_tree(tree, child_link, child_joint)

        tree = Tree()
        build_tree(tree, root_link_name)
        return str(tree)
    else:
        raise NotImplementedError
    

def get_urdf(asset_file_contents: str) -> URDF:
    with tempfile.NamedTemporaryFile() as fp:
        fp.write(asset_file_contents.encode())
        fp.seek(0)
        urdf = URDF.load(fp, load_meshes=False, build_scene_graph=True) # build_scene_graph=True so that base_link is properly set which is needed for forward kinematics
    return urdf


def get_joint_family(urdf: URDF, joint_name: str, mode: str) -> List[Tuple[str, int]]:
    """
    note this doesn't yet support branching embodiments in Y shape where multiple child joints are attach to a single parent link; currently only the first found path is used

    `mode` is either 'parents' or 'children' corresponding to which should be retrieved. The docstring here assumes `mode==parents`

    `j` is considered a parent of `joint` if `joint` has a (`parent_link_name`, `parent_joint_name`) path that leads to `j`

    Returns List of (`parent_name`, `distance`) where `distance` is the number of joints away `parent_name` is from `joint_name`. This is in ascending order of `distance`
    """
    assert mode in ['parents', 'children']

    joint = urdf.joint_map[joint_name]
    cur_link_name = joint.parent if mode == 'parents' else joint.child
    result = []

    match = True
    distance = 1
    while match:
        match = False
        for j in urdf.joint_map.values():
            j_link_name = j.child if mode == 'parents' else j.parent

            if j_link_name == cur_link_name:
                # `j` is a parent of `joint` if mode is `parents` otherwise it's a child
                cur_link_name = j.parent if mode == 'parents' else j.child
                match = True
                result.append((j.name, distance))
                distance += 1
                break
    
    return result


def get_geometry_properties(asset_file_contents, joint_name_to_joint_i):
    """Generates a dictionary of properties for each joint that include joint properties and the properties of the associated children and parent joints"""
    file_format = get_asset_format(asset_file_contents)
    dof_count = len(joint_name_to_joint_i)

    result = { # TODO: convert this to a data class
        'joint_origin': np.zeros((dof_count, 3)),
        'joint_rot': np.zeros((dof_count, 3)),
        'joint_axis': np.zeros((dof_count, 3)),
        'joint_angle_limits': np.zeros((dof_count, 2)),
        'parent_link_origin': np.zeros((dof_count, 3)),
        'parent_link_rot': np.zeros((dof_count, 3)),
        'child_link_origin': np.zeros((dof_count, 3)),
        'child_link_rot': np.zeros((dof_count, 3)),
        'parent_joint_origin': np.zeros((dof_count, 3)),
        'parent_joint_rot': np.zeros((dof_count, 3)),
        'parent_joint_axis': np.zeros((dof_count, 3)),
        'child_joint_origin': np.zeros((dof_count, 3)),
        'child_joint_rot': np.zeros((dof_count, 3)),
        'child_joint_axis': np.zeros((dof_count, 3)),
        'num_serial_chains': np.zeros((1,), dtype=np.int32), # defined as the number of joints that do not have a parent joint
        'serial_chain_length': np.zeros((dof_count, 1), dtype=np.int32),
        'degree_count': np.zeros((dof_count, 1), dtype=np.int32),
        'parent_count': np.zeros((dof_count, 1), dtype=np.int32),
        'child_count': np.zeros((dof_count, 1), dtype=np.int32),
        'parent_link_names': [None] * dof_count,
        'child_link_names': [None] * dof_count,
        'parent_joint_names': [None] * dof_count,
        'child_joint_names': [None] * dof_count
    }

    invalid_value = -1 # TODO: figure out if -1 is enough out of distribution from the values that may actually be seen in URDF; note that increasing this to something like -999 makes the initial loss very high since it scales the model outputs and empirically this caused issues while training

    if file_format == 'urdf':
        urdf = get_urdf(asset_file_contents)
        
        for joint_name, joint_i in joint_name_to_joint_i.items():
            joint = urdf.joint_map[joint_name]
            
            # joint info
            result['joint_origin'][joint_i] = tra.translation_from_matrix(joint.origin)
            result['joint_rot'][joint_i] = tra.euler_from_matrix(joint.origin)
            result['joint_axis'][joint_i] = joint.axis
            result['joint_angle_limits'][joint_i] = np.array([joint.limit.lower, joint.limit.upper])

            # parent link info
            parent_link = urdf.link_map[joint.parent]
            assert len(parent_link.visuals) == 1, 'assume 1 visual mesh per link'
            result['parent_link_origin'][joint_i] = tra.translation_from_matrix(parent_link.visuals[0].origin)
            result['parent_link_rot'][joint_i] = tra.euler_from_matrix(parent_link.visuals[0].origin)

            # child link info
            child_link = urdf.link_map[joint.child]
            assert len(child_link.visuals) == 1, 'assume 1 visual mesh per link'
            result['child_link_origin'][joint_i] = tra.translation_from_matrix(child_link.visuals[0].origin)
            result['child_link_rot'][joint_i] = tra.euler_from_matrix(child_link.visuals[0].origin)

            # parent joint info
            parent_joints = [parent_joint for parent_joint in urdf.joint_map.values() if parent_joint.child == joint.parent]
            if len(parent_joints) == 0:
                result['parent_joint_origin'][joint_i] = invalid_value
                result['parent_joint_rot'][joint_i] = invalid_value
                result['parent_joint_axis'][joint_i] = invalid_value
                result['num_serial_chains'] += 1
            elif len(parent_joints) == 1:
                parent_joint = parent_joints[0]
                result['parent_joint_origin'][joint_i] = tra.translation_from_matrix(parent_joint.origin)
                result['parent_joint_rot'][joint_i] = tra.euler_from_matrix(parent_joint.origin)
                result['parent_joint_axis'][joint_i] = parent_joint.axis
            else:
                raise NotImplementedError
            
            # child joint info
            child_joints = [child_joint for child_joint in urdf.joint_map.values() if child_joint.parent == joint.child]
            if len(child_joints) == 0:
                result['child_joint_origin'][joint_i] = invalid_value
                result['child_joint_rot'][joint_i] = invalid_value
                result['child_joint_axis'][joint_i] = invalid_value
            elif len(child_joints) == 1:
                child_joint = child_joints[0]
                result['child_joint_origin'][joint_i] = tra.translation_from_matrix(child_joint.origin)
                result['child_joint_rot'][joint_i] = tra.euler_from_matrix(child_joint.origin)
                result['child_joint_axis'][joint_i] = child_joint.axis
            else:
                raise NotImplementedError
            
            # degree count
            adjacent_joints = [j for j in urdf.joint_map.values() if (j.parent == joint.parent or j.parent == joint.child or j.child == joint.parent) and j != joint]
            result['degree_count'][joint_i] = len(adjacent_joints)

            parent_count = len(get_joint_family(urdf, joint_name, 'parents'))
            child_count = len(get_joint_family(urdf, joint_name, 'children'))
            result['parent_count'][joint_i] = parent_count
            result['child_count'][joint_i] = child_count
            result['serial_chain_length'][joint_i] = parent_count + child_count + 1 # +1 to include the current joint

            # parent/child link names
            result['parent_link_names'][joint_i] = joint.parent
            result['child_link_names'][joint_i] = joint.child

            # parent/child joint names
            result['parent_joint_names'][joint_i] = parent_joints[0].name if len(parent_joints) > 0 else None
            result['child_joint_names'][joint_i] = child_joints[0].name if len(child_joints) > 0 else None

        return result
    else:
        raise NotImplementedError
    

def compute_spd_matrix(adjacency_matrix: np.array, node_count: int):
    """
    Given adjacency matrix, computes matrix of same shape that indicates shortest path distance from each node to every other node.
    If no path exists, then the corresponding entry is -1. Adjacency matrix is MxM and node_count=N and it's required that M>=N and if M>N it implies M-N padding rows (bottom) and columns (right) of all 0 exist in adjacency matrix. SPD in padding locations will be -1.
    """
    M, _ = adjacency_matrix.shape
    N = node_count
    result = []
    for i in range(M):
        spd = get_distances(adjacency_matrix, i)

        result.append(spd)
    result = np.stack(result)

    # `get_distances` puts 0 along diagonal always, but our adjacency matrices are potentially padded with 0, so we want -1 along the diagonal in padding locations and 0 along the diagonal in non-padding locations
    for i in range(M-1, N-1, -1):
        assert result[i, i] == 0
        result[i, i] = -1

    return result

def compute_shortest_path_matrix(adjacency_matrix: np.array):
    """
    given undirected adjacency_matrix (A) of shape (M,M), returns matrix of shape (M,M,M) where entry i,j,k is equal to the kth node index along the shortest path from i to j. The path ends when the first entry at index k is -1 (all subsequent entries will also be -1).
    """
    M = adjacency_matrix.shape[0]
    A = adjacency_matrix
    R = np.full((M,M,M), -1, dtype=np.int32)

    D, P = shortest_path(A, directed=False, return_predecessors=True)
    
    for i in range(M):
        path_from_i_to_all = P[i]
        for j in range(M):
            i_to_j_path = []
            cur_step_in_path = path_from_i_to_all[j]
            while cur_step_in_path != -9999:
                i_to_j_path.insert(0, cur_step_in_path)
                cur_step_in_path = path_from_i_to_all[cur_step_in_path]
            i_to_j_path.append(j)

            for k, entry in enumerate(i_to_j_path):
                R[i,j,k] = entry
            
    return R

def compute_shortest_path_edge_matrix(adjacency_matrix: np.array, edge_matrix: np.array):
    """
    Given N,N adjacency_matrix and N,N edge matix, returns N,N,N array, R, where R[i,j,k] contains the kth edge feature along the shortest path from node i to node j. kth feature will be -1 once path has ended
    """
    N = adjacency_matrix.shape[0]
    R = np.full((N,N,N), -1, dtype=np.int32)
    SPM = compute_shortest_path_matrix(adjacency_matrix)
    for i in range(N):
        for j in range(N):
            i_to_j_path = SPM[i,j]
            for k in range(N - 1):
                node_start = i_to_j_path[k]
                node_end = i_to_j_path[k+1]
                if node_end == -1:
                    break
                e = edge_matrix[node_start,node_end]
                assert e >= 0, '-1 in edge_matrix means no link and -2 means self, which is unexpected'
                R[i,j,k] = e
    
    return R

class EmbodimentEvaluator:
    """
    An interface to obtain performance data for a distilled checkpoint. For example, in the case of an RL in sim based approach, this would entail running the distilled policy in the simulator for each embodiment and returning some metric about the performance.
    """
    
    def prepare_evaluation(self, distill_run_dir: str, task: str, embodiment_properties_by_id: List[EmbodimentProperties], model_cfg: DictConfig, tokenization_cfg: DictConfig, embodiment_name_to_splits: Dict[str, List[str]], additional_tags: List[str]):
        """
        Args:
        - `distill_run_dir`: experiment logging directory
        - `task`: name of the task
        - `embodiment_properties_by_id`: embodiment properties in order by ID
        - `model_cfg`: configuration parameters used to initialize the EmbodimentTransformer
        - `tokenization_cfg`: configuration parameters used to setup tokenization for the observation of the task
        - `embodiment_name_to_splits`: map from embodiment name to list of split names that the embodiment is a part of. Useful for logging purposes.
        - `additional_tags`: additional tags to use when logging experiments
        """
        raise NotImplementedError

    def evaluate_checkpoint(self, checkpoint: str, split_names: List[str]) -> Dict[str, AsyncResult]:
        """
        Performs evaluation of each embodiment for the given checkpoint. It's suggested to do perform these evaluations in parallel (perhaps in separate processes) for efficiency. Returns a dictionary values containing AsyncResult so that the execution does not block the main process.

        Args:
        - `checkpoint`: absolute path to checkpoint that each embodiment should be evaluated on
        - `split_names`: list of splits names (from 'train', 'val', or 'test') for which to evaluate the checkpoint on 

        Return:
        - Dict mapping from embodiment name to AsyncResult which will return [evaluation metric (float), run_id (str)] for the given checkpoint.
        """
        raise NotImplementedError

    def evaluate_baseline(self) -> Dict[str, AsyncResult]:
        """
        Performs evaluation of the baseline policy for each embodiment. Only uses embodiments for which a baseline is available and embodiments with no baseline are skipped. Returns asynchronous result.

        Return:
        - Dict mapping from embodiment name to AsyncResult which will return the evaluation metric (float) for the baseline policy.
        """
        raise NotImplementedError
    
    def mark_best_runs(self, run_ids: List[str]) -> None:
        """
        Indicates that the runs with IDs `run_ids` were the best performing checkpoint evaluation runs.
        """
        raise NotImplementedError
    
    def finish_evaluation(self):
        """
        Perform any required cleanup.
        """
        raise NotImplementedError
