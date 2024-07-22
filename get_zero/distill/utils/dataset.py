"""Loads multi-embodiment data from different embodiments into a dataloader for network training."""

from glob import glob
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from get_zero.distill.utils.embodiment_util import ObservationTokenizer, EmbodimentProperties, get_urdf
from get_zero.distill.utils.generic import assert_and_set
from tqdm import tqdm
import os
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
import yaml
from dataclasses import dataclass
from get_zero.utils.forward_kinematics import fk


def get_datasets(cfg: DictConfig):
    """
    The dataset will either consist of:
    1) observations and actions from the state logs along with potnetially self modeling information, or
    2) observations containing only fixed information with outputs being self modeling
    """
    # Get names of self modeling datasets that will be used
    self_model_metric_names = []
    if cfg.objective.include_self_modeling:
        for metric_name in cfg.objective:
            if metric_name in ['policy', 'policy_forward_kinematics', 'include_self_modeling']:
                continue
            elif cfg.objective[metric_name].loss_weight > 0:
                self_model_metric_names.append(metric_name)
    else:
        cfg.objective.policy_forward_kinematics.loss_weight = 0

    policy_loss_enabled = cfg.objective.policy.loss_weight > 0
    fk_enabled = cfg.objective.policy_forward_kinematics.loss_weight > 0 and cfg.objective.include_self_modeling

    if policy_loss_enabled or fk_enabled:
        fk_self_model_name = ['policy_forward_kinematics'] if fk_enabled else []
        policy_str = 'enabled' if policy_loss_enabled else 'disabled'
        print(f'Getting policy dataset with the following self modeling losses: {self_model_metric_names + fk_self_model_name} with policy loss {policy_str}')
        return get_policy_datasets(cfg, self_model_metric_names)
    else:
        print(f'Getting self modeling dataset with the following self modeling losses: {self_model_metric_names}')
        assert cfg.objective.include_self_modeling
        return get_self_model_dataset(cfg, self_model_metric_names)


def get_policy_datasets(cfg: DictConfig, self_model_metric_names: List[str]):
    """
    Loads train_train, train_val, val_train and val_val splits and returns separate dataloaders for each (val_train isn't really used since we aren't training on validation embodiment data, but it's there for finetuning support and since we don't need to evaluate the losses during training across the entire validation set).
    `train` refers to the set of embodiments for which we will pull training data from. Within this set of embodiments we want to compute a validation loss for a "within embodiment" performance. Validation and test are then used to determine "zero-shot" performance, but metrics and baselines will be computed for validation embodiments, whereas for test embodiments we don't assume we have demonstration data or baseline performance data.

    dataloader_kwargs are additional args to pass to the dataloaders

    Side effect: sets the following properties of `cfg.tokenization`:
     - `variableGlobalObsSize`, `variableLocalObsSize`, `fixedGlobalObsSize`, `fixedLocalObsSize`, `globalObsSize`, `localObsSize`

    """
    dataloader_device = f'cuda:{cfg.gpu}'
    policy_loss_enabled = cfg.objective.policy.loss_weight > 0
    fk_enabled = cfg.objective.policy_forward_kinematics.loss_weight > 0 and cfg.objective.include_self_modeling

    # info is shared across datasets
    embodiment_properties_by_id = []
    embodiment_name_to_id = {}
    embodiment_id_counter = 0
    embodiment_id_counter_split = 0

    def register_embodiment(embodiment_properties):
        nonlocal embodiment_id_counter
        if embodiment_properties.name in embodiment_name_to_id:
            cur_embodiment_id = embodiment_name_to_id[embodiment_properties.name]
        else:
            embodiment_name_to_id[embodiment_properties.name] = cur_embodiment_id = embodiment_id_counter
            embodiment_properties_by_id.append(embodiment_properties)
            embodiment_id_counter += 1
        return cur_embodiment_id
    
    # Prepare the head info
    info_by_head = {}
    if policy_loss_enabled:
        info_by_head['policy'] = HeadInfo('policy', 'regression', 1, True)
    for metric_name in self_model_metric_names:
        info_by_head[metric_name] = get_self_model_head_info(cfg, metric_name)
    if fk_enabled:
        info_by_head['policy_forward_kinematics'] = HeadInfo('policy_forward_kinematics', 'regression', 3, False)

    # Load the data
    if cfg.dataset.policy.max_training_dirs != -1:
        intial_len = len(cfg.dataset.policy.train_dirs)
        cfg.dataset.policy.train_dirs = cfg.dataset.policy.train_dirs[:cfg.dataset.policy.max_training_dirs]
        print(f'Selected {len(cfg.dataset.policy.train_dirs)} of {intial_len} training directories to actually pull files from (prioritizing earlier items in the list)')
    dataset_device = cfg.dataset.policy.device
    root = cfg.dataset.policy.root
    split_name_to_data = {}
    for split_name, split_dirs in zip(['train', 'val', 'test'], [cfg.dataset.policy.train_dirs, cfg.dataset.policy.validation_dirs, cfg.dataset.policy.test_dirs]):
        paths = [os.path.join(root, d) for d in split_dirs]

        # traverse folder structure to get file names of data files (if multiple files in a folder, then pick the last one alphabetically as this is the most recent run)
        fnames = []
        for dataset_dir in paths:
            cur_fnames = glob(dataset_dir + "/*")
            cur_fnames.sort(reverse=True)
            if len(cur_fnames) == 0:
                print(f'ERROR: could not find state logs for embodiment in directory {dataset_dir}')

            fnames.append(cur_fnames[0])
        
        if len(fnames) == 0:
            split_name_to_data[split_name] = {
                'dataset': None,
                'embodiment_ids': set()
            }
            print(f'WARNING: no data loaded for {split_name} split')
            continue

        # for the test split, only load embodiment information and not any demonstration information (since we don't assume demonstration data exists for test embodiments). If we are in test_only mode then also don't need to load the training data
        if split_name == 'test' or cfg.test_only:
            embodiment_ids_set = set()
            for fname in fnames:
                # it's unfortunate that the entire file has to be loaded into memory include all the observation/action data if it is present because we are only accessing the embodiment properties here. There in an mmap parameter in newer torch versions that could be leveraged to potentially address this
                data = torch.load(fname, map_location=cfg.dataset.policy.device)
                embodiment_properties = EmbodimentProperties.from_dict(data['embodiment_properties'])
                cur_embodiment_id = register_embodiment(embodiment_properties)
                embodiment_ids_set.add(cur_embodiment_id)

            split_name_to_data[split_name] = {
                'dataset': None,
                'embodiment_ids': embodiment_ids_set
            }

            cur_embodiment_names = sorted([embodiment_properties_by_id[i].name for i in embodiment_ids_set])

            fnames_str = '\n'.join(fnames)
            print(f'{split_name} split: Loaded {len(fnames)} embodiments with names {cur_embodiment_names}. Using only `embodiment_properties` from:\n{fnames_str}\n')
            continue

        # for each file in dataset, load the data
        properties_by_fname = {}
        tokenized_obs_list = []
        if policy_loss_enabled:
            actions_list = []
        if fk_enabled:
            forward_kinematics_list = []
        max_obs_dim = 0
        max_action_dim = -1
        dataset_size = 0
        embodiment_ids_by_file = []
        embodiment_ids_set = set()
        
        # Determine variable local and global obs sizes by looking at the first entry in the dataset (note that for both the local and global obs, the observation dimension might be missing, in which case the ObservationTokenizer automatically inserts a dimension of 1)
        data = torch.load(fnames[0], map_location=dataset_device)
        raw_obs = data['obs']
        global_obs = raw_obs['global'] # List of Tensor of size (num_steps, num_envs, obs_size [optional; may be missing this dim -> assume size is 1])
        local_obs = raw_obs['local'] # List of Tensor of size (num_steps, num_envs, num_dof, obs_size [optional; may be missing this dim -> assume size is 1])
        variable_global_obs_size = sum([obs.size(2) if obs.dim() == 3 else 1 for obs in global_obs]) # TODO: instead of having last dimension optional if it is 1, just always requirement that dimension is present
        variable_local_obs_size = sum([obs.size(3) if obs.dim() == 4 else 1 for obs in local_obs])
        assert_and_set(cfg.tokenization, 'variableGlobalObsSize', variable_global_obs_size if cfg.tokenization.includeVariableGlobalObs else 0)
        assert_and_set(cfg.tokenization, 'variableLocalObsSize', variable_local_obs_size if cfg.tokenization.includeVariableLocalObs else 0)

        # setup multithreading to load dataset files
        load_dataset_futures = []
        with ThreadPoolExecutor(max_workers=cfg.dataset.policy.max_load_file_threads) as executor:
            for fname in fnames:
                future = executor.submit(load_policy_dataset_file, fname, cfg)
                load_dataset_futures.append(future)

            # append the observations/actions from each file to the dataset
            for future, fname in tqdm(zip(load_dataset_futures, fnames), desc=f'Loading {split_name} dataset files', total=len(fnames)):
                tokenized_obs, actions, forward_kinematics, embodiment_properties, num_steps, num_envs, dof_count, tokenizer_obs_sizes = future.result()

                # Embodiment ID and properties
                cur_embodiment_id = register_embodiment(embodiment_properties)
                embodiment_ids_by_file.append(cur_embodiment_id)
                embodiment_ids_set.add(cur_embodiment_id)

                # Store properties for logging later
                properties_by_fname[fname] = {
                    'num_steps': num_steps,
                    'num_envs': num_envs,
                    'dof_count': dof_count
                }

                # Validate the sizes of observations from each file are the same using the sizes computed by the ObservationTokenizer. Store these values into the tokenization config
                for k, v in tokenizer_obs_sizes.items():
                    assert_and_set(cfg.tokenization, k, v)

                tokenized_obs_list.append(tokenized_obs)
                if policy_loss_enabled:
                    actions_list.append(actions)
                if fk_enabled:
                    forward_kinematics_list.append(forward_kinematics)

                max_obs_dim = max(max_obs_dim, tokenized_obs.size(1))
                if policy_loss_enabled:
                    max_action_dim = max(max_action_dim, actions.size(1))
                dataset_size += tokenized_obs.size(0)
        
        # put all data into tensors
        max_dof_count_this_dataset = max([embodiment_properties_by_id[id].dof_count for id in embodiment_ids_set])
        tokenized_obs = torch.empty((dataset_size, max_obs_dim), device=dataset_device)
        embodiment_ids = torch.empty(dataset_size, dtype=torch.int, device=dataset_device)
        if policy_loss_enabled:
            actions = torch.empty((dataset_size, max_action_dim), device=dataset_device)
        if fk_enabled:
            forward_kinematics = torch.empty((dataset_size, max_dof_count_this_dataset, 3), device=dataset_device)
        entry_i = 0
        for i in range(len(tokenized_obs_list)):
            cur_tokenized_obs = tokenized_obs_list[i]
            if policy_loss_enabled:
                cur_actions = actions_list[i]
                cur_dof_count = cur_actions.size(1)
            if fk_enabled:
                cur_fks = forward_kinematics_list[i]
                cur_dof_count = cur_fks.size(1)
            num_entries = cur_tokenized_obs.size(0)
            cur_embodiment_id = embodiment_ids_by_file[i]
            
            # pad the observations and actions to match max lengths (using 1e5 (inf breaks it since it causes NaNs) as the value just as a sanity to make sure the value is never used in determining any model output, loss computation, or otherwise)
            cur_tokenized_obs = torch.nn.functional.pad(cur_tokenized_obs, (0, max_obs_dim - cur_tokenized_obs.size(1)), value=1e5)
            if policy_loss_enabled:
                cur_actions = torch.nn.functional.pad(cur_actions, (0, max_action_dim - cur_dof_count), value=1e5)
            if fk_enabled:
                cur_fks = torch.nn.functional.pad(cur_fks, (0, 0, 0, max_dof_count_this_dataset - cur_dof_count), value=1e5)

            tokenized_obs[entry_i:entry_i+num_entries] = cur_tokenized_obs
            embodiment_ids[entry_i:entry_i+num_entries] = cur_embodiment_id
            if policy_loss_enabled:
                actions[entry_i:entry_i+num_entries] = cur_actions
            if fk_enabled:
                forward_kinematics[entry_i:entry_i+num_entries] = cur_fks

            entry_i += num_entries

        # print dataset information
        new_embodiments_this_split = embodiment_id_counter - embodiment_id_counter_split
        embodiment_id_counter_split = embodiment_id_counter
        fnames_str = []
        for fname in fnames:
            props = properties_by_fname[fname]
            fnames_str.append(f' - {fname} {props}')
        fnames_str = '\n'.join(fnames_str)
        print(f'{split_name} split: Loaded {dataset_size} entries with max obs dim {max_obs_dim} and max action dim {max_action_dim} with {new_embodiments_this_split} new embodiments in this split. Data from:\n{fnames_str}')

        dataset_size = tokenized_obs.size(0)

        # Add heads
        outputs_by_name = {}
        if policy_loss_enabled:
            outputs_by_name['policy'] = actions
        if fk_enabled:
            outputs_by_name['policy_forward_kinematics'] = forward_kinematics
        
        # Prepare self modeling metrics and heads
        for metric_name in self_model_metric_names:
            metric_by_embodiment = get_self_model_metric(cfg, metric_name, embodiment_properties_by_id, dataset_device)
            metric_by_embodiment = metric_by_embodiment[:,:max_dof_count_this_dataset] # the metrics are padded for all embodiments seen so far, but instead we want to pad only up to the max DoF count of embodiments that are actually part of the current split (becuase then it will match the size of the policy observations)
            outputs_by_name[metric_name] = metric_by_embodiment[embodiment_ids] # select the embodiment IDs that match the data that will be put into the dataset (likely each embodiment ID will be selected multiple times)

        dataset = DistillDataset(tokenized_obs, outputs_by_name, embodiment_ids)
        split_name_to_data[split_name] = {
            'dataset': dataset,
            'embodiment_ids': embodiment_ids_set
        }
        print()

    # ensure that all splits have independent embodiment types
    for split_name_1 in split_name_to_data:
        for split_name_2 in split_name_to_data:
            if split_name_1 == split_name_2:
                continue

            s1 = split_name_to_data[split_name_1]['embodiment_ids']
            s2 = split_name_to_data[split_name_2]['embodiment_ids']

            assert len(s1.intersection(s2)) == 0, 'data splits should have independent embodiments'

    def split_dataset_into_train_and_val(name):
        # Split dataset into a '{name}_train' and f'{name}_val' splits
        dataset, embodiment_ids = split_name_to_data[name]['dataset'], split_name_to_data[name]['embodiment_ids']

        if dataset is None:
            split_name_to_data.pop(name)
            split_name_to_data[f'{name}_train'] = {
                'dataset': None,
                'embodiment_ids': embodiment_ids
            }
            split_name_to_data[f'{name}_val'] = {
                'dataset': None,
                'embodiment_ids': embodiment_ids
            }
            return

        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [1 - cfg.dataset.policy.val_proportion, cfg.dataset.policy.val_proportion], generator=torch.Generator().manual_seed(cfg.seed))
        split_name_to_data.pop(name)
        split_name_to_data[f'{name}_train'] = {
            'dataset': train_dataset,
            'embodiment_ids': embodiment_ids
        }
        split_name_to_data[f'{name}_val'] = {
            'dataset': val_dataset,
            'embodiment_ids': embodiment_ids
        }
        print(f'Split the {name} dataset into {name}_train with {len(train_dataset)} entries and into train_val with {len(val_dataset)} entries.')
    
    split_dataset_into_train_and_val('train')
    split_dataset_into_train_and_val('val')

    # Convert all datasets into dataloaders
    for split_name, entry in split_name_to_data.items():
        dataset = entry.pop('dataset')
        if dataset is None:
            entry['dataloader'] = None
        else:
            # if requested, take only fixed amount of dataset
            if cfg.dataset.policy.max_split_entries > 0:
                initial_amount = len(dataset)
                num_to_keep = min(cfg.dataset.policy.max_split_entries, initial_amount)
                dataset = Subset(dataset, torch.arange(num_to_keep))
                print(f'Keeping {num_to_keep} of {initial_amount} entries from the {split_name} dataset split')

            # create dataloader
            additional_args = {
                'shuffle': True,
                'batch_size': cfg.train.train_batch_size if '_train' in split_name else cfg.train.eval_batch_size
            }
            if torch.device(dataset_device).type == 'cpu' and torch.device(dataloader_device).type == 'cuda':
                additional_args['pin_memory'] = True
                additional_args['pin_memory_device'] = dataloader_device
            split_name_to_data[split_name]['dataloader'] = DataLoader(dataset, **additional_args)

    split_name_to_data['all'] = {
        'dataloader': None,
        'embodiment_ids': []
    }
    
    return DistillTrainingInfo(split_name_to_data, embodiment_properties_by_id, info_by_head)


def load_policy_dataset_file(fname: str, cfg: DictConfig):
    fk_enabled = cfg.objective.policy_forward_kinematics.loss_weight > 0 and cfg.objective.include_self_modeling
    policy_loss_enabled = cfg.objective.policy.loss_weight > 0

    # Load the data
    data = torch.load(fname, map_location=cfg.dataset.policy.device)
    raw_obs = data['obs']
    global_obs = raw_obs['global'] # List of Tensor of size (num_steps, num_envs, obs_size [optional; may be missing this dim -> assume size is 1])
    local_obs = raw_obs['local'] # List of Tensor of size (num_steps, num_envs, num_dof, obs_size [optional; may be missing this dim -> assume size is 1])
    at_reset_buf = raw_obs['resets'] # Tensor of size (num_steps, num_envs)
    actions = data['actions'] # (time, num_envs, dof_count)
    embodiment_properties = EmbodimentProperties.from_dict(data['embodiment_properties'])

    # Limit data to maximum requested samples
    if cfg.dataset.policy.max_samples_per_file != -1:
        num_steps, num_envs, dof_count = actions.shape
        num_envs_to_keep = max(cfg.dataset.policy.max_samples_per_file // num_steps, 1)
        actions = actions[:, :num_envs_to_keep]
    
    num_steps, num_envs, dof_count = actions.shape

    # Setup observation tokenizer
    observation_tokenizer = ObservationTokenizer(cfg.tokenization, embodiment_properties, cfg.dataset.policy.device, num_envs)
    cur_tokenized_obs_list = []

    # run forward kinematics for each env and each step
    if fk_enabled:
        joint_limits = torch.from_numpy(embodiment_properties.joint_properties['joint_angle_limits']) # (dof_count, 2)
        lower_limits, upper_limits = joint_limits[:, 0], joint_limits[:, 1]

        cur_joint_state = local_obs[cfg.task.variable_local_obs_joint_state_index][:, :num_envs, :] # (num_steps, num_envs, dof_count)
        
        if cfg.task.variable_local_obs_joint_state_normalized:
            # need to go from -1, 1 range to limit_lower, limit_upper range
            cur_joint_state = (cur_joint_state + 1) / 2 # -1,1 -> 0,1 range
            cur_joint_state = cur_joint_state * (upper_limits - lower_limits) + lower_limits # range is between lower_limits and upper_limits

        cur_joint_state = cur_joint_state.reshape(num_steps*num_envs, dof_count) # (num_steps*num_envs, dof_count)
        urdf = get_urdf(embodiment_properties.asset_file_contents)
        forward_kinematics = fk(urdf, cur_joint_state, embodiment_properties.joint_name_to_joint_i) # (num_steps*num_envs, dof_count, 3)
    else:
        forward_kinematics = None

    # step through each step in the data
    if policy_loss_enabled:
        cur_actions_list = []
    for step in range(num_steps):
        # observations
        cur_global_obs = [entry[step, :num_envs] for entry in global_obs]
        cur_local_obs = [entry[step, :num_envs] for entry in local_obs]
        cur_at_reset_buf = at_reset_buf[step, :num_envs]
        cur_tokenized_obs = observation_tokenizer.build_tokenized_observation(cur_global_obs, cur_local_obs, cur_at_reset_buf)
        cur_tokenized_obs_list.append(cur_tokenized_obs.clone())

        # actions
        if policy_loss_enabled:
            cur_actions_list.append(actions[step])

    tokenized_obs = torch.cat(cur_tokenized_obs_list)
    if policy_loss_enabled:
        actions = torch.cat(cur_actions_list)
        assert tokenized_obs.size(0) == actions.size(0), 'obs and actions must have same number of entries'
    else:
        actions = None

    if fk_enabled:
        assert tokenized_obs.size(0) == forward_kinematics.size(0), 'obs and fk must have same number of entries'

    # observation component sizes
    tokenizer_obs_sizes = {
        'fixedGlobalObsSize': observation_tokenizer.fixed_global_obs_size,
        'fixedLocalObsSize': observation_tokenizer.fixed_local_obs_size,
        'variableGlobalObsSize': observation_tokenizer.variable_global_obs_size,
        'variableLocalObsSize': observation_tokenizer.variable_local_obs_size,
        'globalObsSize': observation_tokenizer.global_obs_size,
        'localObsSize': observation_tokenizer.local_obs_size
    }

    return tokenized_obs, actions, forward_kinematics, embodiment_properties, num_steps, num_envs, dof_count, tokenizer_obs_sizes

def get_self_model_dataset(cfg: DictConfig, self_model_metric_names: List[str]):
    """
    sets `globalObsSize`, `localObsSize`, `variableGlobalObsSize` and `variableLocalObsSize` in `cfg.tokenization`
    """
    dataset_device = torch.device(cfg.gpu)
    assert_and_set(cfg.tokenization, 'variableLocalObsSize', 0)
    assert_and_set(cfg.tokenization, 'variableGlobalObsSize', 0)

    # Load all embodiments
    embodiment_properties_by_id: List[EmbodimentProperties] = []
    for asset_fname in os.listdir(cfg.dataset.self_model.asset_dir):
        asset_path = os.path.join(cfg.dataset.self_model.asset_dir, asset_fname)
        cfg_path = os.path.join(cfg.dataset.self_model.config_dir, asset_fname.replace('.urdf', '.yaml'))
        asset_name = Path(asset_path).stem

        with open(asset_path, 'r') as f:
            asset_contents = f.read()

        with open(cfg_path, 'r') as f:
            asset_cfg = yaml.safe_load(f)

        embodiment_properties = EmbodimentProperties(asset_name, asset_contents, joint_name_to_joint_i=asset_cfg['joint_name_to_joint_i'])
        embodiment_properties_by_id.append(embodiment_properties)
    embodiment_properties_by_id.sort(key=lambda embodiment_properties: embodiment_properties.name)
    if cfg.dataset.self_model.max_embodiments != -1:
        embodiment_properties_by_id = embodiment_properties_by_id[:cfg.dataset.self_model.max_embodiments]

    # For each embodiment compute the observation (consisting of only fixed geometry information)
    obs_by_embodiment = []
    for embodiment_properties in embodiment_properties_by_id:
        tokenizer = ObservationTokenizer(OmegaConf.to_container(cfg.tokenization, resolve=True), embodiment_properties, dataset_device, 1)
        obs = tokenizer.build_tokenized_observation([], [], torch.LongTensor([0])) # (batch_size=1, obs_size)
        obs = obs.squeeze(0) # (obs_size,)
        obs_by_embodiment.append(obs)

        assert_and_set(cfg.tokenization, 'localObsSize', tokenizer.local_obs_size)
        assert_and_set(cfg.tokenization, 'globalObsSize', tokenizer.global_obs_size)

    # Right pad each observation to the max length
    max_obs_length = max([obs.size(0) for obs in obs_by_embodiment])
    for i, obs in enumerate(obs_by_embodiment):
        obs_by_embodiment[i] = torch.nn.functional.pad(obs, (0, max_obs_length - obs.size(0)), value=1e5)

    # Prepare self modeling metrics
    outputs_by_name = {}
    info_by_head = {}
    for metric_name in self_model_metric_names:
        metric_by_embodiment, head_info = get_self_model_metric(cfg, metric_name, embodiment_properties_by_id, dataset_device)
        outputs_by_name[metric_name] = metric_by_embodiment
        info_by_head[metric_name] = head_info

    # Create dataset
    full_dataset = DistillDataset(torch.stack(obs_by_embodiment), outputs_by_name, range(len(embodiment_properties_by_id)))
    train_dataset, val_dataset = random_split(full_dataset, [cfg.dataset.self_model.train_size, cfg.dataset.self_model.val_size])

    # Create dataset splits
    split_name_to_data = {}
    for split_name, dataset, batch_size in zip(['train_train', 'train_val', 'val_train', 'val_val', 'test', 'all'], [train_dataset, None, None, val_dataset, None, full_dataset], [cfg.train.train_batch_size, None, None, cfg.train.eval_batch_size, None, cfg.train.eval_batch_size]):
        if dataset is not None and len(dataset) == 0:
            dataset = None # handles case where based on split size no samples ended up in the split

        if dataset is None:
            dataloader = None
            embodiment_ids = []
        else:
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            embodiment_ids = []
            for obs, metric, embodiment_id in dataset:
                embodiment_ids.append(embodiment_id)
        split_name_to_data[split_name] = {'dataloader': dataloader, 'embodiment_ids': embodiment_ids}

    return DistillTrainingInfo(split_name_to_data, embodiment_properties_by_id, info_by_head)

def get_self_model_head_info(cfg, metric_name: str):
    metric_name_to_metric_size = {
        'num_serial_chains': cfg.task.max_dof_count + 1,
        'degree_count': cfg.task.max_dof_count,
        'parent_count': cfg.task.max_dof_count,
        'child_count': cfg.task.max_dof_count,
        'serial_chain_length': cfg.task.max_dof_count + 1,
        'parent_link_id': cfg.task.max_dof_count + 1,
        'parent_child_pose': 12
    }

    # Figure out classification/regression and any preprocessing
    if metric_name == 'parent_child_pose':
        prediction_type = 'regression'
    else:
        prediction_type = 'classification'
    
    metric_size = metric_name_to_metric_size[metric_name]
    head_info = HeadInfo(metric_name, prediction_type, metric_size, False)
    return head_info

def get_self_model_metric(cfg, metric_name: str, embodiment_properties_by_id: List[EmbodimentProperties], device):
    # For each embodiment compute the desired metric
    metric_by_embodiment = []

    for embodiment_properties in embodiment_properties_by_id:
        if metric_name == 'num_serial_chains': # classification for single token [0, max_dof_count]
            metric = torch.from_numpy(embodiment_properties.joint_properties['num_serial_chains']).to(torch.long).repeat(embodiment_properties.dof_count)
        elif metric_name in ['degree_count', 'parent_count', 'child_count']: # classification for each token [0, max_dof_count-1]
            metric = torch.tensor(embodiment_properties.joint_properties[metric_name], dtype=torch.long).squeeze(1) # (dof_count,)
        elif metric_name == 'serial_chain_length': # classification for each token [0, max_dof_count]
            metric = torch.tensor(embodiment_properties.joint_properties[metric_name], dtype=torch.long).squeeze(1) # (dof_count,)
        elif metric_name == 'parent_link_id':
            parent_link_names = embodiment_properties.joint_properties['parent_link_names']
            metric = torch.tensor([cfg.tokenization.linkNameToId[link_name] for link_name in parent_link_names], dtype=torch.long) # (dof_count,)
        elif metric_name == 'parent_child_pose': # regression for each token
            metrics = []
            for sub_metric_name in ['parent_joint_origin', 'parent_joint_rot', 'child_joint_origin', 'child_joint_rot']:
                metrics.append(torch.from_numpy(embodiment_properties.joint_properties[sub_metric_name]).to(torch.float32)) # (dof_count, 3)
            metric = torch.cat(metrics, dim=1)
        else:
            raise NotImplementedError
        metric_by_embodiment.append(metric)

    # Right pad the metric since they metric length is dependent on DoF count
    max_metric_length = max([metric.size(0) for metric in metric_by_embodiment])
    for i, metric in enumerate(metric_by_embodiment):
        padding_size = [0, 0] * (metric.dim() - 1) + [0, max_metric_length - metric.size(0)] # pads the sequence/token dimension
        metric_by_embodiment[i] = torch.nn.functional.pad(metric, padding_size, value=1e5)

    metric_by_embodiment = torch.stack(metric_by_embodiment).to(device) # (num_embodiments, dof_count) if classification and (num_embodiments, dof_count, metric_size) if regression
    return metric_by_embodiment
    

class DistillDataset(Dataset):
    def __init__(self, tokenized_obs: Tensor, outputs_by_name: Dict[str, Tensor], embodiment_ids: Tensor):
        self.tokenized_obs = tokenized_obs
        self.outputs_by_name = outputs_by_name
        self.embodiment_ids = embodiment_ids

    def __len__(self):
        return self.tokenized_obs.size(0)

    def __getitem__(self, idx):
        return {
            'obs': self.tokenized_obs[idx],
            'embodiment_ids': self.embodiment_ids[idx],
            'outputs_by_head': {name: output[idx] for name, output in self.outputs_by_name.items()}
        }

@dataclass
class HeadInfo:
    name: str
    prediction_type: str
    output_dim: int
    squeeze_output_dim: bool

class DistillTrainingInfo:
    def __init__(self, split_name_to_data: Dict[str, DataLoader], embodiment_properties_by_id: List[EmbodimentProperties], info_by_head: Dict[str, HeadInfo]):
        self.train_train_dataloader, self.train_embodiment_ids = split_name_to_data['train_train']['dataloader'], split_name_to_data['train_train']['embodiment_ids']
        self.train_val_dataloader, self.train_embodiment_ids = split_name_to_data['train_val']['dataloader'], split_name_to_data['train_val']['embodiment_ids']
        self.val_train_dataloader, self.val_embodiment_ids = split_name_to_data['val_train']['dataloader'], split_name_to_data['val_train']['embodiment_ids']
        self.val_val_dataloader, self.val_embodiment_ids = split_name_to_data['val_val']['dataloader'], split_name_to_data['val_val']['embodiment_ids']
        self.test_embodiment_ids = split_name_to_data['test']['embodiment_ids']
        self.all_dataloader, self.all_embodiment_ids = split_name_to_data['all']['dataloader'], split_name_to_data['all']['embodiment_ids']
        self.embodiment_properties_by_id = embodiment_properties_by_id
        self.info_by_head = info_by_head
