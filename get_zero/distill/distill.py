"""Takes dataset of (tokenized observation, actions, URDF) from trained policies that have been rolled out and train a transformer policy that can control different embodiments regardless of hardware configuration."""

import torch
import hydra
from omegaconf import OmegaConf, DictConfig, open_dict
from get_zero.distill.utils.dataset import get_datasets
from get_zero.distill.models.embodiment_transformer import EmbodimentTransformer
from get_zero.distill.models.vis_embodiment_transformer import vis_embodiment_transformer
from get_zero.distill.utils.embodiment_util import EmbodimentEvaluator
from get_zero.distill.utils.learning import get_lr_scheduler
from get_zero.distill.utils.generic import set_seed, tensordict_to_device, add_custom_omegaconf_resolvers, assert_and_set
from multiprocessing.pool import AsyncResult
from torch.optim import Adam, AdamW
import wandb
from datetime import datetime
from tqdm import tqdm
import os
from typing import Dict, List, Tuple
import shutil
from typing import Iterable
import gc
import yaml
import copy
import statistics

@hydra.main(version_base="1.2", config_name="config", config_path="./cfg")
def main(cfg: DictConfig):
    distill(cfg)   

def distill(base_cfg: DictConfig, embodiment_evaluator: EmbodimentEvaluator=None):
    """
    Helps orchestrate calls to distill (can run both training and finetuning). This is useful because we want to call our script a single time and do training then potentially finetuning.

    Args:
    - cfg: distillation config
    - embodiment_evaluator: provides a way to evalate distilled policy checkpoints to get real evaluation metrics
    """
    
    # Initialize experiment
    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")    
    add_custom_omegaconf_resolvers()

    # Run training and/or finetuning
    runs = []
    if base_cfg.should_train or base_cfg.test_only:
        runs.append(False)
    if base_cfg.should_finetune and not base_cfg.test_only:
        runs.append(True)

    if not base_cfg.should_train and not base_cfg.should_finetune:
        assert base_cfg.test_only, '`test_only`` should be set if neither training nor finetuning'
    
    for args in runs:
        is_finetuning = args
        cfg = copy.deepcopy(base_cfg)
        
        # if finetuning then merge in finetuning config
        if is_finetuning:
            cfg = OmegaConf.merge(cfg, cfg.finetuning_overrides)
        
        if base_cfg.test_only:
            cfg.train.current_training_mode = 'test'

        experiment_name_suffix = f'_{cfg.experiment}' if cfg.experiment else ''
        experiment_name = f'distill{experiment_name_suffix}_{start_time}'
        experiment_dir = os.path.join('runs', f'{experiment_name}_{cfg.train.current_training_mode}')

        additional_tags = [f'{cfg.train.current_training_mode}_distill']
        base_cfg.checkpoint = _run_distill(cfg, experiment_dir, embodiment_evaluator, additional_tags) # setting checkpoint ensures that if a finetuning run follows the training run then the finetuning will use the best checkpoint from the training run as an initialization

def _run_distill(cfg: DictConfig, experiment_dir: str, embodiment_evaluator: EmbodimentEvaluator, additional_tags: List[str] = []) -> str:
    """Performs entire training procedure either with normal training or finetuning and returns a path to the best checkpoint."""
    device = torch.device(cfg.gpu)
    set_seed(cfg.seed)
    os.makedirs(experiment_dir)
    experiment_name = os.path.basename(experiment_dir)
    embodiment_evaluator = None if cfg.train.skip_embodiment_eval or cfg.objective.policy.loss_weight == 0 else embodiment_evaluator

    # If checkpoint specified, then update model and tokenization config using the values from the checkpoint to ensure that we are using to correct config to initialize the model
    assert not cfg.test_only or cfg.checkpoint, 'must provide checkpoint if using test_only mode'
    if cfg.checkpoint:
        print(f'Using tokenization config and model config from the checkpoint at {cfg.checkpoint}')
        checkpoint_data = torch.load(cfg.checkpoint, map_location='cpu')
        if 'tokenization_config' not in checkpoint_data or 'model_config' not in checkpoint_data:
            print('WARNING: provided checkpoint does not have `tokenization_config` or `model_config` present, so we are assuming that the config used to create the checkpoint matches the config currently specified')
        else:
            cfg.tokenization = OmegaConf.create(checkpoint_data['tokenization_config'])
            cfg.model = OmegaConf.create(checkpoint_data['model_config'])
        del checkpoint_data

    # Add directory count for the various training splits to the config
    assert_and_set(cfg.dataset.policy, 'numTrainDirs', len(cfg.dataset.policy.train_dirs))
    assert_and_set(cfg.dataset.policy, 'numValDirs', len(cfg.dataset.policy.validation_dirs))
    assert_and_set(cfg.dataset.policy, 'numTestDirs', len(cfg.dataset.policy.test_dirs))

    # WandB
    if cfg.wandb_activate:
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            group=cfg.wandb_group,
            tags=cfg.wandb_tags + additional_tags,
            id=f'uid_{experiment_name}',
            name=experiment_name,
            config=OmegaConf.to_container(cfg, resolve=True)
        )
    print(f'--- Beginning distillation with logging directory: {experiment_dir} ---')
        
    # Load dataset
    print(f'\nBEGINNING DATASET LOADING')
    if cfg.test_only:
        cfg.dataset.policy.max_samples_per_file = 1 # if testing, then we don't need to load state logs for training, so just load the smallest amount from each file
    dataset = get_datasets(cfg)
    print(f'FINISHED DATASET LOADING\n')

    # Setup the output head configs based on the metrics used in the dataset
    for head_name, head_info in dataset.info_by_head.items():
        OmegaConf.set_struct(cfg, True)
        with open_dict(cfg):
            cfg.model.heads[head_name] = {
                'output_dim': head_info.output_dim,
                'squeeze_output_dim': head_info.squeeze_output_dim
            }
        print(f'Added `{head_name}` head to network with size {head_info.output_dim}')

    # Save config to file (needs to happen after loading dataset since additional config is set in `get_datasets`)
    config_yaml_path = os.path.join(experiment_dir, 'config.yaml')
    with open(config_yaml_path, 'w') as f:
        yaml.safe_dump(OmegaConf.to_container(cfg, resolve=True), f)
    print(f'Saved config to {config_yaml_path}')

    # Prepare embodiment evaluation
    if embodiment_evaluator:
        # Determine the splits each embodiment is a part of. Each embodiment can be apart of only 1 of ['train', 'val', 'test'] splits and can be apart of 0 or more additional splits.
        embodiment_name_to_splits = {}
        split_name_to_embodiment_ids = {}
        embodiment_name_to_embodiment_id = {}

        for id, embodiment_properties in enumerate(dataset.embodiment_properties_by_id):
            embodiment_name_to_embodiment_id[embodiment_properties.name] = id

        for split_name, embodiment_ids_in_split in zip(['train', 'val', 'test'], [dataset.train_embodiment_ids, 
        dataset.val_embodiment_ids, dataset.test_embodiment_ids]):
            split_name_to_embodiment_ids[split_name] = []
            for embodiment_id in embodiment_ids_in_split:
                embodiment_name_to_splits[dataset.embodiment_properties_by_id[embodiment_id].name] = [split_name]
                split_name_to_embodiment_ids[split_name].append(embodiment_id)
            
        for custom_split_dict in cfg.dataset.policy.custom_splits:
            assert len(custom_split_dict) == 1
            custom_split_name, custom_split_embodiment_names = list(custom_split_dict.keys())[0], list(custom_split_dict.values())[0]
            split_name_to_embodiment_ids[custom_split_name] = []
            for name in custom_split_embodiment_names:
                embodiment_name_to_splits[name].append(custom_split_name)
                split_name_to_embodiment_ids[custom_split_name].append(embodiment_name_to_embodiment_id[name])

        # prepare evaluation
        embodiment_evaluator.prepare_evaluation(experiment_dir, cfg.task.name, dataset.embodiment_properties_by_id, cfg.model, cfg.tokenization, embodiment_name_to_splits, additional_tags)
                
        # start baseline eval
        if not cfg.train.skip_embodiment_baseline_eval:
            print('Starting baseline embodiment evaluation')
            async_baseline_eval = embodiment_evaluator.evaluate_baseline()
        else:
            print('Skipping baseline embodiment evaluation')        

        # determine which checkpoints to run embodiment evaluation on
        if not cfg.test_only:
            assert cfg.train.num_epochs >= cfg.train.embodiment_eval_average_last_epochs, f'request to average over last {cfg.train.embodiment_eval_average_last_epochs} epochs, but only training for {cfg.train.num_epochs} epochs'
            epoch_i_to_evaluate_embodiment = []
            for epoch_i in range(cfg.train.num_epochs + 1):
                is_after_start = epoch_i >= cfg.train.embodiment_eval_start_epoch
                is_on_interval = (epoch_i - cfg.train.embodiment_eval_start_epoch) % cfg.train.embodiment_eval_epoch_freq == 0
                is_final_evaluations = cfg.train.num_epochs - epoch_i < cfg.train.embodiment_eval_average_last_epochs

                if (is_after_start and is_on_interval) or is_final_evaluations:
                    epoch_i_to_evaluate_embodiment.append(epoch_i)
            print(f'Going to run embodiment evaluation at epochs: {epoch_i_to_evaluate_embodiment} and compute average final performance over final {cfg.train.embodiment_eval_average_last_epochs} epochs')
        
        checkpoint_embodiment_eval_metrics = []
    
    """Helper functions"""
    def save_checkpoint(name: str):
        checkpoints_dir = os.path.join(experiment_dir, 'checkpoints')
        os.makedirs(checkpoints_dir, exist_ok=True)
        checkpoint_out_path = os.path.join(checkpoints_dir, f'{name}.pt')
        torch.save({
            'model': model.state_dict(),
            'tokenization_config': OmegaConf.to_container(cfg.tokenization, resolve=True),
            'model_config': OmegaConf.to_container(cfg.model, resolve=True)
        }, checkpoint_out_path)
        print(f'Saved checkpoint to {checkpoint_out_path}')
        return os.path.abspath(checkpoint_out_path)

    def load_checkpoint(path):
        if not path:
            print('Not loading a model checkpoint since none was specified')
            return

        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model'])
        print(f'Loaded checkpoint from {path}')

    def compute_loss_multiple_heads(model_heads, gt_heads, embodiment_ids, **loss_kwargs):
        assert model_heads.keys() == gt_heads.keys()
        total_loss = 0
        average_accuracy = None
        average_metrics_by_head = {}
        num_heads_with_accuracy = 0 # potentially not all heads will have accuracy metric
        self_modeling_loss = None
        num_heads_self_modeling = 0

        for head_name in model_heads:
            model_out = model_heads[head_name]
            gt_out = gt_heads[head_name] 

            head_average_loss, head_average_accuracy = compute_loss_single_head(head_name, model_out, gt_out, embodiment_ids, **loss_kwargs)
            weighted_loss = head_average_loss * cfg.objective[head_name]['loss_weight']

            total_loss += weighted_loss
            average_metrics_by_head[head_name] = {
                'loss': weighted_loss.item()
            }

            if head_name != 'policy':
                if self_modeling_loss is None:
                    self_modeling_loss = 0
                self_modeling_loss += weighted_loss.item()
                num_heads_self_modeling += 1

            if head_average_accuracy is not None:
                num_heads_with_accuracy += 1
                average_metrics_by_head[head_name]['accuracy'] = head_average_accuracy
                if average_accuracy is None:
                    average_accuracy = 0
                average_accuracy += head_average_accuracy
        
        if average_accuracy is not None:
            average_accuracy /= num_heads_with_accuracy

        if self_modeling_loss is not None:
            average_metrics_by_head['average_self_modeling'] = {'loss': self_modeling_loss / num_heads_self_modeling}
        
        return total_loss, average_accuracy, average_metrics_by_head
    
    def compute_loss_single_head(head_name, mu_pred, mu_gt, embodiment_ids, print_misclassified=False):
        """
        The pred and gt will only be valid up to the dof_count entry. So we want to make sure the invalid values in pred and gt match, so that when we compute a loss it's only relevant over the valid values (loss at invalid points will be 0)
        
        Returns:
            - `average_loss`
            - `average_accuracy`: proportion of batch that has correct top class prediction (only appplicable to self modeling)
        """
        head_info = dataset.info_by_head[head_name]

        if head_info.prediction_type == 'regression':
            # Regression loss on target action
            cur_batch_size = mu_pred.size(0)
            dof_counts = dof_counts_by_id[embodiment_ids]
            padding_mask = (torch.arange(mu_pred.size(1), device=device)+1).unsqueeze(0).repeat(cur_batch_size, 1) > dof_counts.unsqueeze(1)
            mu_pred[torch.where(padding_mask)] = mu_gt[torch.where(padding_mask)] # set pred and gt to have the same value at invalid locations, so that loss at invalid locations will be 0

            average_loss = torch.nn.functional.mse_loss(mu_pred, mu_gt)
            average_accuracy = None
        elif head_info.prediction_type == 'classification':
            # Classification loss on target class (each token produces an output token)
            cur_batch_size, cur_max_dof_count, classification_dim = mu_pred.shape
            # each token outputs probability distribution, so need to run cross_entropy for every token. Do this by flattening out the token and batch dimensions
            # first step is setting invalid tokens to have non infinite outputs since inf values break the softmax (the actual value set does not matter as long as it's not inf as we don't compute loss for invalid tokens)
            dof_counts = dof_counts_by_id[embodiment_ids] # (batch_size,)
            padding_mask = (torch.arange(mu_pred.size(1), device=device)+1).unsqueeze(0).repeat(cur_batch_size, 1) > dof_counts.unsqueeze(1) # (batch_size, dof_count)
            mu_gt[torch.where(padding_mask)] = 999 # use 999 as an invalid label

            mu_pred = mu_pred.flatten(0, 1) # (batch, seq_len, output_dim) -> (batch*seq_len, output_dim)
            mu_gt = mu_gt.flatten() # (batch, seq_len) -> (batch*seq_len,)

            average_loss = torch.nn.functional.cross_entropy(mu_pred, mu_gt, ignore_index=999) # 999 is label corresponding to invalid tokens; loss averaged over non ignored values
            
            with torch.inference_mode():
                # First compute accuracy by counting the tokens that match between pred and gt and dividing by the number of valid tokens ()
                top_class_pred = torch.argmax(mu_pred, dim=1) # (batch_size*max_dof_count,)
                top_class_pred = top_class_pred.reshape(cur_batch_size, cur_max_dof_count) # (batch_size, max_dof_count)
                mu_gt = mu_gt.reshape(cur_batch_size, cur_max_dof_count) # (batch_size, max_dof_count)
                average_accuracy = torch.sum(top_class_pred == mu_gt) / torch.sum(mu_gt != 999) # this relies on top_class_pred and mu_gt having different values at invalid tokens (top_class_pred has 0 at invalid locations and gt has 999 at invalid locations), so we don't count them as matches

                # Second compute which embodiments had at least one of their tokens have an incorrect classification
                # If any of the tokens in a batch have incorrect classifcation, then mark the entire batch as incorrect.
                top_class_pred[torch.where(mu_gt == 999)] = 999 # set mu_gt and top_class_pred to have matching values at invalid token spots so that we don't count misclassifications there
                differences = top_class_pred != mu_gt # (batch_size, max_dof_count)
                misclassification_count = differences.sum(axis=1) # (batch_size,) number of tokens for each batch entry that had incorrect classification
                misclassified_idx = torch.where(misclassification_count > 0)[0] # [0] gets first item in tuple that only has 1 item

            average_accuracy = average_accuracy.item()
            if print_misclassified and len(misclassified_idx) > 0:
                print(f'Misclassified:\nEmbodiments:\n{embodiment_ids[misclassified_idx]}\nTop class pred:\n{top_class_pred[misclassified_idx]}\nGround truth:\n{mu_gt[misclassified_idx]}\n')

        return average_loss, average_accuracy

    @torch.inference_mode()
    def compute_dataloader_loss(dataloader, desc: str, **loss_kwargs) -> Tuple[float, float]:
        """
        Computes loss across an entire dataloader in model eval mode with no grads enabled.
        """
        model.eval()
        average_loss = 0
        average_accuracy = None
        num_examples = 0
        average_metrics_by_head = {}
        
        for batch in tqdm(dataloader, desc=f'dataset loss, {desc}'):
            obs, embodiment_ids, outputs_by_head_gt = batch['obs'].to(device), batch['embodiment_ids'].to(device), tensordict_to_device(batch['outputs_by_head'], device)

            cur_batch_size = obs.size(0)
            num_examples += cur_batch_size
            outputs_by_head = model(obs, embodiment_ids)
            cur_loss, cur_accuracy, cur_metrics_by_head = compute_loss_multiple_heads(outputs_by_head, outputs_by_head_gt, embodiment_ids, **loss_kwargs)
            cur_loss = cur_loss.item()

            # average over loss and accuracy
            average_loss += cur_loss * cur_batch_size

            if cur_accuracy is not None:
                if average_accuracy is None:
                    average_accuracy = 0
                average_accuracy += cur_accuracy * cur_batch_size
            else:
                average_accuracy = None

            # average over per head metrics
            for cur_head_name, cur_head_metrics in cur_metrics_by_head.items():
                if cur_head_name not in average_metrics_by_head:
                    average_metrics_by_head[cur_head_name] = {k: 0 for k in cur_head_metrics}
                for cur_metric_name, cur_metric in cur_head_metrics.items():
                    average_metrics_by_head[cur_head_name][cur_metric_name] += cur_metric * cur_batch_size
        
        # normalize by number of examples
        average_loss /= num_examples
        if average_accuracy is not None:
            average_accuracy /= num_examples
        
        for cur_head_name, cur_head_metrics in average_metrics_by_head.items():
            for cur_metric_name, cur_metric in cur_head_metrics.items():
                average_metrics_by_head[cur_head_name][cur_metric_name] /= num_examples
        
        return average_loss, average_accuracy, average_metrics_by_head
    
    def log_per_epoch_metrics(epoch_i: int, additional_metrics={}):
        print(f'\nBeginning to log per epoch metrics for epoch {epoch_i}')
        checkpoint_path = save_checkpoint(f'epoch_{epoch_i}')

        # per epoch embodiment evaluation
        if embodiment_evaluator and epoch_i in epoch_i_to_evaluate_embodiment:
            cur_checkpoint_metrics = embodiment_evaluator.evaluate_checkpoint(checkpoint_path, ['train', 'val'])
            checkpoint_embodiment_eval_metrics.append(cur_checkpoint_metrics)

        metrics = {
            'train/epoch_full': epoch_i
        }

        # compute loss on `train_val` and `val_val` splits
        for split_name, split_dataloader in zip(['train_val', 'val_val'], [dataset.train_val_dataloader, dataset.val_val_dataloader]):
            if split_dataloader is None:
                continue
            
            split_loss, split_accuracy, average_metrics_by_head = compute_dataloader_loss(split_dataloader, f'{split_name} dataset, epoch {epoch_i}')

            # metric logging
            metrics[f'train/{split_name}_loss'] = split_loss
            if split_accuracy is not None:
                metrics[f'train/{split_name}_accuracy'] = split_accuracy

            for head_name, head_metrics in average_metrics_by_head.items():
                for metric_name, metric_value in head_metrics.items():
                    metrics[f'train/{split_name}_{head_name}_{metric_name}'] = metric_value

        metrics.update(additional_metrics)

        print(f'Metrics for {epoch_i}:')
        for k, v in metrics.items():
            print(f' - {k}: {v}')

        if cfg.wandb_activate:
            wandb.log(metrics)

        return_data = {
            **metrics,
            'checkpoint_path': checkpoint_path
        }
        print()

        return return_data

    """Prepare for training"""
    if not cfg.test_only:
        # Initialize model from checkpoint, if provided
        model = EmbodimentTransformer(cfg.model, cfg.tokenization, dataset.embodiment_properties_by_id).to(device)
        load_checkpoint(cfg.checkpoint)

        train_dataloader = getattr(dataset, f'{cfg.train.train_split_name}_dataloader')

        # dof counts list
        dof_counts_by_id = []
        for embodiment_properties in dataset.embodiment_properties_by_id:
            dof_counts_by_id.append(embodiment_properties.dof_count)
        dof_counts_by_id = torch.tensor(dof_counts_by_id, dtype=torch.long, device=device)
        
        # Create optimizer + LR scheduler
        if cfg.train.optimizer.type == 'Adam':
            optimizer_class = Adam
        elif cfg.train.optimizer.type == 'AdamW':
            optimizer_class = AdamW
        else:
            raise NotImplementedError
        optimizer = optimizer_class(model.parameters(), lr=cfg.train.lr.start, betas=(cfg.train.optimizer.beta1, cfg.train.optimizer.beta2))
        lr_scheduler = get_lr_scheduler(cfg.train.lr, optimizer, len(train_dataloader) * cfg.train.num_epochs)

        @torch.inference_mode()
        def get_grad_norm():
            grad_norm = 0
            for p in model.parameters():
                param_norm = p.grad.detach().data.norm(2)
                grad_norm += param_norm.item() ** 2
            grad_norm = grad_norm ** 0.5
            return grad_norm

        """Training loop"""
        print('\nBEGINNING TRAINING LOOP')
        data_per_epoch = [log_per_epoch_metrics(0)]
        iteration = 0
        for epoch_i in range(cfg.train.num_epochs):
            model.train()

            # training batches
            for batch_i, batch in enumerate(tqdm(train_dataloader, desc=f'{cfg.train.train_split_name} epoch {epoch_i + 1}')):
                # TODO see if we can pre-fetch these on the GPU to save time
                # forward pass
                obs, embodiment_ids, outputs_by_head_gt = batch['obs'].to(device), batch['embodiment_ids'].to(device), tensordict_to_device(batch['outputs_by_head'], device)
                cur_batch_size = obs.size(0)
                iteration += cur_batch_size
                optimizer.zero_grad()
                outputs_by_head = model(obs, embodiment_ids)

                # loss & accuracy
                loss, accuracy, metrics_by_head = compute_loss_multiple_heads(outputs_by_head, outputs_by_head_gt, embodiment_ids)

                # backward
                loss.backward()

                # clip grad norm
                if cfg.train.clip_grads.log:
                    pre_clip_grad_norm = get_grad_norm()
                if cfg.train.clip_grads.enabled:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip_grads.value)
                if cfg.train.clip_grads.log:
                    post_clip_grad_norm = get_grad_norm() if cfg.train.clip_grads.enabled else pre_clip_grad_norm

                # step
                optimizer.step()
                cur_lr = lr_scheduler.get_lr()[0]
                lr_scheduler.step()

                # logging
                loss = loss.item() # critical to get item to prevent computation graph from building over time
                metrics = {
                    f'train/{cfg.train.train_split_name}_loss': loss,
                    'train/epoch': iteration / len(train_dataloader.dataset),
                    'train/lr': cur_lr
                }
                if cfg.train.clip_grads.log:
                    metrics['train/pre_clip_grad_norm'] = pre_clip_grad_norm
                    metrics['train/post_clip_grad_norm'] = post_clip_grad_norm
                if accuracy is not None:
                    metrics[f'train/{cfg.train.train_split_name}_accuracy'] = accuracy
                for head_name, head_metrics in metrics_by_head.items():
                    for metric_name, metric_value in head_metrics.items():
                        metrics[f'train/{cfg.train.train_split_name}_{head_name}_{metric_name}'] = metric_value
                
                if cfg.wandb_activate and batch_i + 1 < len(train_dataloader):
                    wandb.log(metrics)

            cur_epoch_data = log_per_epoch_metrics(epoch_i + 1, metrics)
            data_per_epoch.append(cur_epoch_data)

        print(f'TRAINING COMPLETE\n')

        """Perform some cleanup (mostly to clear CPU RAM that is filled with dataset information)"""
        dataloaders_to_clean = [dataset.train_train_dataloader, dataset.train_val_dataloader, dataset.val_train_dataloader, dataset.val_val_dataloader]
        for dataloader in dataloaders_to_clean:
            if dataloader is not None:
                del dataloader.dataset, dataloader
        del optimizer, lr_scheduler
        torch.cuda.empty_cache()
        gc.collect()

        """Determine best checkpoint"""
        print('BEGINNING TO COMPUTE BEST CHECKPOINT')
        assert cfg.train.eval_best_checkpoint_split in ['train', 'val', 'train_and_val']
        if len(dataset.val_embodiment_ids) == 0 and cfg.train.eval_best_checkpoint_split == 'val':
            print(f'WARNING: since eval_best_checkpoint_split == "val", but no validation embodiments were provided in the dataset, defaulting to picking best network checkpoint in embodiment evaluation using "train" embodiments instead')
            cfg.train.eval_best_checkpoint_split = 'train'
        if len(dataset.train_embodiment_ids) == 0 and cfg.train.eval_best_checkpoint_split == 'train':
            print(f'WARNING: since eval_best_checkpoint_split == "train", but no train embodiments were provided in the dataset, defaulting to picking best network checkpoint in embodiment evaluation using "val" embodiments instead')
            cfg.train.eval_best_checkpoint_split = 'val'

    if embodiment_evaluator:
        # We can evaluate directly on the embodiments to determine best network checkpoint
        checkpoint_to_embodiment_name_to_run_id = {}

        # helper functions for embodiment evaluation
        def select_embodiment_eval_by_split(metrics_by_name: Dict[str, float], split_embodiment_ids: Iterable[int]):
            """Selects metrics for the given embodiments"""
            result = {}
            split_embodiment_names = [dataset.embodiment_properties_by_id[i].name for i in split_embodiment_ids]

            for embodiment_name, metric in metrics_by_name.items():
                if embodiment_name in split_embodiment_names:
                    result[embodiment_name] = metric
            
            return result

        def format_embodiment_eval_metrics(eval_metrics: Dict, tag: str, baseline_metrics={}):
            """
            Returns:
            - metrics: Dict of metrics
            - average_by_split: Dict mapping from split name to average metric. Note that if `baseline_metrics` is provided, then this average is normalized with respect to the performance of the baseline
            """

            metrics = {}

            # by split metrics
            all_split_embodiment_ids = {
                'train_and_val': dataset.train_embodiment_ids.union(dataset.val_embodiment_ids),
                'all': dataset.train_embodiment_ids.union(dataset.val_embodiment_ids).union(dataset.test_embodiment_ids),
                **split_name_to_embodiment_ids
            }

            average_by_split = {}
            for split_name, split_embodiment_ids in all_split_embodiment_ids.items():
                split_metrics = select_embodiment_eval_by_split(eval_metrics, split_embodiment_ids)
                if len(split_metrics) == 0:
                    continue

                # it only makes sense to compute an average if we have metrics for every embodiment in this split. In the case of baseline metrics perhaps one of the embodiments doesn't have baseline metrics, so it wouldn't make sense to compute metrics for the split it is a part of
                if len(split_metrics) == len(split_embodiment_ids):
                    average = sum(split_metrics.values()) / len(split_metrics)
                    metrics[f'embodiment_eval/{tag}_{split_name}_average'] = average
                    average_by_split[split_name] = average

                    # if baseline metrics provided for all embodiments in the split, compute performance relative to baseline; weighted such that each embodiment contributes an equal amount to the average performance (embodiments with higher performance don't have disproportionally large impact on the overall performance)
                    has_baseline_metrics_entire_split = all([dataset.embodiment_properties_by_id[id].name in baseline_metrics for id in split_embodiment_ids])
                    if has_baseline_metrics_entire_split:
                        average_of_baseline = 0
                        for embodiment_name in split_metrics:
                            average_of_baseline += split_metrics[embodiment_name] / baseline_metrics[embodiment_name]
                        average_of_baseline /= len(split_metrics)

                        metrics[f'embodiment_eval/{tag}_{split_name}_average_of_baseline'] = average_of_baseline # performance weighted with respect to the baseline (1 would be matching baseline)
                else:
                    average = None                    

            # per embodiment metrics
            for i in range(len(dataset.embodiment_properties_by_id)):
                embodiment_name = dataset.embodiment_properties_by_id[i].name
                if embodiment_name not in eval_metrics:
                    continue

                val = eval_metrics[embodiment_name]
                metrics[f'embodiment_eval/{tag}_{embodiment_name}'] = val

                # if baseline metrics provided, compute performance relative to baseline
                if embodiment_name in baseline_metrics:
                    baseline_val = baseline_metrics[embodiment_name]
                    metrics[f'embodiment_eval/{tag}_{embodiment_name}_of_baseline'] = val / baseline_val
            
            return metrics, average_by_split
        
        def wait_for_metrics(metrics: Dict[str, AsyncResult], checkpoint: str):
            """Blocks until all metrics are available"""
            result = {}
            for embodiment_name, asyc_result in metrics.items():
                metric, run_id = asyc_result.get()
                result[embodiment_name] = metric
                if checkpoint not in checkpoint_to_embodiment_name_to_run_id:
                    checkpoint_to_embodiment_name_to_run_id[checkpoint] = {}
                checkpoint_to_embodiment_name_to_run_id[checkpoint][embodiment_name] = run_id
            
            return result

        # block until baseline metrics computed
        if cfg.train.skip_embodiment_baseline_eval:
            baseline_eval = {}
        else:
            print('Waiting for baseline metrics to complete')
            baseline_eval = wait_for_metrics(async_baseline_eval, 'baseline')
            print('Baseline metrics finished computing')
        baseline_eval_metrics_all, _ = format_embodiment_eval_metrics(baseline_eval, 'baseline')
    
        if not cfg.test_only:
            # figure out best training checkpoint
            best_eval_score = float('-inf')
            best_checkpoint_i = 0
            average_metric_by_split_by_eval_epoch = []
            for i, epoch_i in enumerate(epoch_i_to_evaluate_embodiment):
                print(f'Waiting for epoch {epoch_i} metrics to complete')
                checkpoint_embodiment_eval_metrics[i] = wait_for_metrics(checkpoint_embodiment_eval_metrics[i], data_per_epoch[epoch_i]['checkpoint_path'])
                print(f'Epoch {epoch_i} metrics finished computing')
                cur_epoch_metrics, average_metric_by_split = format_embodiment_eval_metrics(checkpoint_embodiment_eval_metrics[i], 'distill', baseline_eval)
                cur_comparison_split_metric = average_metric_by_split[cfg.train.eval_best_checkpoint_split]
                average_metric_by_split_by_eval_epoch.append(average_metric_by_split)
                if cur_comparison_split_metric > best_eval_score:
                    best_eval_score = cur_comparison_split_metric
                    best_checkpoint_i = i
                
                metrics = {
                    'embodiment_eval/epoch': epoch_i,
                    **cur_epoch_metrics,
                    **baseline_eval_metrics_all
                }

                print(f'Epoch {epoch_i} checkpoint embodiment eval metrics:')
                for k, v in metrics.items():
                    print(f'{k}: {v}')
                
                if cfg.wandb_activate:
                    wandb.log(metrics)
            
            # Compute statistics over the last epoch metrics
            final_epochs_metrics = {}
            average_metric_by_split_last_epochs = average_metric_by_split_by_eval_epoch[-cfg.train.embodiment_eval_average_last_epochs:]
            for split_name in average_metric_by_split_last_epochs[0]:
                last_epoch_values = [average_metric_by_split_for_epoch[split_name] for average_metric_by_split_for_epoch in average_metric_by_split_last_epochs]
                final_epochs_metrics[f'embodiment_eval/final_epochs_distill_{split_name}_average'] = statistics.mean(last_epoch_values)
                final_epochs_metrics[f'embodiment_eval/final_epochs_distill_{split_name}_median'] = statistics.median(last_epoch_values)
                final_epochs_metrics[f'embodiment_eval/final_epochs_distill_{split_name}_min'] = min(last_epoch_values)
                final_epochs_metrics[f'embodiment_eval/final_epochs_distill_{split_name}_max'] = max(last_epoch_values)
                final_epochs_metrics[f'embodiment_eval/final_epochs_distill_{split_name}_stdev'] = statistics.stdev(last_epoch_values)
            
            print(f'\nFinal {cfg.train.embodiment_eval_average_last_epochs} epochs average embodiment eval metrics:')
            for k, v in final_epochs_metrics.items():
                print(f'{k}: {v}')

            if cfg.wandb_activate:
                wandb.log(final_epochs_metrics)

            # evaluate best checkpoint on `test` embodiments (we have already completed the evaluation of this checkpoint on `train` and `val` embodiments)
            best_checkpoint_path = data_per_epoch[epoch_i_to_evaluate_embodiment[best_checkpoint_i]]['checkpoint_path']
            best_checkpoint_test_metrics = wait_for_metrics(embodiment_evaluator.evaluate_checkpoint(best_checkpoint_path, ['test']), best_checkpoint_path)

            embodiment_evaluator.finish_evaluation()

            # Mark which checkpoint is the best and log corresponding metrics
            best_epoch_i = epoch_i_to_evaluate_embodiment[best_checkpoint_i]
            best_checkpoint_metrics_all_splits = {**checkpoint_embodiment_eval_metrics[best_checkpoint_i], **best_checkpoint_test_metrics}
        else:
            # in test mode evaluate on all splits
            best_checkpoint_path = cfg.checkpoint
            best_checkpoint_metrics_all_splits = wait_for_metrics(embodiment_evaluator.evaluate_checkpoint(best_checkpoint_path, ['train', 'val', 'test']), best_checkpoint_path)
        
        baseline_final_metrics, _ = format_embodiment_eval_metrics(baseline_eval, 'best_baseline')
        distill_final_metrics, _ = format_embodiment_eval_metrics(best_checkpoint_metrics_all_splits, 'best_distill', baseline_eval)
        best_checkpoint_metrics = {
            **baseline_final_metrics,
            **distill_final_metrics
        }
        print('\nBest checkpoint embodiment eval metrics:')
        for k, v in best_checkpoint_metrics.items():
            print(f'{k}: {v}')

        if cfg.wandb_activate:
            wandb.log(best_checkpoint_metrics)
        
        # mark best runs
        if cfg.train.embodiment_eval_skip_mark_best_run:
            print('\nSkipping best embodiment evaluation runs as cfg.train.embodiment_eval_skip_mark_best_run is set')
        else:
            print('\nMarking best embodiment evaluation runs')
            best_checkpoint_embodiment_runs = checkpoint_to_embodiment_name_to_run_id[best_checkpoint_path] # note that this will also contain the runs for the test embodiments as well
            embodiment_evaluator.mark_best_runs(list(best_checkpoint_embodiment_runs.values()))
    elif not cfg.test_only:
        """
        No embodiment evaluation case. Use validation losses to compute best checkpoint.
        """
        best_eval_score = float('-inf')
        best_epoch_i = None
        if cfg.train.eval_best_checkpoint_split == 'train_and_val':
            comparison_splits = ['train', 'val']
        else:
            comparison_splits = [cfg.train.eval_best_checkpoint_split] # either ['train'] or ['val']

        for epoch_i, epoch_data in enumerate(data_per_epoch):
            # reweight validation losses by the number of embodiments (note this assumes examples/embodiment is balanced across embodiments in each split, which will be true if training data is balanced)
            cur_model_eval = 0
            for split_name in comparison_splits:
                split_embodiment_ids = getattr(dataset, f'{split_name}_embodiment_ids')
                if len(split_embodiment_ids) == 0:
                    continue

                cur_model_eval -= epoch_data[f'train/{split_name}_val_loss'] * len(split_embodiment_ids)

            if cur_model_eval >= best_eval_score:
                best_eval_score = cur_model_eval
                best_epoch_i = epoch_i

    if not cfg.test_only:
        """Save and load best checkpoint"""
        best_epoch_checkpoint_path = data_per_epoch[best_epoch_i]['checkpoint_path']
        print(f'Highest performing epoch was epoch {best_epoch_i}/{cfg.train.num_epochs} (note epoch 0 is before any training)')
        best_checkpoint_path = os.path.join(os.path.dirname(best_epoch_checkpoint_path), f'best.pt')
        shutil.copyfile(best_epoch_checkpoint_path, best_checkpoint_path)
        print(f'Wrote best checkpoint to {best_checkpoint_path}')
        load_checkpoint(best_checkpoint_path)
        print('FINISHED COMPUTING BEST CHECKPOINT\n')

        """Model visualization"""
        visualization_figures = vis_embodiment_transformer(f'{experiment_dir}/vis', model=model)
        if cfg.wandb_activate:
            wandb.log({f'plots/{k}': wandb.Image(v) for k, v in visualization_figures.items()})

        """Log outputs where the network didn't predict the correct class (if we are just doing self modeling)"""
        if 'policy' not in dataset.info_by_head and 'policy_forward_kinematics' not in dataset.info_by_head:
            print(f'Logging misclassifications')
            compute_dataloader_loss(dataset.all_dataloader, 'all', print_misclassified=True)
    else:
        best_checkpoint_path = cfg.checkpoint
    
    print(f'--- Finished distillation with logging directory: {experiment_dir} ---')

    if cfg.wandb_activate:
        wandb.finish()

    return best_checkpoint_path

if __name__ == "__main__":
    main()
