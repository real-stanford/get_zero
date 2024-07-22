"""
Performs policy distillation across many embodiments and uses Isaac Gym to evaluate the performance of the distilled policies during policy training. The format of args to this script are:
`python distill.py <args for this script> -- <hydra args for distill script> -- <hydra args for rl setup> -- <hydra args for both distill and rl>`
`python distill.py <args for this script> -- <hydra args for distill script> -- <hydra args for rl setup>`
`python distill.py <args for this script> -- <hydra args for distill script>`
`python distill.py <args for this script>`
The `--` acts as the separator between sets of args.
"""
import isaacgym # added to ensure it's imported before torch as required

import hydra
from typing import List, Dict
import sys
import os
import yaml
from multiprocessing import Pool, Manager
from multiprocessing.pool import AsyncResult
import multiprocessing
from tbparse import SummaryReader
from argparse import ArgumentParser
from datetime import datetime
import traceback
import signal
from omegaconf import OmegaConf, DictConfig, open_dict

from get_zero.rl.train import rl_runner
from get_zero.distill.utils.embodiment_util import EmbodimentEvaluator, EmbodimentProperties
from get_zero.distill.distill import distill

def parse_args(args):
    parser = ArgumentParser()
    parser.add_argument('--gpus', type=int, nargs='+', default=[0], help='GPUs to run RL evaluation on. First GPU in list is used to run network distill training.')
    parser.add_argument('--no_rl_runs_on_distill_gpu', action='store_true', help='if specified, do not use the first gpu in --gpus to do any RL rollouts and only use the latter GPUs. requies that at least two GPUs are specified')
    parser.add_argument('--rl_runs_per_gpu_eval', type=int, default=8, help='max concurrent runs on GPU during evaluation')
    parser.add_argument('--rl_runs_per_gpu_best_run', type=int, default=1, help='max concurrent runs on GPU during best runs video recording')

    return parser.parse_args(args)

def start_rl_experiment(rl_cfg, task, gpu_queue: multiprocessing.Queue):
    """Run RL experiment using GPU provided by the queue and then returns (eval metric, run ID)"""
    gpu = gpu_queue.get(block=False) # if the thread was started there should be a GPU available
    print(f'Starting RL run with GPU{gpu}')

    start_runner = rl_runner(rl_cfg, runner_only=True)
    run_dir = start_runner(False, is_last_run=True)
    eval_metric = get_eval_metric_from_run_dir(task, run_dir)
    experiment_id = f'uid_{os.path.basename(run_dir)}'

    gpu_queue.put(gpu)
    
    return eval_metric, experiment_id

def get_embodiment_rl_config_path(task: str, embodiment_name: str):
    """Converts from an RL task name and the name of an embodiment into the specific embodiment config path used to run the RL experiments with that embodiment"""
    if task == 'LeapHandRot':
        if embodiment_name.isdigit():
            # generated embodiment of the form of integer `XYZ`
            return f'LeapHand/generated/{embodiment_name}'
        elif embodiment_name == 'original':
            return 'LeapHand/OrigRepo'
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    
def get_eval_metric_from_run_dir(task, run_dir):
    """Reads eval metric from the log dir of the experiment"""
    if task == 'LeapHandRot':
        # eval metric is average yaw rotation rate (cumulative yaw rotation / num_steps)
        summaries_dir = os.path.join(run_dir, 'summaries')
        df = SummaryReader(summaries_dir).scalars

        yaw_values = df[df['tag'] == 'yaw_finite_diff_cumulative/frame']['value']
        final_cumulative_yaw = yaw_values.iloc[-1]
        global_step_values = df[df['tag'] == 'global_counter/frame']['value']
        num_steps = global_step_values.iloc[-1]
        rad_per_step = final_cumulative_yaw / num_steps
        steps_per_second = 20 # 120hz simulation frequency with control every 6 steps -> 20Hz control frequency
        rad_per_second = rad_per_step * steps_per_second
        seconds_of_runtime = num_steps / steps_per_second
        print(f'{run_dir} Able to rotate a total of {final_cumulative_yaw} radians over {seconds_of_runtime} seconds leading to average of {rad_per_second} rad/sec')
        return rad_per_second
    else:
        raise NotImplementedError
    
class RLEmbodimentEvaluator(EmbodimentEvaluator):
    def __init__(self, custom_rl_config_overrides, rl_gpus, rl_runs_per_gpu_eval, rl_runs_per_gpu_best_run):
        self.custom_rl_config_overrides = custom_rl_config_overrides
        self.rl_gpus = rl_gpus
        self.rl_runs_per_gpu_eval = rl_runs_per_gpu_eval
        self.rl_runs_per_gpu_best_run = rl_runs_per_gpu_best_run
        self.num_experiments = 0
        self.num_completed = 0

    def _init_multiprocessing_resources(self, runs_per_gpu: int):
        self.multiprocess_manager = Manager()
        self.gpu_queue = self.multiprocess_manager.Queue(len(self.rl_gpus) * runs_per_gpu)
        self.pool = Pool(len(self.rl_gpus) * runs_per_gpu, maxtasksperchild=1)
        for rl_gpu in self.rl_gpus:
            for gpu in [rl_gpu] * runs_per_gpu:
                self.gpu_queue.put(gpu)
    
    def _cleanup_multiprocessing_resources(self):
        self.pool.close()
        self.multiprocess_manager.shutdown()
    
    def prepare_evaluation(self, distill_run_dir: str, task: str, embodiment_properties_by_id: List[EmbodimentProperties], model_cfg: DictConfig, tokenization_cfg: DictConfig, embodiment_name_to_splits: Dict[str, List[str]], additional_tags: List[str]):
        self.distill_run_dir = distill_run_dir
        self.task = task
        self.embodiment_properties_by_id = embodiment_properties_by_id
        self.embodiment_name_to_splits = embodiment_name_to_splits
        self.additional_tags = additional_tags

        self._init_multiprocessing_resources(self.rl_runs_per_gpu_eval)

        # Save the model_configs and tokenization_configs to a file, so that they can be loaded into the RL setup. This requires adding the run directory as an additional hydra search path and then loading the saved model yaml configuration as the network parameters
        experiment_config_dir = os.path.join(self.distill_run_dir, 'configs')
        os.makedirs(experiment_config_dir)
        search_path_override = f'hydra.searchpath=[file://{self.distill_run_dir}]'
        # Model config
        model_config_yaml_path = os.path.join(experiment_config_dir, 'Model.yaml')
        with open(model_config_yaml_path, 'w') as f:
            yaml.safe_dump(OmegaConf.to_container(model_cfg, resolve=True), f)
        print(f'Wrote model config to {model_config_yaml_path}')
        model_config_override = '+configs@train.params.network.actor=Model'
        # Tokenization config
        tokenization_config_yaml_path = os.path.join(experiment_config_dir, 'Tokenization.yaml')
        with open(tokenization_config_yaml_path, 'w') as f:
            yaml.safe_dump(OmegaConf.to_container(tokenization_cfg, resolve=True), f)
        print(f'Wrote tokenization config to {tokenization_config_yaml_path}')
        tokenization_config_override = f'+configs@task.env.tokenization=Tokenization'

        """Prepare configuration for RL runs"""
        # configs shared across all RL runs
        shared_rl_overrides = [ # can safely overwrite these values over command line
            'test=True',
            'num_envs=100',
            'test_steps=500',
            'seed=12321', # seed shouldn't match seed used when generating RL state logs to train the multi embodiment policy, but should be shared among all RL rollouts, including baselines
            'record_video=False', # don't log videos because it slows down execution and videos aren't really needed except for the best checkpoint (see `mark_best_runs` for video logging that happens on best runs)
            'wandb_activate=False' # don't log to W&B because of rate limiting issues and it's not critical to see upload metrics for checkpoints that are not the best checkpoint (best checkpoint will have runs uploaded to W&B)
        ] + self.custom_rl_config_overrides

        # overrides specific to runs on the baseline policy
        self.baseline_rl_overrides = ['task.env.logStateInTest=False', 'wandb_tags=[]'] + shared_rl_overrides
        
        # overrides specific to runs on the distilled policy
        self.distill_rl_overrides = [
            f'task={task}',
            'model=generic/TransformerBase',
            'wandb_log_freq=10', # log less frequently due to WandB rate limits when doing many parallel experiments; also set separately for baseline runs futher below in this script
            search_path_override,
            model_config_override,
            tokenization_config_override
        ] + shared_rl_overrides

        # overrides specific to best run rollouts
        self.best_run_rl_overrides = ['record_video=true', 'wandb_activate=true', 'num_envs=1'] + self.custom_rl_config_overrides # need to put `custom_rl_config_overrides` again in case we want to override any items in the preceeding list

    def _launch_experiments(self, rl_cfgs, embodiment_names, blocking=False) -> Dict[str, AsyncResult]:
        experiment_args_lst = []
        for run_i in range(len(rl_cfgs)):
            gpu_index = self.num_experiments % len(self.rl_gpus)
            self.num_experiments += 1
            rl_cfg = rl_cfgs[run_i]
            rl_cfg.gpu = self.rl_gpus[gpu_index]

            # Add a W&B tag indicating which split the embodiment is a part of
            OmegaConf.set_struct(rl_cfg, True)
            with open_dict(rl_cfg):
                for split_name in self.embodiment_name_to_splits[embodiment_names[run_i]]:
                    rl_cfg.wandb_tags.append(f'{split_name}_embodiment')
                rl_cfg.wandb_tags.extend(self.additional_tags)

            experiment_args_lst.append((rl_cfg, self.task, self.gpu_queue))

        def job_completion(args):
            self.num_completed += 1
            print(f'COMPLETED {self.num_completed} OF {self.num_experiments} EMBODIMENT EVALUATIONS')
        
        def job_failure(err):
            print('RL run failed with exception:')
            print("".join(traceback.format_exception(type(err), err, err.__traceback__)))
            self._cleanup_multiprocessing_resources()
            print('Ending distill process due to exception')
            os.kill(os.getpid(), signal.SIGINT) # this function is run in a thread, so need to exit main process since 'exit()' will just quit the thread

        # launch jobs for embodiment. It's important that these jobs are new processes because WandB needs separate processes to track runs and Isaac Gym also can only be instantiated once per process
        # note that maxtasksperchild is 1 since Isaac Gym has trouble reinitializing if the same process is used, so we should just make new processes after each RL run rather than reusing processes in the Pool
        # it's also important that chunksize=1 is set if using `map` or `imap` so that all workers in a set of processes must finish before moving onto the next set of processes. This is because for processes getting reused (even though maxtasksperchild=1) since tasks were getting bundled into one task and that would cause issues reinitializing Isaac Gym.
        run_async_results = []
        for experiment_args in experiment_args_lst:
            cur_async_result = self.pool.apply_async(start_rl_experiment, experiment_args, callback=job_completion, error_callback=job_failure)
            run_async_results.append(cur_async_result)

        # collect and aggregate eval metrics
        metrics_by_embodiment: Dict[str, AsyncResult] = {}
        for run_i in range(len(run_async_results)):
            cur_run_result = run_async_results[run_i]
            if blocking:
                cur_run_result = cur_run_result.get()

            metrics_by_embodiment[embodiment_names[run_i]] = cur_run_result
        
        return metrics_by_embodiment 

    def evaluate_checkpoint(self, checkpoint: str, split_names: List[str]) -> Dict[str, AsyncResult]:
        rl_cfgs = []
        embodiment_names = []
        for embodiment_properties in self.embodiment_properties_by_id:
            # only run on the embodiments of the requested split
            if not any([split_name in split_names for split_name in self.embodiment_name_to_splits[embodiment_properties.name]]):
                continue

            embodiment_name = embodiment_properties.name
            embodiment_config_path = get_embodiment_rl_config_path(self.task, embodiment_name)

            overrides = self.distill_rl_overrides + [
                f'embodiment={embodiment_config_path}',
                f'checkpoint={checkpoint}'
            ]

            with hydra.initialize(version_base="1.2", config_path="./cfg"):
                rl_cfg = hydra.compose(config_name="config", overrides=overrides)

            embodiment_names.append(embodiment_name)
            rl_cfgs.append(rl_cfg)

        return self._launch_experiments(rl_cfgs, embodiment_names)
    
    def evaluate_baseline(self) -> Dict[str, AsyncResult]:
        rl_cfgs = []
        embodiment_names = []
        for embodiment_properties in self.embodiment_properties_by_id:
            # embodiments that have baseline have `experiment_dir` entry in the metadata. Skip the ones that do not have this.
            if not embodiment_properties.metadata or 'experiment_dir' not in embodiment_properties.metadata:
                continue

            experiment_dir = embodiment_properties.metadata['experiment_dir']

            # TODO: the idea here is to run the baseline with the exact RL config used when the baseline was generated. This is nice in that we can ensure a consistent baseline, but it is fairly problematic since we are often changing and adding new config values, which causes lots of compatability issues. Perhaps we should simply store the network type and the checkpoint and use those as overrides to the base config.
            with hydra.initialize(version_base="1.2", config_path=experiment_dir):
                rl_cfg = hydra.compose(config_name="config", overrides=self.baseline_rl_overrides)
            assert rl_cfg.checkpoint, 'the state logs should have been generated from a run that has checkpoint set'

            # hack fix to override 'wandb_log_freq' even if the parameter was initially missing from the training runs (since it was added later)
            OmegaConf.set_struct(rl_cfg, True)
            with open_dict(rl_cfg):
                rl_cfg.wandb_log_freq = 10

            embodiment_names.append(embodiment_properties.name)
            rl_cfgs.append(rl_cfg)

        return self._launch_experiments(rl_cfgs, embodiment_names)
    
    def mark_best_runs(self, run_ids: List[str]) -> None:
        """
        Rerun the the given runs with the exact same configuration, except enable video recording and W&B logging and add 'best_rl' as a W&B tag. This is more computationally expensive, so potentially uses a fewer number of concurrent runs (self.rl_runs_per_gpu_best_run).
        """
        self._init_multiprocessing_resources(self.rl_runs_per_gpu_best_run)

        rl_cfgs = []
        embodiment_names = []
        for run_id in run_ids:
            run_id = run_id[len('uid_'):] # remove 'uid_' prefix from run name
            experiment_dir = os.path.join('runs', run_id)
            with hydra.initialize(version_base="1.2", config_path=experiment_dir):
                rl_cfg = hydra.compose(config_name="config", overrides=self.best_run_rl_overrides)

            # Add 'best_rl' as an additional W&B tag
            OmegaConf.set_struct(rl_cfg, True)
            with open_dict(rl_cfg):
                rl_cfg.wandb_tags.append('best_rl')

            embodiment_names.append(rl_cfg.task.env.embodiment_name)
            rl_cfgs.append(rl_cfg)

        metrics = self._launch_experiments(rl_cfgs, embodiment_names, blocking=True)
        self._cleanup_multiprocessing_resources()

        print('\nLogging metrics for best runs:')
        for embodiment_name, (metric, run_id) in metrics.items():
            print(f'{embodiment_name} [{run_id}]: {metric}')
        
    def finish_evaluation(self):
        self._cleanup_multiprocessing_resources()

def run_distill(this_script_args, rl_config_overrides, distill_config_overrides):
    with hydra.initialize(version_base="1.2", config_path="../distill/cfg"):
        distill_cfg = hydra.compose(config_name="config", overrides=distill_config_overrides)

    rl_gpus = this_script_args.gpus
    if this_script_args.no_rl_runs_on_distill_gpu: # if requested, don't do RL evaluation on the GPU used for distill training
        assert len(rl_gpus) >= 2, 'must have at least two GPUs specified if using --no_rl_runs_on_distill_gpu'
        rl_gpus = rl_gpus[1:]
    
    embodiment_evaluator = RLEmbodimentEvaluator(rl_config_overrides, this_script_args.gpus, this_script_args.rl_runs_per_gpu_eval, this_script_args.rl_runs_per_gpu_best_run)
    distill(distill_cfg, embodiment_evaluator)
        
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn') # create new processes that are not forked, but contain minimal resources needed to run these experiments

    """Load arguments for this script, the distill run and the RL run"""
    args = sys.argv[1:]
    partitions = []
    while '--' in args:
        loc = args.index('--')
        cur_args = args[:loc]
        partitions.append(cur_args)
        args = args[loc + 1:]
    
    if len(args) > 0:
        partitions.append(args)

    this_script_args = []
    distill_args = []
    rl_args = []

    if len(partitions) >= 1:
        this_script_args = partitions[0]
    if len(partitions) >= 2:
        distill_args = partitions[1]
    if len(partitions) >= 3:
        rl_args = partitions[2]
    if len(partitions) >= 4:
        distill_args.extend(partitions[3])
        rl_args.extend(partitions[3])
    assert len(partitions) <= 4

    print(f'This script args: {this_script_args}')
    print(f'Distill command line args: {distill_args}')
    print(f'RL command line args: {rl_args}')

    # Create a wandb group for all of the runs
    wandb_group_arg = f'wandb_group=rl_distill_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")}'
    distill_args.append(wandb_group_arg)
    rl_args.append(wandb_group_arg)

    this_script_args = parse_args(this_script_args)

    # use the first GPU specified for distill training
    distill_args.append(f'gpu={this_script_args.gpus[0]}')

    """Launch run"""
    run_distill(this_script_args, rl_args, distill_args)
