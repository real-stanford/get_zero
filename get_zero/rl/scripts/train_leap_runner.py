"""
Manages RL training runs across the different LEAP hardware configurations and records the overall performance. It starts new runs with new seeds as needed based on their performance. For runs that perform better than the required metric, the policy is rolled out and state logs with observations/actions are generated.
"""

import wandb
import os
from wandb.apis.public import Run
from typing import Dict, List
from argparse import ArgumentParser
import multiprocessing
from multiprocessing import Pool, Manager
from time import sleep
import traceback
import hydra
import random
import math
import signal
from get_zero.rl.train import rl_runner

from rich.console import Console
console = Console(highlight=False)
print = console.print

NUM_EMBODIMENTS = 633

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--max_metric', type=int, default=30, help='min metric needed to consider training success') # metric is average time to complete full rotation
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--runs_per_gpu', type=int, default=1)
    parser.add_argument('--status_only', action='store_true', help='set to just display current best status for embodiments')
    parser.add_argument('--log_only', action='store_true', help='whether to only log what experiments will be run instead of actually running them')
    parser.add_argument('--wandb_tag', default='leap_train_get_zero', help='tag used to identify training runs')
    parser.add_argument('--wandb_proj', default='get_zero', help='project name to use on W&B')
    parser.add_argument('--max_tries', type=int, default=5, help='max amount of seeds to try to reach --max_metric before giving up')
    parser.add_argument('--random_ordering', action='store_true', help='whether to process embodiments in random ordering. If not passed, then does increasing order')
    parser.add_argument('--embodiment_start', default='001', help='starting embodiment name to run on')
    parser.add_argument('--embodiment_end', default=f'{NUM_EMBODIMENTS:03}', help='ending embodiment name to run on')
    parser.add_argument('--debug', action='store_true', help='enables debug mode which allows you to use `--debug_overrides` and add `_debug` to the `wandb_tag`')
    parser.add_argument('--debug_overrides', default=[], nargs='+', help='additional parameters to add to the RL config in the format `key1=val1 key2=val2 ...`. Must supply --debug for these to be used')
    parser.add_argument('--stages', default=['train', 'rollout'], nargs='+', help='Selects whether train or rollout happens or both.')
    return parser.parse_args()

class RunManager:
    def __init__(self):
        self.running_state = {}

    def start(self, name):
        self.running_state[name] = True

    def finish(self, name):
        self.running_state[name] = False
        
    def is_running(self, name):
        return self.running_state.get(name, False)

def run_to_metric(run: Run) -> float:
    """Metric used in average yaw radians per second"""
    if run is None or run.state != 'finished':
        return float('inf')
    
    metric_key = 'yaw_finite_diff_cumulative'
    metric_key_deprecated = 'yaw_finite_diff_cumulative/iter'
    if metric_key in run.summary or metric_key_deprecated in run.summary:
        if 'global_counter' in run.summary: # some earlier runs didn't log this
            if run.summary['global_counter'] != run.config['test_steps']:
                # for some reason some runs were marked as `finished` by W&B, but stopped part way through. This is a sanity check that the final counter state matches the number of test steps we expect to perform, so that we only use stats from runs that are truly completed.
                return float('inf')
        
        steps_per_second = 20 # 20Hz control frequency
        yaw_cumulative = run.summary[metric_key] if metric_key in run.summary else run.summary[metric_key_deprecated]
        average_yaw_velocity_per_step = yaw_cumulative / run.config['test_steps']
        average_yaw_velocity_per_second = average_yaw_velocity_per_step * steps_per_second
        time_to_complete_one_full_rotation = 2 * math.pi / average_yaw_velocity_per_second
        
        return time_to_complete_one_full_rotation

    return float('inf') # run likely marked as finished, but didn't finish properly since missing the expected keys

def get_status_by_embodiment(wandb_tag, enable_logging=True, embodiment_start=None, embodiment_end=None, max_metric=None, wandb_proj='get_zero') -> Dict[str, Run]:
    """Returns mapping from embodiment name to best run"""
    def output_format(msg, properties):
        if len(properties) == 0:
            return msg
        else:
            return f'[{" ".join(properties)}]{msg}[/{" ".join(properties)}]'

    def format_state(state):
        state_to_color = {
            'running': 'purple4',
            'finished': 'green'
        }

        if state in state_to_color:
            color = state_to_color[state]
        else:
            color = 'red'

        return output_format(state, [color])
    
    def format_metric(metric):
        properties = []
        if max_metric:
            if metric < max_metric and metric > 0:
                properties.append('green')
            else:
                properties.append('red')
        
        return output_format(f'{metric:.3f}', properties)
    
    def format_bool(value):
        return output_format(f'{value}', ['green' if value else 'red'])
        
    api = wandb.Api()
    runs: List[Run] = list(api.runs(wandb_proj, {"tags": wandb_tag}))
    rollout_runs: List[Run] = list(api.runs(wandb_proj, {"tags": f'rollout_{wandb_tag}'}))
    if enable_logging:
        print('--- CURRENT STATUS BY EMBODIMENT ---')
        print(f'Found {len(runs)} total training runs. Goal metric is below {max_metric} seconds per full 2pi rotation.')
        print(f'Found {len(rollout_runs)} total rollout runs.')

    data_by_embodiment: Dict = {} # map from embodiment name to Dict with format {'best_run': Run, 'all_runs': List[Run]}
    logs_by_embodiment: Dict = {} # map from embodiment name to List[str] of messages about that embodiment
    seeds_by_embodiment: Dict = {}

    def log(embodiment_name, run_name, message):
        if embodiment_name not in logs_by_embodiment:
            logs_by_embodiment[embodiment_name] = []

        logs_by_embodiment[embodiment_name].append(f'[{run_name}] {message}')

    # sort runs by increasing run name (chronologically)
    runs.sort(key=lambda run: run.name)

    # sort rollout runs with most recent runs first (since distill will read most recent state logs)
    rollout_runs.sort(key=lambda run: run.name, reverse=True)

    # Figure out best run and all runs per embodiment
    for run in runs:
        config = run.config
        name = run.name

        if 'task' not in config:
            # run didn't properly sync config (probably killed very early on)
            continue

        embodiment_name = os.path.basename(config['task']['env']['asset']['handAsset']).replace('.urdf', '')
        metric = run_to_metric(run)
        seed = config['seed']

        # add run to list of all runs
        if embodiment_name not in data_by_embodiment and (run.state == 'finished' or run.state == 'running'):
            data_by_embodiment[embodiment_name] = {'best_run': None, 'all_runs': [], 'is_rolled_out': False}   

        if run.state != 'finished':
            log(embodiment_name, name, f'Found run with status: {format_state(run.state)}')

        if run.state == 'finished':
            cur_data = data_by_embodiment[embodiment_name]
            cur_data['all_runs'].append(run)
            prior_best_metric = run_to_metric(cur_data['best_run'])
            if (cur_data['best_run'] is None) or (run_to_metric(run) < prior_best_metric and run_to_metric(run) > 0) or (prior_best_metric < 0 and run_to_metric(run) > prior_best_metric):
                cur_data['best_run'] = run
                log(embodiment_name, name, f'Found run with better metric {format_metric(metric)} < {format_metric(prior_best_metric)}')
            else:
                log(embodiment_name, name, f'Found run with worse metric {format_metric(metric)} > {format_metric(prior_best_metric)}')
        elif run.state == 'running':
            cur_data = data_by_embodiment[embodiment_name]
            cur_data['all_runs'].append(run)

        # add run seed to list of seeds
        if embodiment_name not in seeds_by_embodiment:
            seeds_by_embodiment[embodiment_name] = []

        if run.state == 'finished' or run.state == 'running':
            seeds_by_embodiment[embodiment_name].append(seed)

    # handle rollout runs
    rollout_runs = [run for run in rollout_runs if run.state == 'finished' or run.state == 'running']
    best_rollout_tag = 'rollout_best_leap_train'
    for rollout_run in rollout_runs:
        config = rollout_run.config

        if 'task' not in config:
            # run didn't properly sync config (probably killed very early on)
            continue

        embodiment_name = os.path.basename(config['task']['env']['asset']['handAsset']).replace('.urdf', '')
        cur_data = data_by_embodiment[embodiment_name]

        if cur_data['best_run'].name not in config['checkpoint']:
            continue

        if cur_data['is_rolled_out']:
            # if a rollout out run already found, then ignore additional runs

            # untag this run if it is tagged
            if best_rollout_tag in rollout_run.tags:
                rollout_run.tags.remove(best_rollout_tag)
                rollout_run.update()
                print(f'For run {rollout_run.id} and embodiment {embodiment_name} remove tag `{best_rollout_tag}` on WandB')
            continue
    
        rollout_found_locally = has_rollout_finished(cur_data['best_run'], embodiment_name)
        if rollout_run.state == 'finished' and rollout_found_locally:
            cur_data['is_rolled_out'] = True
            cur_data['best_run_rollout'] = rollout_run
            cur_data['rollout_state'] = 'finished'

            # tag the best performing runs with W&B tag
            if best_rollout_tag not in rollout_run.tags:
                rollout_run.tags.append(best_rollout_tag)
                rollout_run.update()
                print(f'For run {rollout_run.id} and embodiment {embodiment_name} add tag `{best_rollout_tag}` on WandB')

        elif rollout_run.state != 'finished':
            cur_data['is_rolled_out'] = False
            cur_data['rollout_state'] = rollout_run.state

    if enable_logging:
        embodiment_names_sorted = sorted(logs_by_embodiment.keys())
        for embodiment_name in embodiment_names_sorted:
            messages = logs_by_embodiment[embodiment_name]

            if embodiment_name in data_by_embodiment and data_by_embodiment[embodiment_name]["best_run"] is not None:
                best_run = data_by_embodiment[embodiment_name]["best_run"]
                is_rolled_out = data_by_embodiment[embodiment_name]['is_rolled_out']
            else:
                best_run = None
                is_rolled_out = False

            if is_rolled_out:
                rollout_run = data_by_embodiment[embodiment_name]['best_run_rollout']
                rollout_message = f', metric: {format_metric(run_to_metric(rollout_run))}, ID: {rollout_run.name}'
            else:
                if embodiment_name in data_by_embodiment and 'rollout_state' in data_by_embodiment[embodiment_name]:
                    rollout_message = f', state={format_state(data_by_embodiment[embodiment_name]["rollout_state"])}'
                else:
                    rollout_message = ''
            
            print(output_format(f'Messages for embodiment {output_format(embodiment_name, ["light_coral"])} (Best: {format_metric(run_to_metric(best_run))} [ID: {best_run.name if best_run else "N/A"}]) (Rolled out? {format_bool(is_rolled_out)}{rollout_message}) (Seeds: {sorted(seeds_by_embodiment.get(embodiment_name, []))})', ['deep_sky_blue1']))
            for message in messages:
                print(message)

            print()

    # add reminaing embodiments that currently don't have any results
    for embodiment_i in range(NUM_EMBODIMENTS):
        embodiment_name = f'{embodiment_i + 1:03}'
        if embodiment_name not in data_by_embodiment:
            data_by_embodiment[embodiment_name] = {'best_run': None, 'all_runs': []}

    # only select embodiments that are requested
    if embodiment_start is not None:
        for embodiment_i in range(1, int(embodiment_start)):
            embodiment_name = f'{embodiment_i:03}'
            del data_by_embodiment[embodiment_name]
    if embodiment_end is not None:
        for embodiment_i in range(int(embodiment_end) + 1, NUM_EMBODIMENTS + 1):
            embodiment_name = f'{embodiment_i:03}'
            del data_by_embodiment[embodiment_name]

    return data_by_embodiment

def next_run_generator(run_manager: RunManager, gpu_queue, script_args):
    while True:
        print('Waiting for GPU from the queue')
        gpu = gpu_queue.get() # wait until GPU is available before proceeeding
        print(f'Retrieved GPU{gpu} from the queue')
        print('Calling `get_status_by_embodiment`')
        status = get_status_by_embodiment(script_args.wandb_tag, enable_logging=False, embodiment_start=script_args.embodiment_start, embodiment_end=script_args.embodiment_end, max_metric=args.max_metric, wandb_proj=args.wandb_proj)
        embodiment_names = list(status.keys())
        if script_args.random_ordering:
            random.shuffle(embodiment_names)
        else:
            embodiment_names.sort()
        found_something = False
        for embodiment_name in embodiment_names:
            cur_status = status[embodiment_name]
            cur_best_metric = run_to_metric(cur_status['best_run'])
            train_used_seeds = [run.config['seed'] for run in status[embodiment_name]['all_runs']]
            any_train_runs_in_progress = any([run.state == 'running' for run in status[embodiment_name]['all_runs']])
            if cur_best_metric > script_args.max_metric and not run_manager.is_running(embodiment_name) and len(train_used_seeds) < script_args.max_tries and not any_train_runs_in_progress and 'train' in script_args.stages:
                # Need to perform a new training run
                # pick a new seed to run
                seed = random.randint(0, 1000)
                while seed in train_used_seeds:
                    seed = random.randint(0, 1000)

                # start run
                run_manager.start(embodiment_name)
                yield (embodiment_name, gpu, (seed,), None, script_args)
                found_something = True
                break
            elif (cur_best_metric < script_args.max_metric and cur_best_metric > 0) and not run_manager.is_running(embodiment_name) and not cur_status['is_rolled_out'] and 'rollout' in script_args.stages:
                # make sure there is not a run in progress
                if 'rollout_state' in cur_status and cur_status['rollout_state'] == 'running':
                    continue

                # Need to perform state log rollout
                run_manager.start(embodiment_name)
                yield (embodiment_name, gpu, None, (cur_status['best_run'].name,), script_args)
                found_something = True
                break

        if not found_something:
            break

def has_rollout_finished(run, embodiment_name):
    """Rolllout is considered finished if the state log corresponding to the run is present and it's the most recent run"""
    if run is None:
        return False
    state_log_dir = os.path.join('..', 'state_logs', 'LeapHandRot', embodiment_name)
    if not os.path.exists(state_log_dir):
        return False

    existing_state_logs = os.listdir(state_log_dir)
    existing_state_logs.sort(reverse=True) # most recent first

    if len(existing_state_logs) == 0:
        return False

    result = run.name in existing_state_logs[0] # run should be most recent state log
    return result

def execute_run(embodiment_name, gpu, train_args, rollout_args, script_args):
    args = embodiment_name, gpu, train_args, rollout_args, script_args
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')) # change to root of `rl` folder as `train.py` assumes calls are made from that folder

    try:
        if train_args:
            seed, = train_args

            debug_overrides = script_args.debug_overrides if script_args.debug else []
            wandb_tag = f'{script_args.wandb_tag}_debug' if script_args.debug else script_args.wandb_tag

            overrides = [
                'task=LeapHandRot',
                f'embodiment=LeapHand/generated/{embodiment_name}',
                f'gpu={gpu}',
                f'seed={seed}',
                f'wandb_tags=[{wandb_tag}]',
                'record_video=False'
            ] + debug_overrides
            print(f'--- Starting train job with embodiment {embodiment_name} and overrides ', end='')
            print(overrides) # two separate prints to fix weird issue where wandb_tag displayed incorrect with f formatting
            with hydra.initialize(version_base="1.2", config_path="../cfg"):
                rl_cfg = hydra.compose(config_name="config", overrides=overrides)

            if not script_args.log_only:
                start_runner = rl_runner(rl_cfg, runner_only=True)
                start_runner(True, is_last_run=False)
                start_runner(False, is_last_run=True)
        else:
            checkpoint_run_id, = rollout_args
            print(f'--- Starting rollout process with embodiment {embodiment_name} and GPU {gpu}')

            # find checkpoint dir (due to naming change there are two possible names. Option 1 is the newer naming convention)
            checkpoint_dir = f'runs/{checkpoint_run_id}/nn'
            checkpoint_option1 = os.path.join(checkpoint_dir, 'LeapHand.pth')
            checkpoint_option2 = os.path.join(checkpoint_dir, 'LeapHandRot.pth')
            if os.path.exists(checkpoint_option1):
                checkpoint = checkpoint_option1
            elif os.path.exists(checkpoint_option2):
                checkpoint = checkpoint_option2
            else:
                assert False, f'Could not find checkpoint to rollout for the given run {checkpoint_run_id}'

            debug_overrides = script_args.debug_overrides if script_args.debug else []
            wandb_tag = f'{script_args.wandb_tag}_debug' if script_args.debug else script_args.wandb_tag

            overrides = [
                'task=LeapHandRot',
                f'embodiment=LeapHand/generated/{embodiment_name}',
                f'gpu={gpu}',
                f'wandb_tags=[rollout_{wandb_tag}]',
                'test=True',
                f'checkpoint={checkpoint}',
                'task.env.logStateInTest=True',
                f'task.env.logStateSuffix=trainrun-{checkpoint_run_id}',
                'test_steps=1000',
                'num_envs=5000'
            ] + debug_overrides
            print(f'Starting rollout job with embodiment {embodiment_name} and overrides ', end='')
            print(overrides) # two separate prints to fix weird issue where wandb_tag displayed incorrect with f formatting
            with hydra.initialize(version_base="1.2", config_path="../cfg"):
                rl_cfg = hydra.compose(config_name="config", overrides=overrides)

            if not script_args.log_only:
                start_runner = rl_runner(rl_cfg, runner_only=True)
                start_runner(False, is_last_run=True)

    except Exception as e:
        # Intercept the exception so we can log it, then re-raise the exception
        print(f'Task `{args}` ended with exception:')
        traceback.print_exc()
        raise e

    print(f'Successfully finished run on embodiment {embodiment_name} on GPU {gpu}')
    return embodiment_name, gpu

if __name__ == '__main__':
    args = parse_args()

    if args.status_only:
        get_status_by_embodiment(args.wandb_tag, enable_logging=True, embodiment_start=args.embodiment_start, embodiment_end=args.embodiment_end, max_metric=args.max_metric, wandb_proj=args.wandb_proj)
        exit()

    multiprocessing.set_start_method('spawn') # create new processes that are not forked, but contain minimal resources needed to run these experiments

    multiprocess_manager = Manager()
    run_manager = RunManager()
    num_resources = len(args.gpus) * args.runs_per_gpu
    gpu_queue = multiprocess_manager.Queue(num_resources)
    for gpu in args.gpus:
        for gpu_run in [gpu] * args.runs_per_gpu:
            gpu_queue.put(gpu_run)

    total_completed = 0

    def job_finish_callback(result):
        global total_completed
        embodiment_name, gpu = result
        total_completed += 1
        print(f'--- Completed a total of {total_completed} runs ---')
        run_manager.finish(embodiment_name)
        gpu_queue.put(gpu)
        print(f'Returned GPU{gpu} to the queue')

    def job_fail_callback(exception: Exception):
        print('Exiting as a result of the exception')
        os.kill(os.getpid(), signal.SIGINT)
        
    with Pool(num_resources, maxtasksperchild=1) as p:
        for run_args in next_run_generator(run_manager, gpu_queue, args):
            p.apply_async(execute_run, run_args, callback=job_finish_callback, error_callback=job_fail_callback)            

    multiprocess_manager.shutdown()
    
    print('train_leap_runner.py finished all requested runs!')
