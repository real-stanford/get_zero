"""
Collects both model checkpoints and state logs (demonstration data from those checkpoints) into a dataset.
"""

from train_leap_runner import get_status_by_embodiment, run_to_metric
from gen_zero_shot_state_log import gen_zero_shot_state_log
from argparse import ArgumentParser
import os
import shutil
from glob import glob

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--wandb_proj', default='get_zero', help='project name to use on W&B')
    parser.add_argument('--wandb_tag', default='leap_train_get_zero', help='tag to use on W&B')
    parser.add_argument('--max_metric', type=int, default=30, help='min metric needed to consider training success') # metric is average time to complete full rotation
    parser.add_argument('--dataset_version', type=int, default=1, help='version tag for the dataset') # metric is average time to complete full rotation
    parser.add_argument('--debug', action='store_true', help='only put demonstration data for the first embodiment to debug faster')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    print('loading status from W&B... (potentially slow)')

    data_by_embodiment = get_status_by_embodiment(args.wandb_tag, max_metric=args.max_metric, wandb_proj=args.wandb_proj)

    os.makedirs('tmp', exist_ok=True)
    out_path = os.path.join('tmp', 'get_zero_dataset')
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    out_zip_path = os.path.join('tmp', f'get_zero_dataset_v{args.dataset_version}.zip')
    if os.path.exists(out_zip_path):
        os.remove(out_zip_path)
    out_checkpoint_dir = os.path.join(out_path, 'rl_expert_checkpoints')
    out_state_log_dir = os.path.join(out_path, 'state_logs')
    out_run_dir = os.path.join(out_path, 'runs')
    os.makedirs(out_checkpoint_dir)
    os.makedirs(out_state_log_dir)
    os.makedirs(out_run_dir)

    embodiment_names_sorted = sorted(data_by_embodiment.keys())
    valid_embodiment_names = []
    for embodiment_name in embodiment_names_sorted:
        if data_by_embodiment[embodiment_name]["best_run"] is not None:
            best_run = data_by_embodiment[embodiment_name]["best_run"]
        else:
            best_run = None

        metric = run_to_metric(best_run)

        state_log_path = os.path.join('..', 'state_logs', 'LeapHandRot', embodiment_name)
        cur_state_log_fnames = glob(state_log_path + "/*")
        cur_state_log_fnames.sort(reverse=True)
        if metric is not None and metric < args.max_metric and metric > 0 and len(cur_state_log_fnames) > 0:
            print(embodiment_name)

            """checkpoint dir"""

            # train checkpoint to checkpoint dir
            valid_embodiment_names.append(embodiment_name)
            checkpoint_path = os.path.join('..', 'runs', best_run.name, 'nn', 'LeapHand.pth')
            if not os.path.exists(checkpoint_path):
                checkpoint_path = os.path.join('..', 'runs', best_run.name, 'nn', 'LeapHandRot.pth')
            print(embodiment_name, metric, best_run.name)
            shutil.copy(checkpoint_path, os.path.join(out_checkpoint_dir, f'{embodiment_name}.pth'))

            """train run dir"""

            # make train run dir
            cur_train_run_dir = os.path.join(out_run_dir, best_run.name)
            os.makedirs(cur_train_run_dir)
            
            # train checkpoint to train run dir
            cur_train_nn_dir = os.path.join(cur_train_run_dir, 'nn')
            os.makedirs(cur_train_nn_dir)
            shutil.copy(checkpoint_path, os.path.join(cur_train_nn_dir, os.path.basename(checkpoint_path)))
            
            # train config to train run dir
            cur_train_config_path = os.path.join('..', 'runs', best_run.name, 'config.yaml')
            shutil.copy(cur_train_config_path, cur_train_run_dir)

            """rollout run dir"""

            rollout_run = data_by_embodiment[embodiment_name]['best_run_rollout']

            # make rollout run dir
            cur_rollout_run_dir = os.path.join(out_run_dir, rollout_run.name)
            os.makedirs(cur_rollout_run_dir)

            # rollout config to rollout run dir
            shutil.copy(os.path.join('..', 'runs', rollout_run.name, 'config.yaml'), os.path.join(cur_rollout_run_dir))

            """state logs"""

            # copy state logs
            state_log_path = cur_state_log_fnames[0]
            embodiment_state_log_dir = os.path.join(out_state_log_dir, embodiment_name)
            os.makedirs(embodiment_state_log_dir)
            shutil.copy(state_log_path, os.path.join(embodiment_state_log_dir, 'state_log.pt'))

        if args.debug:
            break

    print('Generating zero-shot state logs for embodiments without demonstration data')
    gen_zero_shot_state_log(out_state_log_dir)

    print(f'Wrote dataset to {out_path} with embodiments: {valid_embodiment_names} ({len(valid_embodiment_names)})')

    print('starting to zip (will take a while)')
    shutil.make_archive(out_zip_path.replace('.zip', ''), 'zip', os.path.dirname(out_path), os.path.basename(out_path))

    print(f'wrote zip to: {out_zip_path}')
