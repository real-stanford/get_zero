"""logs information about the status of the Leap Hand grasp cache status for the generated embodiments"""

import os
import yaml
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--get_missing', action='store_true', help='If passed will only print out the names of the embodiments that do not have grasp cache results and will not log anything else')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    def optional_print(message='', force=False):
        if args.get_missing and not force:
            return
        
        print(message)

    script_dir = os.path.dirname(os.path.realpath(__file__))
    cache_dir = os.path.join(script_dir, '..', 'cache', 'leap_hand')
    config_dir = os.path.join(script_dir, '..', 'cfg', 'embodiment', 'LeapHand', 'generated')

    embodiment_names = [fname.replace('.yaml', '') for fname in os.listdir(config_dir)]
    embodiment_names.sort()

    errors_by_dof = {}
    dof_frequency = {}
    embodiments_with_errors = []
    for i in range(16):
        errors_by_dof[i+1] = {'missing': 0, 'file_count': 0, 'missing_names': [], 'file_count_names': []}
        dof_frequency[i+1] = 0

    for embodiment_name in embodiment_names:
        embodiment_config_name = f'{embodiment_name}.yaml'
        with open(os.path.join(config_dir, embodiment_config_name), 'r') as fp:
            config = yaml.safe_load(fp)
        dof_count = config['dofCount']
        dof_frequency[dof_count] += 1

        cur_embodiment_cache = os.path.join(cache_dir, embodiment_name)

        if not os.path.exists(cur_embodiment_cache):
            optional_print(f'{embodiment_name} missing cache dir')
            errors_by_dof[dof_count]['missing'] += 1
            errors_by_dof[dof_count]['missing_names'].append(embodiment_name)
            embodiments_with_errors.append(embodiment_name)
        else:
            num_cache_files = len(os.listdir(cur_embodiment_cache))
            if num_cache_files != 5:
                optional_print(f'{embodiment_name} has wrong number of cache files ({num_cache_files} while expected 5)')
                embodiments_with_errors.append(embodiment_name)
                errors_by_dof[dof_count]['file_count'] += 1
                errors_by_dof[dof_count]['file_count_names'].append(embodiment_name)

    optional_print()
    optional_print(f'Errors aggregated:')
    for i in range(16):
        if dof_frequency[i+1] == 0:
            continue
        optional_print(f'Dof count {i+1} (with {dof_frequency[i+1]} embodiments) has errors: {errors_by_dof[i+1]}')

    optional_print()
    optional_print('Embodiments with missing/incomplete grasp cache:')
    optional_print(' '.join(embodiments_with_errors), force=True)
