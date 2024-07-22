import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

leap_debug_path = os.path.join('..', '..', 'debug', 'leap')

def load_sim_data(sim_name):
    return np.load(os.path.join(leap_debug_path, 'sim', sim_name, 'data.npz'))

def load_real_data(real_name):
    return np.load(os.path.join(leap_debug_path, 'real', real_name, 'data.npz'))

joint_i_to_name = {
    0: 'Index Abduction',
    1: 'Index Base',
    2: 'Index Middle',
    3: 'Index Tip',
    4: 'Middle Abduction',
    5: 'Middle Base',
    6: 'Middle Middle',
    7: 'Middle Tip',
    8: 'Ring Abduction',
    9: 'Ring Base',
    10: 'Ring Middle',
    11: 'Ring Tip',
    12: 'Thumb Base',
    13: 'Thumb Middle 1',
    14: 'Thumb Middle 2',
    15: 'Thumb Tip'
} # defined based on real indices

def plot_all(sim_name=None, real_name=None):
    modes = ['targets', 'network_targets', 'obs', 'joints', 'network_actions', 'joints_vs_targets']
    for mode in modes:
        plot_run(mode, sim_name, real_name)
    
def plot_run(mode, sim_name=None, real_name=None,):
    """
    - plot data comparing sim and real
    - must supply mode parameter
    - mode=='targets' outputs targets
    - mode=='network_targets' outputs network targets
    - mode=='obs' outputs entire obs vector
    - mode=='joints' outputs joints from obs vector
    - mode=='network_actions' outputs network actions
    - mode=='joints_vs_targets' outputs joint state and the targets for those joints
    """
    assert real_name or sim_name, 'need to provide either real or sim path'

    if real_name:
        real_data = load_real_data(real_name)
        sim_to_real_indices = real_data['sim_to_real_indices']
        real_to_sim_indices = real_data['real_to_sim_indices']
        real_joints_unscaled = real_data['joints_unscaled'][:, sim_to_real_indices]
        real_targets = real_data['target_list'][:, sim_to_real_indices]
        real_joints_scaled = real_data['joints_scaled'][:, sim_to_real_indices]
        real_obs = real_data['obs_list']
        real_network_targets = real_data['network_targets_list'][:, sim_to_real_indices]
        real_network_actions = real_data['network_actions_list'][:, sim_to_real_indices]
        leap_dof_lower = real_data['leap_dof_lower'][sim_to_real_indices]
        leap_dof_upper = real_data['leap_dof_upper'][sim_to_real_indices]

    if sim_name:
        sim_data = load_sim_data(sim_name)

        # These will have already been loaded if real path supplied as well
        real_sim_to_real_indices = sim_data['sim_to_real_indices']
        real_real_to_sim_indices = sim_data['real_to_sim_indices']
        real_leap_dof_lower = sim_data['leap_dof_lower'][sim_to_real_indices]
        real_leap_dof_upper = sim_data['leap_dof_upper'][sim_to_real_indices]
        if not real_name:
            sim_to_real_indices = real_sim_to_real_indices
            leap_dof_lower = real_leap_dof_lower
            leap_dof_upper = real_leap_dof_upper
        else:
            # as a santity check assert they are the same between sim and real sources in the case that both are present
            assert np.allclose(real_sim_to_real_indices, sim_to_real_indices)
            assert np.allclose(real_real_to_sim_indices, real_to_sim_indices)
            assert np.allclose(real_leap_dof_lower, leap_dof_lower)
            assert np.allclose(real_leap_dof_upper, leap_dof_upper)            

        sim_joints_unscaled = sim_data['joints_unscaled'][:, 0, sim_to_real_indices]
        sim_joints_scaled = sim_data['joints_scaled'][:, 0, sim_to_real_indices]
        sim_targets = sim_data['target_list'][:, sim_to_real_indices]
        sim_actions = sim_data['actions_list'][:, sim_to_real_indices]
        sim_obs = sim_data['obs_list']

    rows = 102 if mode == 'obs' else 16
    cols = 2 if mode == 'joints_vs_targets' and sim_name and real_name else 1
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(14 * cols, rows))
    plt.subplots_adjust(hspace=1, top=0.95, bottom=0.05)
    for r in range(rows):
        for c in range(cols):
            if cols == 1:
                ax = axs[r]
            else:
                ax = axs[r, c]

            limits = None        
            data_list = []
            labels_list = []
            kwargs_list = [{'linewidth': 5}, {}]
            if mode == 'joints':
                if real_name:
                    data_list.append(real_joints_unscaled[:, r])
                    labels_list.append('Real joints unscaled')

                if sim_name:
                    data_list.append(sim_joints_unscaled[:, r])
                    labels_list.append('Sim joints unscaled')
            elif mode == 'targets':
                if real_name:
                    data_list.append(real_targets[:, r])
                    labels_list.append('Real targets')

                if sim_name:
                    data_list.append(sim_targets[:, r])
                    labels_list.append('Sim targets')

                limits = [leap_dof_lower[r], leap_dof_upper[r]]
            elif mode == 'network_targets':
                if real_name:
                    data_list.append(real_network_targets[:, r])
                    labels_list.append('Real network targets')

                if sim_name:
                    data_list.append(sim_targets[:, r])
                    labels_list.append('Sim targets')

                limits = [leap_dof_lower[r], leap_dof_upper[r]]
            elif mode == 'network_actions':
                if real_name:
                    data_list.append(real_network_actions[:, r])
                    labels_list.append('Real network actions')

                if sim_name:
                    data_list.append(sim_actions[:, r])
                    labels_list.append('Sim actions')
                
                limits = [-1, 1]
            elif mode == 'obs':
                if real_name:
                    data_list.append(real_obs[:, r])
                    labels_list.append('Real obs')

                if sim_name:
                    data_list.append(sim_obs[:, r])
                    labels_list.append('Sim obs')
            elif mode == 'joints_vs_targets':
                if sim_name and real_name:
                    plotting_real = c == 0
                else:
                    plotting_real = real_name is not None

                if plotting_real:
                    kwargs_list[0]['color'] = 'r'
                    kwargs_list[1]['color'] = 'b'
                    data_list.extend([real_joints_scaled[:, r], real_targets[:, r]])
                    labels_list.extend(['Real joints unscaled', 'Real joints targets'])
                else:
                    kwargs_list[0]['color'] = 'g'
                    kwargs_list[1]['color'] = 'm'
                    data_list.extend([sim_joints_scaled[:, r], sim_targets[:, r]])
                    labels_list.extend(['Sim joints unscaled', 'Sim joints targets'])
            else:
                raise NotImplementedError

            for data, kwargs, label in zip(data_list, kwargs_list, labels_list):
                if r == 0:
                    kwargs['label'] = label
                ax.plot(data, **kwargs)

            if mode == 'obs':
                title = str(r)
            else:
                title = f'Joint {r} - {joint_i_to_name[r]}'
            ax.set_title(title)
            
            if limits:
                ax.set_ylim(limits)
    
    plt.figlegend(loc='lower center')
    
    if sim_name and real_name:
        folder_name = os.path.join('sim_vs_real', f'{real_name}_{sim_name}')
    elif sim_name:
        folder_name = os.path.join('sim', sim_name)
    elif real_name:
        folder_name = os.path.join('real', real_name)

    out_dir = os.path.join(leap_debug_path, folder_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'plot_{mode}.png')
    plt.savefig(out_path, dpi=300)
    print(f'Output to {out_path}')

def compare_sim_and_real_obs(sim_name, real_name):
    np.set_printoptions(formatter={'float_kind':"{:+.4f}".format}, linewidth=200)

    sim_data = load_sim_data(sim_name)
    real_data = load_real_data(real_name)

    sim_obs = sim_data['obs_list']
    real_obs = real_data['obs_list']

    print(sim_obs.shape, real_obs.shape)
    print('sim time 0:', sim_obs[0])
    print('real time 0:', real_obs[0])

    # compare different in observation vectors between sim and real
    print(f'Computing diffs between real and sim diffs')
    max_diff, max_diff_i, max_diff_obs_i = 0, 0, 0
    for i in range(min(sim_obs.shape[0], real_obs.shape[0])):
        cur_sim_obs = sim_obs[i]
        cur_real_obs = real_obs[i]

        abs_diff = np.abs(cur_sim_obs - cur_real_obs)
        diff = max(abs_diff)
        diff_obs_i = np.argmax(abs_diff)
        if diff > max_diff:
            max_diff = diff
            max_diff_i = i
            max_diff_obs_i = diff_obs_i
        print(f'time: {i} largest dif: {diff}')
    print(f'Largest diff across all {sim_obs.shape[0]} timesteps is {max_diff} at timestep {max_diff_i} obs index {max_diff_obs_i}. Real value: {real_obs[max_diff_i, max_diff_obs_i]} Sim value: {sim_obs[max_diff_i, max_diff_obs_i]}')
    print(f'Sim at time {max_diff_i}: {sim_obs[max_diff_i]}\nReal at time {max_diff_i}: {real_obs[max_diff_i]}')

    if sim_obs.shape[0] != real_obs.shape[0]:
        print(f'WARNING: sim_obs {sim_obs.shape} and real_obs {real_obs.shape} do not have the same shape!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim_name', type=str, default=None)
    parser.add_argument('--real_name', type=str, default=None)
    args = parser.parse_args()

    if args.sim_name and args.real_name:
        compare_sim_and_real_obs(args.sim_name, args.real_name)
    
    plot_all(args.sim_name, args.real_name)
