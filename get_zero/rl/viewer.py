"""
Opens a viewer with the provided asset(s) loaded in (requires $DISPLAY variable to be set or X11 forwarding if using a remote server). If `--screenshot` arg is passed, then no viewer is created (and $DISPLAY is not used) and instead a camera is added to the environment.

See keybindings section in the code for keyboard shortcuts.
"""

from isaacgym import gymapi, gymtorch, gymutil
import argparse
from math import sqrt
import torch
import numpy as np
import os
from tqdm import trange, tqdm
from datetime import datetime
from PIL import Image
from yourdfpy import URDF
from get_zero.utils.forward_kinematics import fk
from scipy.spatial.transform import Rotation

VALID_VIEWPOINTS = ['angle', 'px', 'nx', 'py', 'ny', 'pz', 'nz']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--sim_device_id', type=int, default=-1)
    parser.add_argument('--graphics_device_id', type=int, default=-1)
    parser.add_argument('--asset', default='leap/leap_hand/generated/urdf/001.urdf', help='path relative to `assets` folder. Can be a specific asset, or a directly, in which case all assets from that directly are loaded (up to --max_assets)')
    parser.add_argument('--max_assets', default=10, type=int, help='If --asset corresponds to a directory, then load at most --max_asset assets')
    parser.add_argument('--n_envs', default=1, type=int, help='number of actors to put in environment')
    parser.add_argument('--width', default=1000, type=int, help='viewer width')
    parser.add_argument('--aspect_ratio', default=9/16, type=float, help='aspect ratio for viewer size')
    parser.add_argument('--fix_base', type=bool, default=True)
    parser.add_argument('--static', default=True, type=bool, help='whether or not to take physic steps')
    parser.add_argument('--rot_axis', type=float, nargs=3, default=[0,1,0], help='rotate asset around rot_axis by rot_deg; used only if n_envs==1')
    parser.add_argument('--rot_deg', type=float, default=180, help='rotate asset around rot_axis by rot_deg; used only if n_envs==1')
    parser.add_argument('--cam_dist', type=float, default=0.3, help='how far camera is away from asset when n_evs==1')
    parser.add_argument('--pose', type=float, nargs='+', default=None, help='Joint state for each dof for starting pose')
    parser.add_argument('--env_spacing', default=0.15, type=float, help='env spacing')
    parser.add_argument('--viewpoint', default='pz', help='viewpoint to start from. Used only if not in screenshot mode. If in screenshot mode then use --screenshot_path', choices=VALID_VIEWPOINTS)
    parser.add_argument('--screenshot', action='store_true', help='takes a screenshot and immediately quits')
    parser.add_argument('--screenshot_path', default='tmp', help='dir to save screenshots to or filename')
    parser.add_argument('--screenshot_viewpoints', default=['pz'], nargs='+', help='list of viewpoints to take screenshots from', choices=VALID_VIEWPOINTS)
    parser.add_argument('--no_time_in_screenshot_name', action='store_true', help='whether to include current time when saving a screenshot')
    parser.add_argument('--screenshot_transparent', action='store_true', help='whether to make background of screenshot transparent instead of showing ground plane')
    args = parser.parse_args()

    if args.gpu != -1:
        args.sim_device_id = args.graphics_device_id = args.gpu
    
    assert args.sim_device_id != -1 and args.graphics_device_id != -1, "must set --gpu or all three of: --sim_device_id --rl_device --graphics_device_id"

    return args


def open_viewer(args):
    """
    returns True if we want to reload viewer, False otherwise
    """
    gym = gymapi.acquire_gym()

    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z # Gym defaults to y axis as up-axis
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    sim_params.use_gpu_pipeline = False
    sim_params.physx.use_gpu = True
    sim = gym.create_sim(args.sim_device_id, args.graphics_device_id, gymapi.SIM_PHYSX, sim_params)

    # configure the ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
    plane_params.distance = 0
    plane_params.static_friction = 1
    plane_params.dynamic_friction = 1
    plane_params.restitution = 0

    # create the ground plane
    gym.add_ground(sim, plane_params)

    # Load all assets
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = args.fix_base

    asset_root = "assets"
    full_asset_path = os.path.join(asset_root, args.asset)
    robot_urdf = URDF.load(full_asset_path, load_meshes=False)
    if os.path.isdir(full_asset_path):
        asset_fnames = []
        for fname in os.listdir(full_asset_path):
            if '.urdf' in fname:
                asset_fnames.append(os.path.join(args.asset, fname))
        asset_fnames.sort()
        print(f'Found a total of {len(asset_fnames)} assets, and keeping at most {args.max_assets} to show!')
        asset_fnames = asset_fnames[:args.max_assets]
        args.n_envs = len(asset_fnames)
    else:
        asset_fnames = [args.asset]
    asset_files = [gym.load_asset(sim, asset_root, asset_file, asset_options) for asset_file in tqdm(asset_fnames, desc='loading assets')]
    dof_count = gym.get_asset_dof_count(asset_files[0])

    # set up the env grid
    asset_height = 1.0
    envs_per_row = int(sqrt(args.n_envs))
    env_spacing = args.env_spacing
    env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0)
    env_upper = gymapi.Vec3(env_spacing, env_spacing, asset_height)

    # cache some common handles for later use
    envs = []
    actor_handles = []

    # create and populate the environments
    asset_rotation = Rotation.from_rotvec(np.array(args.rot_axis) * np.deg2rad(args.rot_deg))
    for i in trange(args.n_envs, desc='creating envs'):
        asset_name = os.path.basename(asset_fnames[i % len(asset_files)]).replace('.urdf', '')
        asset = asset_files[i % len(asset_files)]
        env = gym.create_env(sim, env_lower, env_upper, envs_per_row)
        envs.append(env)

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, asset_height)
        pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(*args.rot_axis), np.deg2rad(args.rot_deg))

        actor_handle = gym.create_actor(env, asset, pose, asset_name, i)
        for i in range(gym.get_actor_rigid_body_count(env, actor_handle)):
            gym.set_rigid_body_segmentation_id(env, actor_handle, i, 1) # add segmentation ID to distinguish from background
        actor_handles.append(actor_handle)

    gym.prepare_sim(sim) # required by Tensor API when using gpu_pipeline
    
    # acquire root state tensor descriptor
    _root_tensor = gym.acquire_actor_root_state_tensor(sim)
    root_tensor = gymtorch.wrap_tensor(_root_tensor)

    # acquire DOF state tensor descriptor
    _dof_states = gym.acquire_dof_state_tensor(sim)
    dof_states = gymtorch.wrap_tensor(_dof_states).view(args.n_envs, dof_count, 2)
    gym.refresh_dof_state_tensor(sim)
    saved_dof_state = dof_states.clone()

    # go to initial pose if defined
    if args.pose:
        dof_states[:, :, 0] = torch.tensor(args.pose).unsqueeze(0).repeat(args.n_envs, 1)
        saved_dof_state = dof_states.clone()
        gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_states))
        gym.simulate(sim) # need to update to go to the target pose
    else:
        dof_states[:, :, 0] = 0
        saved_dof_state = dof_states.clone()
        gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_states))
        gym.simulate(sim) # need to update to go to the target pose

    # save root state
    gym.refresh_actor_root_state_tensor(sim)
    saved_root_tensor = root_tensor.clone()
    
    # asset properties
    joint_name_to_joint_i = gym.get_asset_joint_dict(asset_files[0])
    joint_i_to_joint_name = {v: k for k, v in joint_name_to_joint_i.items()}
        
    # viewer or camera setup
    using_viewer = not args.screenshot
    camera_props = gymapi.CameraProperties()
    camera_props.width = args.width
    camera_props.height = int(args.width * args.aspect_ratio)
    if using_viewer:
        viewer = gym.create_viewer(sim, camera_props)
    else:
        camera_props.far_plane = 10.0
        camera_props.near_plane = 1e-2
        camera_env = envs[0]
        cam_handle = gym.create_camera_sensor(camera_env, camera_props)
        output_screenshot = None

    camera_mode = None
    def cam_lookat(mode):
        nonlocal camera_mode
        camera_mode = mode

        print(f'Changing to viewpoint: {mode}')

        target_pos = np.array([0,0,asset_height])
        if mode == 'px' or mode == 'nx':
            dir = -1 if mode == 'nx' else 1
            camera_direction = np.array([dir,1e-5,0])
        elif mode == 'py' or mode == 'ny':
            dir = -1 if mode == 'ny' else 1
            camera_direction = np.array([1e-5,dir,0])
        elif mode == 'pz' or mode == 'nz':
            dir = -1 if mode == 'nz' else 1
            camera_direction = np.array([1e-5,0,dir])
        elif mode == 'angle':
            camera_direction = np.array([1,1,1])
        else:
            raise NotImplementedError
        camera_pos = target_pos + camera_direction * args.cam_dist
        
        target_pos = gymapi.Vec3(*target_pos)
        camera_pos = gymapi.Vec3(*camera_pos)

        if using_viewer:
            gym.viewer_camera_look_at(viewer, None, camera_pos, target_pos)
        else:
            gym.set_camera_location(cam_handle, camera_env, camera_pos, target_pos)

    if using_viewer:
        cam_lookat(args.viewpoint)
    else:
        cam_lookat(args.screenshot_viewpoints[0])
        cur_viewpoint_i = 0

    def screenshot():
        nonlocal output_screenshot

        # setup output path if this is the viewer or if this is the final viewpoint in screenshot mode
        if using_viewer or cur_viewpoint_i + 1 == len(args.screenshot_viewpoints):
            basename = os.path.basename(args.screenshot_path)
            if '.png' in basename or '.jpg' in basename:
                # path is file
                os.makedirs(os.path.dirname(args.screenshot_path), exist_ok=True)
                out_path = args.screenshot_path
            else:
                # path is folder
                out_dir = args.screenshot_path
                os.makedirs(out_dir, exist_ok=True)
                asset_name_for_file = f"{os.path.basename(asset_fnames[0]).replace('.urdf', '')}" if len(asset_fnames) == 1 else 'multiple_assets'
                time_for_file = '' if args.no_time_in_screenshot_name else datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
                connector = '_' if asset_name_for_file and time_for_file else ''
                out_path = os.path.join(out_dir, f"{asset_name_for_file}{connector}{time_for_file}.png")
        
        if using_viewer:
            gym.write_viewer_image_to_file(viewer, out_path)
            print(f'Saved screenshot to {out_path}')
        else:
            # Take picture of current viewpoint and concatenate it to any existing screenshots
            gym.render_all_camera_sensors(sim)
            img = gym.get_camera_image(sim, camera_env, cam_handle, gymapi.IMAGE_COLOR)
            img = img.reshape((img.shape[0], img.shape[1] // 4, 4))
            if args.screenshot_transparent:
                seg = gym.get_camera_image(sim, camera_env, cam_handle, gymapi.IMAGE_SEGMENTATION)
                img[:,:,3][seg == 0] = 0 # make background transparent
            else:
                img = img[:,:,:3] # remove alpha channel

            if output_screenshot is None:
                output_screenshot = img
            else:
                output_screenshot = np.concatenate((output_screenshot, img))

            # Save screenshot only on last viewpoint
            if cur_viewpoint_i + 1 == len(args.screenshot_viewpoints):
                Image.fromarray(output_screenshot).save(out_path)
                print(f'Saved screenshot to {out_path}')


    # keybindings
    if using_viewer:
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_Z, "reload")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_P, "pause")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_X, "actuate")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_C, "camera")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_S, "screenshot")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_D, "debug")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_F, "fk")

    def reset():
        # position reset
        gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(saved_root_tensor))
        gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(saved_dof_state))

        # force reset
        num_dofs = gym.get_sim_dof_count(sim)
        actions = torch.zeros(num_dofs, dtype=torch.float32)
        gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(actions))

        # step the physics
        if not args.static:
            gym.simulate(sim)
        gym.fetch_results(sim, True)

    reset()

    # sim loop
    going_to_reload = False
    is_paused = False
    loop_count = 0
    fk_enabled = False
    fk_joint_i = -1
    while not using_viewer or (using_viewer and not gym.query_viewer_has_closed(viewer)):
        loop_count += 1

        if using_viewer:
            for evt in gym.query_viewer_action_events(viewer):
                if evt.action == "reset" and evt.value > 0:
                    reset()
                if evt.action == "reload" and evt.value > 0:
                    going_to_reload = True
                if evt.action == "pause" and evt.value > 0:
                    is_paused = not is_paused
                if evt.action == "actuate" and evt.value > 0:
                    num_dofs = gym.get_sim_dof_count(sim)
                    actions = 1.0 - 2.0 * torch.rand(num_dofs, dtype=torch.float32)
                    gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(actions))
                if evt.action == "camera" and evt.value > 0:
                    # move to next viewpoint
                    cam_lookat(VALID_VIEWPOINTS[(VALID_VIEWPOINTS.index(camera_mode) + 1) % len(VALID_VIEWPOINTS)])
                if evt.action == "screenshot" and evt.value > 0:
                    screenshot()
                if evt.action == "debug" and evt.value > 0:
                    asset = asset_files[0]
                    rb_count = gym.get_asset_rigid_body_count(asset)
                    asset_dof_dict = gym.get_asset_dof_dict(asset)
                    asset_joint_dict = gym.get_asset_joint_dict(asset)
                    asset_joint_dict_invert = {v: k for k,v in asset_joint_dict.items()}
                    asset_rigid_body_dict = gym.get_asset_rigid_body_dict(asset)
                    asset_rigid_body_dict_invert = {v: k for k,v in asset_rigid_body_dict.items()}

                    print('\nDEBUG INFO')
                    print(f'asset_dof_dict: {asset_dof_dict}\nasset_joint_dict: {asset_joint_dict}\nasset_rigid_body_dict: {asset_rigid_body_dict}')
                    for i in range(rb_count):
                        print(f'body {i} has name {asset_rigid_body_dict_invert[i]}')

                    print()

                    for i in range(dof_count):
                        print(f'DoF {i} has joint name: \"{asset_joint_dict_invert[i]}\"')
                if evt.action == "fk" and evt.value > 0:
                    fk_enabled = True
                    
                    fk_joint_i = (fk_joint_i + 1) % gym.get_asset_dof_count(asset_files[0])
                    print(f'fk switched to joint "{joint_i_to_joint_name[fk_joint_i]}"')

            # if 'f' key is pressed, then the origin of the joint will be drawn on the viewer using forward kinematics (computed outside Isaac Gym). Press 'f' to toggle through joints in sim ordering.
            if fk_enabled:
                gym.clear_lines(viewer)

                # run forward kinematics across all envs
                cur_joint_angles = dof_states[:, :, 0] # index 0 is joint angles; ordered based on sim ordering; (num_envs, dof_count)
                joint_pos = fk(robot_urdf, cur_joint_angles, joint_name_to_joint_i) # (num_envs, dof_count, 3)
                pose = joint_pos[:, fk_joint_i] # (num_envs, 3)
                
                for i in range(args.n_envs):
                    env = envs[i]

                    # offset consists of both offset of environment and offset from environment to base link
                    offset = gym.get_env_origin(env)
                    offset = np.array([offset.x, offset.y, offset.z]) + root_tensor[i][:3].cpu().numpy()

                    # draw the pose
                    cur_pose = asset_rotation.apply(pose[i]) # apply the global asset rotation
                    glob_pos = gymapi.Vec3(offset[0] + cur_pose[0], offset[1] + cur_pose[1], offset[2] + cur_pose[2])
                    
                    target_pose = gymapi.Transform(
                        p=glob_pos,
                        r=gymapi.Quat(1,0,0,0),
                    )
                    axis_geom = gymutil.AxesGeometry(scale=0.1)
                    gymutil.draw_lines(axis_geom, gym, viewer, env, target_pose)

        if going_to_reload:
             break

        if not is_paused:
            # step the physics
            if not args.static:
                gym.simulate(sim)
            gym.fetch_results(sim, True)

        # update the viewer
        gym.step_graphics(sim)
        if using_viewer:
            gym.draw_viewer(viewer, sim, True)

        # Wait for dt to elapse in real time.
        # This synchronizes the physics simulation with the rendering rate.
        gym.sync_frame_time(sim)

        if loop_count >= 2 and args.screenshot:
            screenshot()
            cur_viewpoint_i += 1
            if cur_viewpoint_i < len(args.screenshot_viewpoints):
                cam_lookat(args.screenshot_viewpoints[cur_viewpoint_i])
            else:
                break
             
    if using_viewer:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

    return going_to_reload


if __name__ == '__main__':
    args = parse_args()
    
    while open_viewer(args):
         pass
    