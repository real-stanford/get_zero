"""
Takes in the original leap hand URDF and generates all possible hardware configurations, outputting them to new URDF files.

The first set of generated embodiments will correspond to joint removal to create different connectivities.
The second set of generated embodiments will correspond to geometry differences with extended limbs.

Notes on valid hardware configurations:
General:
    - all fingers must end in fingertip
    - the base motor for each finger is already included in palm_lower
    - joints will need to be updated with new parent and child
    - Joint transform is specified with respect to the parent link, so joints are tied to the parent link, thus given a parent link we can determine a child link
    - index finger has no additional suffix for link names. Middle finger uses _2 suffix. Ring finger uses _3 suffix

Notes on sim2real joint ordering:
    - the names of the joints in the URDF correspond to the ID of the motors on the real hand (this seems to be true except the thumb_pip.stl might be an exception since it's the only case where there are additional parts that come after it, but it itself does not have a physical motor as a part of it. To handle this, we have a custom operation that takes correctly sets the real motor ID for this case.)
    - Isaac Gym assigns a new joint ordering which is the sim ordering
"""

import os
import xml.etree.ElementTree as ET
from treelib import Tree
import numpy as np
import shutil
import yaml
from copy import deepcopy
from itertools import product
from yourdfpy import URDF
from typing import Dict
import random

"""read canoncial pose from LEAP config file"""
with open('../cfg/embodiment/LeapHand/OrigRepo.yaml', 'r') as f:
    cfg = yaml.safe_load(f)
canoncial_pose_original_sim = np.array(cfg['canonical_pose']) # notably, this is ordered based on sim indices of the default LEAP hand configuration
canoncial_pose_original_real = canoncial_pose_original_sim[cfg['sim_to_real_indices']]
print(canoncial_pose_original_real)

# Specification for canonical pose as a function of the finger sequence. Note this is unique for every single finger, but we make an independence assumption that the canoncial pose for one finger doesn't depend on the configuration of the other fingers
fingertip_pose = [1]
dip_fingertip_pose = [0.5, 1]
finger_chain_to_canoncial_pose = {
    # index
    # ('mcp_joint', 'pip', 'dip', 'fingertip'): [], # don't need to define this since already in original canonical pose
    ('mcp_joint', 'pip', 'fingertip'): [0.3, -0.3, 1.5],
    ('dip', 'fingertip'): dip_fingertip_pose,
    ('fingertip',): fingertip_pose,

    # middle
    # ('mcp_joint_2', 'pip_2', 'dip_2', 'fingertip_2'): [], # don't need to define this since already in original canonical pose
    ('mcp_joint_2', 'pip_2', 'fingertip_2'): [0.3, 0, 1.5],
    ('dip_2', 'fingertip_2'): dip_fingertip_pose,
    ('fingertip_2',): fingertip_pose,

    # ring
    # ('mcp_joint_3', 'pip_3', 'dip_3', 'fingertip_3'): [], # don't need to define this since already in original canonical pose
    ('mcp_joint_3', 'pip_3', 'fingertip_3'): [0.3, 0.3, 1.5],
    ('dip_3', 'fingertip_3'): dip_fingertip_pose,
    ('fingertip_3',): fingertip_pose,

    # thumb
    # ('pip_4', 'thumb_pip', 'thumb_dip', 'thumb_fingertip'): [], # don't need to define this since already in original canonical pose
    ('pip_4', 'thumb_pip', 'thumb_fingertip'): [1.57, 1.57, 1.5]
}

def remove_suffix(string, suffix_list=['_2', '_3', '_4']):
    for suffix in suffix_list:
        string = string.replace(suffix, '')
    return string

def remove_long_suffix(string):
    return remove_suffix(string, suffix_list=['_long'])

def rpy_to_np(rpy_str):
    return np.array([float(x) for x in rpy_str.split(' ')])

# Parse the XML to automatically get joint and link connectivity and mapping
tree = ET.parse('../assets/leap/leap_hand/original_refactor.urdf')
root = tree.getroot()

joint_to_child_link = {}
joint_to_parent_link = {}
joint_to_xml = {}
link_to_xml = {}
link_to_child_joints = {}
finger_name_to_base_joint = {
    "thumb": "12",
    "index": "1",
    "middle": "5",
    "ring": "9"
}

# maps from f'{parent_link}__{child_link}' -> rpy string giving the rotation transform that goes in the connecting joint. Note that _2 or _3 should be removed from parent_link and child_link before acessing this Dict since they are identical the the case with no suffix (the index finger). _4 also can be removed since the connections in the thumb have unique names so they will have unique entries in this Dict.
# The entries that manually specified here are the link to link connections that don't exist in the default hand configuration, so we will need to manually define the transforms between these links
parent_link_to_child_link_rpy = {
    "pip__fingertip": "1.5707963267948958919 -1.570796326794896336 0",
    "palm_lower__dip": "1.5707963267948950037 -1.5707963267948945596 0",
    "palm_lower__fingertip": "1.5707963267948950037 -1.5707963267948945596 0",
    "thumb_pip__thumb_fingertip": "-1.570796326794896558 3.14 -1.9721522630516624601e-31"
}

# TODO: the following code could be rewritten with `yourdfpy` to be much cleaner
for child_xml in root:
    if child_xml.tag == 'link':
        link_name = child_xml.attrib["name"]
        link_to_xml[link_name] = child_xml
        print(f'Found link: "{link_name}"')
    elif child_xml.tag == 'joint':
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
        if parent_link_name not in link_to_child_joints:
            link_to_child_joints[parent_link_name] = []
        link_to_child_joints[parent_link_name].append(joint_name)
        joint_to_child_link[joint_name] = child_link_name
        joint_to_xml[joint_name] = child_xml

        # save joint rpy transform
        joint_rpy = child_xml.find('origin').attrib['rpy']
        parent_to_child_key = f'{remove_suffix(parent_link_name)}__{remove_suffix(child_link_name)}'
        if parent_to_child_key in parent_link_to_child_link_rpy:
            existing_joint_rpy = parent_link_to_child_link_rpy[parent_to_child_key]
            if not np.isclose(rpy_to_np(joint_rpy), rpy_to_np(existing_joint_rpy)).all(): # isclose needed since sometimes the transforms were off by a very small decimal
                print(f'existing rpy: {existing_joint_rpy} cur rpy: {joint_rpy}')
                assert False, 'rotation transform between corresponding links should be the same between fingers'
        else:
            parent_link_to_child_link_rpy[parent_to_child_key] = joint_rpy

        print(f'Found joint "{joint_name}" with parent: "{parent_link_name}" and child: "{child_link_name}"')

print()

# print out the tree structure of the original hand configuration
def build_tree(tree, root_link_name, parent_node_name=None):
    tree.create_node(root_link_name, root_link_name, parent=parent_node_name)
    for child_joint in link_to_child_joints.get(root_link_name, []):
        tree.create_node(child_joint, child_joint, parent=root_link_name)
        child_link = joint_to_child_link[child_joint]
        build_tree(tree, child_link, child_joint)

tree = Tree()
build_tree(tree, 'palm_lower')
print('Original hand configuration')
print(tree)

# Enumerate all possible hardware configurations
def append_suffix_to_config(config, suffix):
    result = []
    for lst in config:
        result.append([f'{x}{suffix}' for x in lst])
    return result

index_connectivity_configurations = [
    ['mcp_joint', 'pip', 'dip', 'fingertip'],
    ['mcp_joint', 'pip', 'fingertip'],
    ['dip', 'fingertip'],
    ['fingertip'],
    []
]

hand_connectivity_configurations = {
    'thumb': [
        ['pip_4', 'thumb_pip', 'thumb_dip', 'thumb_fingertip'],
        ['pip_4', 'thumb_pip', 'thumb_fingertip']
    ],
    'index': index_connectivity_configurations,
    'middle': append_suffix_to_config(index_connectivity_configurations, '_2'),
    'ring': append_suffix_to_config(index_connectivity_configurations, '_3')
}

# create outputs dirs
relative_asset_dir = 'leap/leap_hand/generated'
out_dir = os.path.join('../assets/', relative_asset_dir)
urdf_dir = os.path.join(out_dir, 'urdf')
tree_dir = os.path.join(out_dir, 'tree')
config_dir = '../cfg/embodiment/LeapHand/generated'
if os.path.exists(urdf_dir):
    shutil.rmtree(urdf_dir)
if os.path.exists(tree_dir):
    shutil.rmtree(tree_dir)
if os.path.exists(config_dir):
    shutil.rmtree(config_dir)
os.makedirs(urdf_dir)
os.makedirs(tree_dir)
os.makedirs(config_dir)

"""Generate embodiments with different connectivity"""
# Go through every configuration option and save the generated URDF file
config_count = 0
for thumb_config in hand_connectivity_configurations['thumb']:
    for index_config in hand_connectivity_configurations['index']:
        for middle_config in hand_connectivity_configurations['middle']:
            for ring_config in hand_connectivity_configurations['ring']:
                # ensure at least 2 fingers are present in the embodiment
                num_missing_fingers = (len(thumb_config) == 0) + (len(index_config) == 0) + (len(middle_config) == 0) + (len(ring_config) == 0)
                num_present_fingers = 4 - num_missing_fingers
                if num_present_fingers < 2:
                    continue
                
                # ensure that across all non thumb fingers there are at least 3 joints or at least one finger with at least two joints
                if len(index_config) + len(middle_config) + len(ring_config) < 3 and len(index_config) < 2 and len(middle_config) < 2 and len(ring_config) < 2:
                    continue

                config_count += 1

                root = ET.Element('robot', attrib={'name': 'onshape'})
                root.append(link_to_xml['palm_lower']) # add "palm_lower" to XML
                tree = ET.ElementTree(root)

                print_tree = Tree()
                print_tree.create_node('palm_lower', 'palm_lower')

                dof_count = 0
                fingertip_indices = []
                sim_to_real_map = {}
                canoncial_pose = []
                joint_name_to_joint_i = {}

                isaacgym_finger_ordering = ['index', 'thumb', 'middle', 'ring'] # isaac gym does depth first ordering of the joints/bodies (though I'm not sure how it choses which fingers to go through first, but this is the ordering found by loading the assets into the simulator and it appears to be consistent across different hand configurations)

                for finger_name, seq in zip(isaacgym_finger_ordering, [index_config, thumb_config, middle_config, ring_config]):
                    cur_parent_link = 'palm_lower'
                    cur_joint = finger_name_to_base_joint[finger_name]
                    for seq_i, cur_child_link in enumerate(seq):
                        # note that this traversal exactly matches the ordering that Isaac Gym loads the joints/bodies, which makes it conveinient to construct the all the configuration metadata without explicitly having to load the asset into the simulator to test the import order
                        """Add joint"""
                        joint_xml = deepcopy(joint_to_xml[cur_joint]) # since we are modifying the joint_xml, it's best to make a copy to ensure we aren't messing with the values for other configurations
                        assert joint_xml.find('parent').attrib['link'] == cur_parent_link, 'the joint position is based on the parent, so we need to ensure the parent has not changed in this new configuration. This should hold since we always get the joint based on its parent'
                        joint_xml.find('child').attrib['link'] = cur_child_link
                        parent_child_key = f'{remove_suffix(cur_parent_link)}__{remove_suffix(cur_child_link)}'
                        rpy = parent_link_to_child_link_rpy[parent_child_key]
                        joint_xml.find('origin').attrib['rpy'] = rpy

                        # Special case of needing to fix joint name for thumb if there is a connection from thumb_pip to thumb_fingertip (since thumb_pip doesn't have a motor and thumb_fingertip does and we need the name of the real motor)
                        if cur_parent_link == 'thumb_pip' and cur_child_link == 'thumb_fingertip':
                            old_name = joint_xml.attrib['name']
                            new_name = joint_to_xml[link_to_child_joints['thumb_dip'][0]].attrib['name']
                            joint_xml.attrib['name'] = cur_joint = new_name
                        
                        # save sim and real indices (note this has to be done after the special case handling above since we want to used the new value of `cur_joint` if it's updated)
                        sim_i = dof_count
                        real_i = int(cur_joint) # the joint name in the URDF file corresponds to the real motor ID
                        sim_to_real_map[sim_i] = real_i
                        dof_count += 1
                        joint_name_to_joint_i[cur_joint] = sim_i

                        # Canonical pose
                        if tuple(seq) in finger_chain_to_canoncial_pose:
                            cur_finger_canonical_pose = finger_chain_to_canoncial_pose[tuple(seq)]
                            cur_joint_canonical_pose = cur_finger_canonical_pose[seq_i]
                            canoncial_pose.append(cur_joint_canonical_pose)
                        else:
                            canoncial_pose.append(None)
                        
                        root.append(joint_xml)
                        print_tree.create_node(cur_joint, cur_joint, parent=cur_parent_link)

                        """Add link"""
                        link_xml = link_to_xml[cur_child_link]
                        root.append(link_xml)
                        print_tree.create_node(cur_child_link, cur_child_link, parent=cur_joint)

                        """update vars"""
                        cur_parent_link = cur_child_link
                        # there should only be 1 child for joints after the base, except for 0 children at the very end on the kinematic chain
                        if cur_child_link in link_to_child_joints:
                            cur_joint = link_to_child_joints[cur_child_link][0]
                        else:
                            # this is a fingertip position, add it to the list
                            fingertip_indices.append(dof_count) # this is only a valid operation becuase we traverse through the fingers in the same ordering that isaac gym does (see `isaacgym_finger_ordering`). Notably set at dof_count rather than dof_count - 1 (which would be the joint index) because this term is actually referring to the rigid link body not the joint index and because of this we need to account for the palm of the hand which is at rigid body index 0
                
                id_str = f'{config_count:03}'
                
                # Save the configuration
                urdf_name = f'{id_str}.urdf'
                fpath = os.path.join(urdf_dir, urdf_name)
                tree.write(fpath)

                # Save the tree image
                fpath = os.path.join(tree_dir, f'{id_str}.txt')
                with open(fpath, 'w') as f:
                    f.write(str(print_tree))

                """Generate the embodiment config for the RL tasks"""
                real_to_sim_indices = [None] * dof_count
                sim_to_real_indices = [-1] * 16 # note that this list is based off the original 16 motors which have IDs, so it shold always be length 16, but will have -1 in places where the real motor is not present in this configuration
                for sim_index, real_index in sim_to_real_map.items():
                    real_to_sim_indices[sim_index] = real_index
                    sim_to_real_indices[real_index] = sim_index

                # For places where canonical pose isn't explicitly defined in finger_chain_to_canoncial_pose, pull values from the original canonical pose
                for sim_i in range(dof_count):
                    real_i = real_to_sim_indices[sim_i]
                    if canoncial_pose[sim_i] is None:
                        canoncial_pose[sim_i] = float(canoncial_pose_original_real[real_i])

                # save embodiment parameter file for task
                fpath = os.path.join(config_dir, f'{id_str}.yaml')
                hand_asset_path = os.path.join(relative_asset_dir, 'urdf', urdf_name)
                params = {
                    'dofCount': dof_count,
                    'asset': {'handAsset': hand_asset_path},
                    'canonical_pose': canoncial_pose,
                    'real_to_sim_indices': real_to_sim_indices,
                    'sim_to_real_indices': sim_to_real_indices,
                    'fingertip_indices': fingertip_indices,
                    'joint_name_to_joint_i': joint_name_to_joint_i,
                    'robot_asset': "${.asset.handAsset}",
                    'embodiment_name': id_str
                }
                with open(fpath, 'w') as f:
                    f.write("# @package _global_.task.env\n\n")
                    yaml.dump(params, f, default_flow_style=None)

num_connectivity_variations = config_count
print(f'Generated {num_connectivity_variations} different connectivity variations')

"""Generate embodiments that have link length variations"""
# enumerate all combinations of the 16 links with the normal and long variants
index_components = ['mcp_joint', 'pip', 'dip', 'fingertip']
middle_components = append_suffix_to_config([index_components], '_2')[0]
ring_components = append_suffix_to_config([index_components], '_3')[0]
thumb_components = ['pip_4', 'thumb_pip', 'thumb_dip', 'thumb_fingertip']

hand_components = index_components + middle_components + ring_components + thumb_components # list of all link names in URDF

# map from long link name to link position relative to parent joint. None indicates to not overwrite original value (in the case of the link pos not changing after the geometry variation). This value is determined by measuring the distance from the origin of the STL file (open the STL in Blender and the origin will be indicated with yellow dot; this is the same origin used to load the link) to the center of the parent joint (use ruler tool). Start with original xyz position of the link from the URDF and then update if needed.
long_link_to_link_pos = {
    'mcp_joint_long': None,
    'dip_long': None,
    'fingertip_long': None,
    'thumb_dip_long': '0.043968715707239175439 0.072952952973709198625 -0.0086286764493694757122', # changed middle entry from 0.057952952973709198625; 1.5cm increase
    'thumb_fingertip_long': None
}

# map from long link name to child joint position. None indicates to not overwrite original value. This value is determined by measuring joint to joint distance in the modified (increase length) link STL file (you can open the STL in Blender and measure the parent joint center to child joint center using the ruler). Start with original xyz position of the link from the URDF and then modify the enlongated dimension.
long_link_name_to_child_joint_pos = {
    'mcp_joint_long': '-0.012200000000000007713 0.053099999999999994982 0.014500000000000000736', # changed middle entry from 0.038099999999999994982; 1.5cm increase
    'dip_long': '-4.0880582495572692636e-09 -0.051100004210367367397 0.00020000000000007858714', # changed middle entry from -0.036100004210367367397; 1.5cm increase
    'thumb_dip_long': '-1.249000902703301108e-16 0.061599999999999863753 0.00019999999999997710581' # changed middle entry from 0.046599999999999863753; 1.5cm increase
    # do not need to include 'fingertip_long' or 'thumb_fingertip_long' here because it's at the end of the fingers and thus does not have a child joint
}

def gen_length_variations(embodiment_name, has_long_variant, max_modifications_per_finger=2, max_fingers_at_max_modifications=None, min_modifications=0, max_modifications=None, min_non_fingertip_modifications=0, max_generated_embodiments=None, shuffle=False):
    global config_count

    urdf = URDF.load(os.path.join(urdf_dir, f'{embodiment_name}.urdf'), load_meshes=False, build_scene_graph=False)
    link_names = list(urdf.link_map.keys())
    link_names = [x for x in hand_components if x in link_names]
    geometry_configs = []

    def select_links_in_embodiment(link_names_list):
        return [x for x in link_names_list if x in link_names]

    present_index_components = select_links_in_embodiment(index_components)
    present_middle_components = select_links_in_embodiment(middle_components)
    present_ring_components = select_links_in_embodiment(ring_components)
    present_thumb_components = select_links_in_embodiment(thumb_components)

    for component_name in link_names:
        if remove_suffix(component_name) in has_long_variant:
            geometry_configs.append([False, True])
        else:
            geometry_configs.append([False])

    permutations = product(*geometry_configs)
    if shuffle:
        permutations = list(permutations)
        random.seed(321)
        random.shuffle(permutations)

    num_generated = 0
    for geometry_config in permutations:
        if max_generated_embodiments is not None and num_generated == max_generated_embodiments:
            return

        # geometry_config is list of bool where true at index i if ith link in link_names should use the long variant at this configuration
        start_i = 0
        cur_length = len(present_index_components)
        index_config = geometry_config[start_i:start_i+cur_length]; start_i += cur_length; cur_length = len(present_middle_components)
        middle_config = geometry_config[start_i:start_i+cur_length]; start_i += cur_length; cur_length = len(present_ring_components)
        ring_config = geometry_config[start_i:start_i+cur_length]; start_i += cur_length; cur_length = len(present_thumb_components)
        thumb_config = geometry_config[start_i:start_i+cur_length]

        # skip generating this configuration if it doesn't meet certain requirements
        if sum(index_config) > max_modifications_per_finger or sum(middle_config) > max_modifications_per_finger or sum(ring_config) > max_modifications_per_finger or sum(thumb_config) > max_modifications_per_finger:
            # maximum number of changes in a single finger
            continue
        if sum(index_config) + sum(middle_config) + sum(ring_config) + sum(thumb_config) < min_modifications:
            # require a certain number of changes
            continue
        if sum(index_config[:-1]) + sum(middle_config[:-1]) + sum(ring_config[:-1]) + sum(thumb_config[:-1]) < min_non_fingertip_modifications:
            # require at least min_non_fingertip_modifications modifications to be not just fingertips
            continue
        if max_modifications is not None and sum(geometry_config) > max_modifications:
            # require at at most max_modifications modifications across entire hand
            continue
        if max_fingers_at_max_modifications is not None and (sum(index_config) == max_modifications_per_finger) + (sum(middle_config) == max_modifications_per_finger) + (sum(ring_config) == max_modifications_per_finger) + (sum(thumb_config) == max_modifications_per_finger) > max_fingers_at_max_modifications:
            # require at at most max_modifications modifications across entire hand
            continue


        config_count += 1
        num_generated += 1

        modified_xml: ET.Element = urdf.write_xml()

        link_name_to_link_xml: Dict[str, ET.Element] = {}
        for link_xml in modified_xml.findall('link'):
            link_name_to_link_xml[link_xml.attrib['name']] = link_xml
        joint_name_to_joint_xml: Dict[str, ET.Element] = {}
        for joint_xml in modified_xml.findall('joint'):
            joint_name_to_joint_xml[joint_xml.attrib['name']] = joint_xml

        tree_str = open(os.path.join(tree_dir, f'{embodiment_name}.txt')).read()

        ## replace link at index link_i in modified_urdf with the long variant if long variant selected
        for link_i, use_long_variant_link_i in enumerate(geometry_config):
            if not use_long_variant_link_i:
                continue

            link_name = link_names[link_i]
            new_link_name = f'{link_name}_long'
            suffix_free_new_link_name = f'{remove_suffix(link_name)}_long'
            link_xml = link_name_to_link_xml[link_name]

            # update link name
            link_xml.attrib['name'] = f'{link_name}_long'
            
            # replace mesh filename in geometry visual and collision meshes to reflect the longer version stl file
            for geometry_tag in link_xml.iter('geometry'):
                geometry_tag.find('mesh').attrib['filename'] = f'custom_models/{suffix_free_new_link_name}.stl'

            # update link origin if necessary
            if long_link_to_link_pos[suffix_free_new_link_name]:
                new_link_pos = long_link_to_link_pos[suffix_free_new_link_name]
                link_xml.find('visual').find('origin').attrib['xyz'] = new_link_pos
                link_xml.find('collision').find('origin').attrib['xyz'] = new_link_pos
            
            # update link length extension color to orange
            link_xml.find('visual').find('material').find('color').attrib['rgba'] = "0.878 0.467 0.0 1.0"
            
            # update adjacent joints to reference the new link name
            child_joint = [j for j in urdf.joint_map.values() if j.parent == link_name]
            if len(child_joint) > 0:
                assert len(child_joint) == 1
                child_joint_xml = joint_name_to_joint_xml[child_joint[0].name]
                child_joint_xml.find('parent').attrib['link'] = new_link_name
            parent_joint = [j for j in urdf.joint_map.values() if j.child == link_name][0]
            parent_joint_xml = joint_name_to_joint_xml[parent_joint.name]
            parent_joint_xml.find('child').attrib['link'] = new_link_name

            # update child joint to have longer offset due to extended joint
            if len(child_joint) > 0:
                child_joint_xml.find('origin').attrib['xyz'] = long_link_name_to_child_joint_pos[suffix_free_new_link_name]

            # update tree to have new link name
            tree_str = tree_str.replace(' ' + link_name + '\n', ' ' + new_link_name + '\n')
        
        id_str = f'{config_count:03}'
        
        # write URDF
        urdf_name = f'{id_str}.urdf'
        fpath = os.path.join(urdf_dir, urdf_name)
        modified_xml.write(fpath, pretty_print=True)

        # write tree text file
        fpath = os.path.join(tree_dir, f'{id_str}.txt')
        with open(fpath, 'w') as f:
            f.write(f'Link length variant of {embodiment_name}\n\n')
            f.write(tree_str)

        # write yaml config
        fpath = os.path.join(config_dir, f'{id_str}.yaml')

        with open(os.path.join(config_dir, f'{embodiment_name}.yaml')) as f:
            config = yaml.safe_load(f)

        hand_asset_path = os.path.join(relative_asset_dir, 'urdf', f'{id_str}.urdf')
        config['asset']['handAsset'] = hand_asset_path
        config['embodiment_name'] = id_str
        
        with open(fpath, 'w') as f:
            f.write("# @package _global_.task.env\n\n")
            yaml.dump(config, f, default_flow_style=None)

# Generate length variations for 001
saved_config_count = config_count
gen_length_variations('001', ['mcp_joint', 'dip', 'fingertip'])
print(f'Generated {config_count - saved_config_count} different link length variations from 001')

# Generate length variations for validation embodiments
validation_embodiments = ['004', '024', '046', '071', '098', '122', '139', '142', '200', '204']
train_embodiments = ['001', '002', '003', '006', '007', '008', '009', '010', '011', '012', '013', '014', '021', '022', '023', '026', '051', '056', '095', '103', '119', '120', '121', '123', '124', '125', '126', '127', '129', '130', '131', '132', '134', '140', '141', '144', '154', '164', '165', '169', '174', '189', '194', '221']
for embodiment_name in validation_embodiments + train_embodiments:
    saved_config_count = config_count
    
    # start at a high requirement for number of changed components and decrease the requirement until at least one embodiment is generated
    min_modifications = 3 + 1 # + 1 to account for the -1 at start of loop
    while saved_config_count - config_count == 0:
        min_modifications -= 1
        assert min_modifications > 0
        min_non_fingertip_modifications = max(1, min_modifications - 1)
        gen_length_variations(embodiment_name,
                              ['mcp_joint', 'dip', 'fingertip', 'thumb_dip', 'thumb_fingertip'],
                              min_modifications=min_modifications,
                              shuffle=True,
                              max_generated_embodiments=1,
                              max_fingers_at_max_modifications=1,
                              min_non_fingertip_modifications=min_non_fingertip_modifications,
                              max_modifications=5)
    print(f'Generated {config_count - saved_config_count} different link length variations from {embodiment_name} with at least {min_modifications} length extensions of which at least {min_non_fingertip_modifications} are non fingertip extensions')

print(f'This script resulted in {config_count} total hardware configurations located at:\n - {urdf_dir}\n - {tree_dir}\n - {config_dir}')
