# Saves a picture of the given LEAP Hand configuration at a pose from its grasp cache

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR
cd ..

GPU=${1:-0}
EMBODIMENT=${2:-001}
GRASP_INDEX=${3:-0}
CACHE_NAME=${4:-leap_hand_in_palm_cube_grasp_50k_s09}

POSE=$(cd scripts && python get_leap_grasp_cache_pose.py --config_name $EMBODIMENT --cache_name $CACHE_NAME.npy --grasp_index $GRASP_INDEX)

python viewer.py --asset leap/leap_hand/generated/urdf/$EMBODIMENT.urdf --static True --rot_deg 180 --cam_dist 0.3 --screenshot --screenshot_path=$SCRIPT_DIR/tmp/vis_leap_grasp_cache/$EMBODIMENT/$CACHE_NAME/$GRASP_INDEX.png --gpu $GPU --pose $POSE
