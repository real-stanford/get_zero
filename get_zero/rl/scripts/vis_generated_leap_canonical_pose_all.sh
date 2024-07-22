# Saves a picture of the given LEAP hand configuration at its canonical pose

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR
cd ..

MAX_CONFIGURATIONS=633

START=${1:-1}
END=${2:-$MAX_CONFIGURATIONS}
GPUS=${3:-"(0)"}
CONCURRENT_RUNS=${4:-1}

echo "start" $START
echo "end" $END
echo "GPUs" $GPUS
echo "runs per gpu" $CONCURRENT_RUNS

COMMAND="gpus=$GPUS script_dir=$SCRIPT_DIR"
COMMAND+=' && python viewer.py --asset leap/leap_hand/generated/urdf/$1.urdf --static True --rot_deg 180 --cam_dist 0.3 --screenshot --screenshot_path=$script_dir/tmp/vis_leap_canonical_pose/$1.png --screenshot_viewpoints px py ny pz --gpu=${gpus[$((($0 -1) % ${#gpus[@]}))]} --pose $(cd scripts && python get_generated_leap_canonical_pose.py --config_name $1) || exit 0'

for ((i = $START ; i <= $END ; i++)); do
    fname=$(printf "%03d" $i)
    echo $fname
done | cat -n | xargs -P $CONCURRENT_RUNS -l bash -c "$COMMAND"
