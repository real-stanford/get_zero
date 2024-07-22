# Goes through every genereated LEAP hand configuration and generates a rendered image

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
echo "concurrent runs" $CONCURRENT_RUNS

COMMAND="gpus=$GPUS"
COMMAND+=' && python viewer.py --asset leap/leap_hand/generated/urdf/$1 --static True --rot_deg 180 --cam_dist 0.22 --screenshot --screenshot_transparent --width 1024 --aspect_ratio 1 --no_time_in_screenshot_name --screenshot_path=assets/leap/leap_hand/generated/image --gpu=${gpus[$((($0 -1) % ${#gpus[@]}))]} || exit 0'

for ((i = $START ; i <= $END ; i++)); do
    fname=$(printf "%03d" $i).urdf
    echo $fname
done | cat -n | xargs -P $CONCURRENT_RUNS -l bash -c "$COMMAND"
