# Generates grasp cache for generated LEAP hand configurations in the specified range

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR
cd ..

MAX_CONFIGURATIONS=633

START=${1:-1}
END=${2:-$MAX_CONFIGURATIONS}
GPUS=${3:-"(0)"}
CONCURRENT_RUNS=${4:-1}

COMMAND="ECHO=$ECHO gpus=$GPUS"
COMMAND+=' && $ECHO python train.py task=LeapHandGrasp test=true pipeline=cpu test_steps=10000 gpu=${gpus[$((($0 -1) % ${#gpus[@]}))]} num_envs=1024 task.env.baseObjScale=$2 embodiment=LeapHand/generated/$1 wandb_activate=False record_video=False || exit 0 && $ECHO ./scripts/vis_generated_leap_grasp_cache_single.sh ${gpus[$((($0 -1) % ${#gpus[@]}))]} $1 0 leap_hand_in_palm_cube_grasp_50k_s${2//.}'
# TODO: this command is repeated in gen_generated_leap_grasp_cache_missing.sh, which could lead to issues if their values change

for ((i = $START ; i <= $END ; i++)); do
    for cube_scale in 0.9 0.95 1.0 1.05 1.1; do
        embodiment_tag=$(printf "%03d" $i)
        echo $embodiment_tag $cube_scale
    done
done | cat -n | xargs -P $CONCURRENT_RUNS -l bash -c "$COMMAND"
