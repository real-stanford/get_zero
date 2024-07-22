# Rolls out trained LeapHandRot policies for generated LEAP hand configurations to collect training data. Must provide a W&B tag that indicates where the checkpoint will come from. If multiple runs have the same tag and embodiment, then the most recent run will be returned (based on alphabetical sorting of run directory names).

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR
cd ..

MAX_CONFIGURATIONS=633

START=${1:-1}
END=${2:-$MAX_CONFIGURATIONS}
GPUS=${3:-"(0)"}
CONCURRENT_RUNS=${4:-1}
TRAIN_WANDB_TAG=${5:-"leap_train_get_zero"}
TEST_WANDB_TAG_ADDITION=${6:-"rollout"}

# extra args
array=( "$@" )
len=${#array[@]}
EXTRA_ARGS=${array[@]:6:$len}

COMMAND="ECHO=$ECHO gpus=$GPUS EXTRA_ARGS=\"$EXTRA_ARGS\" TRAIN_WANDB_TAG=$TRAIN_WANDB_TAG TEST_WANDB_TAG_ADDITION=$TEST_WANDB_TAG_ADDITION NUM_ENVS=$NUM_ENVS"
COMMAND+=' BEST_RUN_NAME=$(cd scripts && python get_leap_run_dir_from_wandb_tag.py --wandb_tag=$TRAIN_WANDB_TAG --embodiment_name=$1) && $ECHO python train.py task=LeapHandRot gpu=${gpus[$((($0 -1) % ${#gpus[@]}))]} embodiment=LeapHand/generated/$1 test=True checkpoint=runs/$BEST_RUN_NAME/nn/LeapHandRot.pth task.env.logStateInTest=True task.env.logStateSuffix=$BEST_RUN_NAME wandb_tags=\[${TEST_WANDB_TAG_ADDITION}_${TRAIN_WANDB_TAG}\] test_steps=1000 num_envs=5000 $EXTRA_ARGS || exit 0'

for ((i = $START ; i <= $END ; i++)); do
    embodiment_tag=$(printf "%03d" $i)
    echo $embodiment_tag
done | cat -n | xargs -P $CONCURRENT_RUNS -l bash -c "$COMMAND"
