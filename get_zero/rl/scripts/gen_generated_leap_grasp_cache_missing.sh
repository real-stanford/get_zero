# Generates grasp cache for generated LEAP hand configurations that currently don't have succesful results

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR
cd ..

GPUS=${1:-"(0)"}
CONCURRENT_RUNS=${2:-1}

MISSING=$(cd scripts && python leap_grasp_cache_status.py --get_missing)

if [ -z "${MISSING}" ]; then
    echo "entire grasp cache has already been generated!"
else
    echo "MISSING: " $MISSING
    COMMAND="ECHO=$ECHO gpus=$GPUS"
    COMMAND+=' && $ECHO python train.py task=LeapHandGrasp test=true pipeline=cpu test_steps=10000 gpu=${gpus[$((($0 -1) % ${#gpus[@]}))]} num_envs=1024 task.env.baseObjScale=$2 embodiment=LeapHand/generated/$1 wandb_activate=False record_video=False || exit 0 && $ECHO ./scripts/vis_generated_leap_grasp_cache_single.sh ${gpus[$((($0 -1) % ${#gpus[@]}))]} $1 0 leap_hand_in_palm_cube_grasp_50k_s${2//.}'
    # TODO: this command is repeated in gen_generated_leap_grasp_cache_all.sh, which could lead to issues if their values change

    for embodiment_tag in ${MISSING}; do
        for cube_scale in 0.9 0.95 1.0 1.05 1.1; do
            echo $embodiment_tag $cube_scale
        done
    done | cat -n | xargs -P $CONCURRENT_RUNS -l bash -c "$COMMAND"
fi
