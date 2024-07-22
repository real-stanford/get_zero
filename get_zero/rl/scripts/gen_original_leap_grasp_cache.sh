# Generates grasp cache for the original LEAP hand configuration
# single optional argument for GPU index, defaults to GPU0

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR
cd ..

GPU=${1:-0}

echo "Generating grasp cache for the original LEAP Hand configuration"

for cube_scale in 0.9 0.95 1.0 1.05 1.1 
do
    python train.py task=LeapHandGrasp test=true pipeline=cpu gpu=$GPU num_envs=1024 task.env.baseObjScale=$cube_scale embodiment=LeapHand/OrigRepo wandb_activate=False record_video=False
done
