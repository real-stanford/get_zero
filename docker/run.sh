#!/bin/bash
# File adopted from isaacgym
# TODO: currently fails with no display. Add a flag to disable display
set -e
set -u

CONTAINER_SUFFIX=${1:-""} # suffix added to the container name
PROJECT_MOUNT_MODE=${2:-"full"} # `full` means we mount the entire projcect repo, `data` means we mount only the run directories and state logs from the project repo (this is good for running experiments in a separate Docker container while making edits in main container), and `none` means we don't mount any data into the repo so project files are fully isolated

CONTAINER_NAME="get_zero_container"

if [ -n "$CONTAINER_SUFFIX" ]; then
    CONTAINER_NAME="${CONTAINER_NAME}_${CONTAINER_SUFFIX}"
fi

echo "Container name: $CONTAINER_NAME"

USERNAME=user

REPOROOT="$( cd "$(dirname "$0")" ; pwd -P )"
REPOROOT=$(dirname $REPOROOT)

# --privileged is only needed as a hack to get around an issue where nvidia-smi sometimes causes `Failed to initialize NVML: Unknown Error`. See: https://github.com/NVIDIA/nvidia-docker/issues/1671. It's also needed for accessing serial devices
RUNARGS="--gpus=all --privileged --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -d --rm --network=host --name=$CONTAINER_NAME -t get_zero"

# mount repo from host to sync changes to host, the .netrc file to give access to WandB API key, .ssh for ssh connections, and /dev for serial devices
MOUNTARGS="-v $HOME/.netrc:/home/$USERNAME/.netrc -v $HOME/.ssh:/home/$USERNAME/.ssh -v /dev:/dev"

# mount project files
if [ $PROJECT_MOUNT_MODE == "full" ]; then
    PROJECT_MOUNT_ARGS="-v $REPOROOT:/home/$USERNAME/get_zero"
elif [ $PROJECT_MOUNT_MODE == "data" ]; then
    PROJECT_MOUNT_ARGS="-v $REPOROOT/get_zero/distill/runs:/home/$USERNAME/get_zero/get_zero/distill/runs -v $REPOROOT/get_zero/rl/runs:/home/$USERNAME/get_zero/get_zero/rl/runs -v $REPOROOT/get_zero/rl/state_logs:/home/$USERNAME/get_zero/get_zero/rl/state_logs"
elif [ $PROJECT_MOUNT_MODE == "none" ]; then
    PROJECT_MOUNT_ARGS=""
else
    echo "unknown PROJECT_MOUNT_MODE: $PROJECT_MOUNT_MODE"
fi

echo "using display $DISPLAY"
DISPLAYARGS="-v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY"

ENVARGS="-e CUDA_ID_TO_VULKAN_ID" # CUDA_ID_TO_VULKAN_ID maps from CUDA device id to the vulkan graphics device ID that corresonds to each GPU. For example if you have 4 GPU and your graphics device ordering is flipped you would set CUDA_ID_TO_VULKAN_ID="3 2 1 0" before running this script. Setting this value is optional

docker run $ENVARGS $DISPLAYARGS $PROJECT_MOUNT_ARGS $MOUNTARGS $RUNARGS

CONTAINERID=$(docker ps -aqf "name=^${CONTAINER_NAME}$")

echo $CONTAINERID

# update project files to latest version
if [ $PROJECT_MOUNT_MODE != "full" ]; then
    docker exec $CONTAINERID bash -c "cd get_zero && git pull"
fi

# run container initialization script
docker exec $CONTAINERID bash -i /home/$USERNAME/get_zero/docker/init_in_container.sh

# setup .Xauthority
. docker/fix_xauthority.sh $CONTAINER_SUFFIX
