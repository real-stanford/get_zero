#!/bin/bash

# container name
CONTAINER_SUFFIX=${1:-""} # suffix added to the container name
CONTAINER_NAME="get_zero_container"
if [ -n "$CONTAINER_SUFFIX" ]; then
    CONTAINER_NAME="${CONTAINER_NAME}_${CONTAINER_SUFFIX}"
fi

# If .Xauthority changes in the host machine, then we can run this script to copy the updates back into the container
# This is done since mounting the .Xauthority file directly with -v in Docker wasn't synchronizing updates (likely due to: https://medium.com/@jonsbun/why-need-to-be-careful-when-mounting-single-files-into-a-docker-container-4f929340834) and the fact that .Xauthority uses a swap file .Xauthority-n when making changes which messes with Docker's -v mounting since .Xauthority is temporarily deleted
USERNAME=user
CONTAINERID=$(docker ps -aqf "name=^${CONTAINER_NAME}$")
docker cp $HOME/.Xauthority $CONTAINERID:/home/$USERNAME/.Xauthority

# .fix_display.sh script is loaded by .bashrc so whenever a new shell is created it will have correct display set
docker exec -d $CONTAINERID bash -c "echo \"export DISPLAY=$DISPLAY\" > /home/$USERNAME/.fix_display.sh"
