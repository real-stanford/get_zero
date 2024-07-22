#!/bin/bash

# container name
CONTAINER_SUFFIX=${1:-""} # suffix added to the container name
CONTAINER_NAME="get_zero_container"
if [ -n "$CONTAINER_SUFFIX" ]; then
    CONTAINER_NAME="${CONTAINER_NAME}_${CONTAINER_SUFFIX}"
fi

CONTAINERID=$(docker ps -aqf "name=^${CONTAINER_NAME}$")
echo $CONTAINERID
docker stop -t 1 $CONTAINERID
