#!/bin/bash
set -e
set -u
SCRIPTROOT="$( cd "$(dirname "$0")" ; pwd -P )"
cd "${SCRIPTROOT}/.."

docker build --build-arg USERID=$(id -u) --network host -t get_zero -f docker/Dockerfile .
