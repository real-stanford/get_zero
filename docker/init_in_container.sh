#!/bin/bash

cd /home/user/get_zero && pip install -e .
cd /opt/isaacgym/docs && python -m http.server 3000 & # run http server in background with Isaac Gym documentation
