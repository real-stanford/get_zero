# Modified from IsaacGymEnvs
"""Installation script for the 'get_zero' python package."""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from setuptools import setup, find_packages

import os

root_dir = os.path.dirname(os.path.realpath(__file__))

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "isaacgym==1.0rc4",
    "isaacgymenvs==1.5.1",
    "torch",
    "omegaconf",
    "hydra-core>=1.2",
    "rl-games==1.6.1",
    "pyvirtualdisplay",
    "Pillow>=10.3.0", # need a version that has XCB support included if using virtual display capture
    "imageio-ffmpeg",
    "treelib",
    "tbparse",
    "wandb>=0.12.5", # due to https://docs.wandb.ai/guides/track/log/distributed-training,
    "scikit-network",
    "numpy<1.24", # due to deprecation of `np.float`, which is used in Isaac Gym
    "yourdfpy",
    "trimesh",
    "rich",
    "transformers",
    "pytorch_kinematics<=0.7.1", # newer versions limit forward kinematics to serial chain (see https://github.com/UM-ARM-Lab/pytorch_kinematics/commit/baa6ced4d837e09e51f18dd3c93f41fe7b75f20f and get_zero/utils/forward_kinematics.py)
    "arm-pytorch-utilities",
    "dynamixel_sdk",
    "pyrealsense2"
    ]

# Installation operation
setup(
    name="get_zero",
    author="Austin Patel",
    version="1.0.0",
    description="Multi-embodiment policy learning",
    keywords=["robotics", "rl"],
    include_package_data=True,
    python_requires=">=3.6",
    install_requires=INSTALL_REQUIRES,
    packages=find_packages("."),
    classifiers=["Natural Language :: English", "Programming Language :: Python :: 3.6, 3.7, 3.8"],
    zip_safe=False,
)

# EOF
