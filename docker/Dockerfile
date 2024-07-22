# File adopted from isaacgym
# pytorch container information: https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html

ARG USERNAME=user

FROM nvcr.io/nvidia/pytorch:23.04-py3 AS base
ARG USERNAME
ENV DEBIAN_FRONTEND=noninteractive

# Fixes issue where importing torch fails to find hpcx
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/hpcx/ompi/lib"

# dependencies for gym
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
 libxcursor-dev \
 libxrandr-dev \
 libxinerama-dev \
 libxi-dev \
 mesa-common-dev \
 zip \
 unzip \
 make \
 gcc-8 \
 g++-8 \
 vulkan-utils \
 mesa-vulkan-drivers \
 pigz \
 git \
 libegl1 \
 git-lfs

# Additional dependencies
RUN apt-get install -y python3-tk

# Force gcc 8 to avoid CUDA 10 build issues on newer base OS
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8

# WAR for eglReleaseThread shutdown crash in libEGL_mesa.so.0 (ensure it's never detected/loaded)
# Can't remove package libegl-mesa0 directly (because of libegl1 which we need)
RUN rm /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0 /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0.0.0 /usr/share/glvnd/egl_vendor.d/50_mesa.json

COPY docker/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

RUN apt-get update && apt-get install -y x11-apps tmux xvfb htop

WORKDIR /opt

# setup user
# TODO: one limitation of this apporach is that the user id is fixed (it needs to match the host user ID which is done automatically in `build.sh`), which makes this container less portable for built images running on different machines with different user IDs. This is because user ID has to match when so that files mounted in the Docker container can be properly accessed. To fix this, you could use https://github.com/boxboat/fixuid
ARG USERID=1000
RUN useradd -u $USERID --create-home $USERNAME
ENV PATH="/home/$USERNAME/.local/bin:$PATH"

# IsaacGymEnvs clone and change permissions so user can pip install
RUN git clone https://github.com/NVIDIA-Omniverse/IsaacGymEnvs.git
RUN chown -R $USERNAME /opt/IsaacGymEnvs

# copy repo to container workdir
COPY --chown=$USERNAME dependencies/isaacgym/ isaacgym/

# rl_games clone and give permission to user
RUN git clone https://github.com/Denys88/rl_games.git
RUN chown -R $USERNAME /opt/rl_games

# LEAP hand sim clone and give permission to user
RUN git clone https://github.com/leap-hand/LEAP_Hand_Sim.git
RUN chown -R $USERNAME /opt/LEAP_Hand_Sim

# swap to user
USER $USERNAME

RUN /usr/bin/python -m pip install --upgrade pip

# install specific version of rl games so we can debug into it (f5bd8f2 is v1.6.1)
RUN cd rl_games && git reset --hard f5bd8f2 && pip install -e .

# install isaac gym and isaac gym envs
RUN cd isaacgym/python && pip install -q -e .

# install specific version of Isaac Gym Envs so we can debug into it (aeed298 is v1.5.1)
RUN cd IsaacGymEnvs && git reset --hard aeed298 && pip install -e .

# install specific version of LEAP Hand sim codebase (no deps since LEAP hand requires a rl_games version that is earlier than what IsaacGymEnvs requires). The rest of the LEAP hand deps will be installed already through other package installations
RUN cd LEAP_Hand_Sim && git reset --hard ebc49e0 && pip install --no-deps -e .

ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all

# file to source for updaing DISPLAY variable for X11 forwarding (see fix_xauthority.sh)
RUN echo . /home/$USERNAME/.fix_display.sh >> /home/$USERNAME/.bashrc && touch /home/$USERNAME/.fix_display.sh

RUN pip install gpustat

WORKDIR /home/$USERNAME

# Clone LEAP Hand API
WORKDIR /opt
USER root
RUN git clone https://github.com/leap-hand/LEAP_Hand_API.git && cd LEAP_Hand_API && git reset --hard a8a5c47
RUN chown -R $USERNAME LEAP_Hand_API

# Dynamixel Wizard download (run ./DynamixelWizard2Setup-x86_64 in container with display to install)
RUN apt-get update && apt-get install -y libxkbcommon-x11-0
RUN curl -O -L https://www.dropbox.com/s/dl/csawv9qzl8m8e0d/DynamixelWizard2Setup-x86_64 && chmod 775 DynamixelWizard2Setup-x86_64 && usermod -aG dialout $USERNAME

USER $USERNAME
WORKDIR /home/$USERNAME

# Install librealsense
USER root
RUN apt-get install -y software-properties-common
RUN mkdir -p /etc/apt/keyrings && curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | tee /etc/apt/keyrings/librealsense.pgp > /dev/null
RUN apt-get install apt-transport-https
RUN echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo `lsb_release -cs` main" | tee /etc/apt/sources.list.d/librealsense.list
RUN apt-get update && apt-get install -y librealsense2-dkms librealsense2-utils
USER $USERNAME

# Install fonts
USER root
RUN echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | debconf-set-selections
RUN apt-get install -y ttf-mscorefonts-installer
USER $USERNAME

# either chose to copy a version of get_zero that is in the dependencies folder or clone a version from GitHub
# COPY --chown=$USERNAME dependencies/get_zero/ get_zero/
RUN git clone https://github.com/real-stanford/get_zero.git

CMD [ "sleep", "infinity" ]
