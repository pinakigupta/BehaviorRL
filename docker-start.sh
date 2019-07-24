#!/bin/bash

## Install NVIDIA-docker

# Add the package repositories
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update

# Install nvidia-docker2 and reload the Docker daemon configuration
sudo apt-get install -y nvidia-docker2
sudo pkill -SIGHUP dockerd

sudo apt autoremove

HOST_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo $HOST_DIR

# Clone all

# if [ -d "$HOST_DIR/rl_baselines_ad" ] 
# then
#     echo "rl_baselines_ad already cloned" 
# else
#     git clone --recurse-submodules -j8 https://gitlab.com/havalus-gemini/behavior/rl_baselines_ad.git
# fi 

if [ -d "$HOST_DIR/gym" ] 
then
    echo "gym already cloned" 
else    
    git clone --recurse-submodules https://github.com/openai/gym.git
fi

# We are ready to run the main docker container

xhost +
sudo docker run -it --runtime=nvidia -v $HOST_DIR:/rl_baselines_ad -e DISPLAY=unix$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --name PEPE munirjojoverge/rl_baselines:demo /bin/bash
