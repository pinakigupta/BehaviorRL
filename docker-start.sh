#!/bin/bash

## Install NVIDIA-docker

UNAME=$(uname | tr "[:upper:]" "[:lower:]")


# If Linux, try to determine specific distribution
if [ "$UNAME" == "linux" ]; then
    # If available, use LSB to identify distribution
    if [ -f /etc/lsb-release -o -d /etc/lsb-release.d ]; then
        export DISTRO=$(lsb_release -i | cut -d: -f2 | sed s/'^\t'//)
    # Otherwise, use release info file
    else
        export DISTRO=$(ls -d /etc/[A-Za-z]*[_-][rv]e[lr]* | grep -v "lsb" | cut -d'/' -f3 | cut -d'-' -f1 | cut -d'_' -f1)
    fi
fi
# For everything else (or if above failed), just use generic identifier
[ "$DISTRO" == "" ] && export DISTRO=$UNAME
unset UNAME

echo $DISTRO

if [ "$DISTRO" == "AmazonAMI" ]; then # Can also support Centos
	# If you have nvidia-docker 1.0 installed: we need to remove it and all existing GPU containers
	docker volume ls -q -f driver=nvidia-docker | xargs -r -I{} -n1 docker ps -q -a -f volume={} | xargs -r docker rm -f
	sudo yum remove nvidia-docker

	# Add the package repositories
	distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
	curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo | \
	  sudo tee /etc/yum.repos.d/nvidia-docker.repo

	# Install nvidia-docker2 and reload the Docker daemon configuration
	sudo yum install -y nvidia-docker2

	# Test nvidia-smi with the latest official CUDA image
	#docker run --runtime=nvidia --rm nvidia/cuda:9.0-base nvidia-smi

	sudo yum install -y python36-devel.x86_64 zlib-devel libjpeg-devel cmake swig python-pyglet python3-opengl libboost-all-dev libsdl2-dev \
    libosmesa6-dev patchelf ffmpeg xvfb

else  # Ubuntu 
	# Add the package repositories
	curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
	  sudo apt-key add -
	distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
	curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
	  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
	sudo apt-get update

	# Install nvidia-docker2 and reload the Docker daemon configuration
	sudo apt-get install -y nvidia-docker2
	sudo apt autoremove
fi

	sudo pkill -SIGHUP dockerd




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
if [ "$DISTRO" == "AmazonAMI" ]; then
        echo "docker run Amazon AMI version"
	sudo docker run -it --runtime=nvidia -v $HOST_DIR:/rl_baselines_ad -v /tmp/.X11-unix:/tmp/.X11-unix munirjojoverge/rl_baselines /bin/bash
else
        echo "docker run Ubuntu version"
	xhost +
	sudo docker run -it --runtime=nvidia -v $HOST_DIR:/rl_baselines_ad -e DISPLAY=unix$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix  munirjojoverge/rl_baselines /bin/bash
fi

