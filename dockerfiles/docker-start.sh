#!/bin/bash

#HOST_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
sudo bash ./docker-PreStart.sh ${HOST_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"}

EC2Instance=false
# This first, simple check will work for many older instance types.
if [ -f /sys/hypervisor/uuid ]; then
  # File should be readable by non-root users.
  if [ `head -c 3 /sys/hypervisor/uuid` == "ec2" ]; then
    EC2Instance=true
  fi
# This check will work on newer m5/c5 instances, but only if you have root!
elif [ -r /sys/devices/virtual/dmi/id/product_uuid ]; then
  # If the file exists AND is readable by us, we can rely on it.
  if [ `head -c 3 /sys/devices/virtual/dmi/id/product_uuid` == "EC2" ]; then
    EC2Instance=true
  fi
fi

echo "We are ready to run the main docker container"
if [ $EC2Instance == true ]; then
        echo "docker run EC2Instance version"
	sudo docker run -it  pinakigupta/rl_baselines /bin/bash
else
        echo "docker run Ubuntu version"
	xhost +
	sudo docker run -it --runtime=nvidia --volume $HOST_DIR:/rl_baselines_ad -e DISPLAY=unix$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix  pinakigupta/rl_baselines /bin/bash
fi


