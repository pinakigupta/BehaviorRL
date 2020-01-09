#!/bin/bash

docker_name=lgsvl_docker_local
HOST_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
HOST_DIR=$HOST_DIR/lg-sim
echo "HOST_DIR" $HOST_DIR
echo "We are ready to run the main docker container"
sudo docker stop $docker_name && sudo docker container rm $docker_name
docker run -it\
     --name $docker_name\
     --runtime=nvidia\
     --net=host\
     -e DISPLAY\
     -e XAUTHORITY=/tmp/.Xauthority\
     -v ${XAUTHORITY}:/tmp/.Xauthority\
     -v /tmp/.X11-unix:/tmp/.X11-unix\
     -v $HOST_DIR:/lg-sim\
     -v lgsvlsimulator:/root/.config/unity3d\
     lgsvlsimulator


