FROM tensorflow/tensorflow:latest-py3
#MAINTAINER Munir Jojo-Verge <munir-jojoverge@havalus.com>
LABEL authors="first Munir Jojo-Verge <munir-jojoverge@havalus.com>,second Pinaki Gupta <pinaki.gupta@havalus.com>"

RUN apt-get update && apt-get install -y \
	sudo \
	libblacs-mpi-dev \
	git

# Install Gym libraries
RUN apt-get update && apt install -y python3-dev zlib1g-dev libjpeg-dev cmake swig python-pyglet python3-opengl libboost-all-dev libsdl2-dev \
    libosmesa6-dev patchelf ffmpeg xvfb 
RUN apt-get update && apt-get install -y python-pip
RUN pip3 install tensorflow  

COPY ./docker-ini-script.sh /
COPY ./docker-start.sh /
#ENTRYPOINT ["/docker-start.sh"]

# ENTRYPOINT ["/bin/bash", "-c", "echo Hello, $name! Welcome to RL in Autonomous Driving by Munir Jojo-Verge"]

# CMD [ "/bin/bash" ]





