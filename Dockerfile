FROM tensorflow/tensorflow:latest
MAINTAINER Munir Jojo-Verge <munir-jojoverge@havalus.com>

RUN apt-get update && apt-get install -y \
	sudo \
	libblacs-mpi-dev \
	git

# Install Gym libraries
RUN apt install -y python3-dev zlib1g-dev libjpeg-dev cmake swig python-pyglet python3-opengl libboost-all-dev libsdl2-dev \
    libosmesa6-dev patchelf ffmpeg xvfb

COPY ./docker-ini-script.sh /
# ENTRYPOINT ["/docker-ini-script.sh"]

# ENTRYPOINT ["/bin/bash", "-c", "echo Hello, $name! Welcome to RL in Autonomous Driving by Munir Jojo-Verge"]

# CMD [ "/bin/bash" ]





