FROM ubuntu:18.04
MAINTAINER Munir Jojo-Verge <munir-jojoverge@havalus.com>

RUN apt-get update

RUN apt install -y git

# Install Gym
RUN apt install -y python3-dev zlib1g-dev libjpeg-dev cmake swig python-pyglet python3-opengl libboost-all-dev libsdl2-dev libosmesa6-dev patchelf ffmpeg xvfb

RUN apt-get -y install zlib1g python-opengl libsdl2-2.0-0 libboost-python1.58.0 libboost-thread1.58.0 libboost-filesystem1.58.0 libboost-system1.58.0 fluidsynth build-essential wget unzip && \
    /usr/local/bin/pip --no-cache-dir install --upgrade 'gym[all]' && \
    dpkg --purge libsdl2-dev libboost-all-dev wget unzip && \
    apt-get -y autoremove && \
    dpkg --purge build-essential swig python-dev cmake zlib1g-dev && \
    apt-get -y autoremove && \
    apt-get clean && \
    rm -r /var/lib/apt/lists/* /root/.cache/pip/

# Install pip
RUN apt install -y python3-pip

# Clone and install rl_baselines_AD
RUN mkdir ~/rl_baselines_AD
RUN cd ~/rl_baselines_AD
RUN git clone --recurse-submodules -j8 https://gitlab.com/havalus-gemini/behavior/rl_baselines_ad.git 

# Install baselines
RUN cd ~/rl_baselines_AD/open_ai_baselines/baselines
RUN pip install tensorflow-gpu/
RUN pip install -e .
RUN cd open_ai_baselines
RUN cd ..


