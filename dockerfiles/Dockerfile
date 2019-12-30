FROM tensorflow/tensorflow:latest-py3 AS base 
# stage 1. Multi stage docker build


RUN apt-get update && apt-get install -y \
	sudo \
	libblacs-mpi-dev \
	git

# Install Gym libraries
RUN apt-get update && apt install -y python3-dev zlib1g-dev libjpeg-dev cmake swig python-pyglet python3-opengl libboost-all-dev libsdl2-dev \
    libosmesa6-dev patchelf ffmpeg xvfb nfs-common autofs gedit rsync
RUN apt-get update && apt-get install -y python-pip
RUN pip install -U ray ray[debug] gitpython
RUN mkdir /root/.aws
COPY ./requirements.txt /
RUN pip install -r requirements.txt
RUN pip install --ignore-installed gym
RUN pip3 install tensorflow tensorflow-gpu pillow requests setproctitle psutil lz4 awscli boto boto3 tabulate
RUN git clone https://github.com/lgsvl/PythonAPI.git && pip3 install --user PythonAPI/

FROM base AS intermediate
# stage 2
#RUN apt-get update && apt-get install -y software-properties-common apt-transport-https wget
#RUN mkdir /mnt/datastore &&  touch /root/fstab  && echo "LABEL=cloudimg-rootfs / ext4 defaults,discard 0 0" >> /etc/fstab \
#&& echo "fs-137189bb.efs.us-west-2.amazonaws.com:/ /mnt/datastore/ nfs4 auto,nofail,noatime,nolock,intr,tcp,actimeo=1800,nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport 0 0 #ls /mnt/datastore/" >> /etc/fstab




FROM intermediate
COPY ./docker-start.sh /
COPY ./docker-mountEFSdrive.sh /







