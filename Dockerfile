FROM tensorflow/tensorflow:latest-py3 AS base 
# stage 1. Multi stage docker build
#MAINTAINER Munir Jojo-Verge <munir-jojoverge@havalus.com>
LABEL authors="first Munir Jojo-Verge <munir-jojoverge@havalus.com>,second Pinaki Gupta <pinaki.gupta@havalus.com>"

RUN apt-get update && apt-get install -y \
	sudo \
	libblacs-mpi-dev \
	git

# Install Gym libraries
RUN apt-get update && apt install -y python3-dev zlib1g-dev libjpeg-dev cmake swig python-pyglet python3-opengl libboost-all-dev libsdl2-dev \
    libosmesa6-dev patchelf ffmpeg xvfb nfs-common autofs
RUN apt-get update && apt-get install -y python-pip
RUN pip3 install tensorflow  

FROM base AS intermediate
# stage 2

RUN mkdir /mnt/datastore &&  touch /root/fstab  && echo "LABEL=cloudimg-rootfs / ext4 defaults,discard 0 0" >> /etc/fstab \
&& echo "fs-137189bb.efs.us-west-2.amazonaws.com:/ /mnt/datastore/ nfs4 auto,nofail,noatime,nolock,intr,tcp,actimeo=1800,nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport 0 0 ls /mnt/datastore/" >> /etc/fstab

FROM intermediate
COPY ./docker-ini-script.sh /
COPY ./docker-start.sh /
COPY ./docker-mountEFSdrive.sh /



#ENTRYPOINT ["/docker-start.sh"]
# ENTRYPOINT ["/bin/bash", "-c", "echo Hello, $name! Welcome to RL in Autonomous Driving by Munir Jojo-Verge"]
# CMD [ "/bin/bash" ]





