#!/bin/bash

if [ $# \> 0 ]; then
    extpath=$1
    echo "extpath:" 
    echo $extpath
    echo $#
  else
    extpath="/groups/behavior/Pinaki/RL/rl_baselines_ad"
fi
echo "external path = " $extpath
mountdir="/rl_baselines_ad"
sudo apt-get install -y nfs-common autofs
sudo mkdir $mountdir
sudo mount -t nfs4 -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport fs-137189bb.efs.us-west-2.amazonaws.com:/$extpath $mountdir


