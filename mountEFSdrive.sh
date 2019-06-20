#!/bin/bash
codedir="/rl_baselines_ad"
sudo apt-get install -y nfs-common autofs
sudo mkdir $codedir
sudo mount -t nfs4 -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport fs-137189bb.efs.us-west-2.amazonaws.com://groups/behavior/Pinaki/RL/rl_baselines_ad $codedir
sudo $codedir/docker-ini-script.sh 1

