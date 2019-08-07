#!/bin/bash
# This will be executed the very first time to create all the needed folders, clone the repo, etc..

# Clone the project
#git clone --recurse-submodules https://gitlab.com/havalus-gemini/behavior/rl_baselines_ad.git




# run test
echo "###########################################"
echo "## Running rl_baselines (Crtl+c to stop) ##"
echo "###########################################"
numprocessors=$(nproc --all)
if [ $# \> 0 ]; then
    worker_numbers=$1
  else
    worker_numbers=$numprocessors
fi


#mpirun -bind-to none -np $numprocessors --allow-run-as-root python baselines_run.py
#mount_FILENAME=docker-mountEFSdrive.s
#if [ -f $mount_FILENAME ];   then
#      sudo chmod a+x $mount_FILENAME
#      bash ./$mount_FILENAME
#fi
cd open_ai_baselines
if cd baselines; then
	git pull; 
	cd ..;
else 
  git clone https://github.com/openai/baselines.git .;
fi


pip install --ignore-installed -e .
cd ..

sudo apt-get -y install tmux

EXECDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
#bash $EXECDIR/run_baseline.sh $worker_numbers

cp -a /tmp/.aws/. ~/.aws
cp -a /tmp/.ssh/. ~/.ssh
chown -R root:root ~/.ssh
chown -R root:root ~/.aws
