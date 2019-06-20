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
mount_FILENAME=docker-mountEFSdrive.sh
if [ -f $mount_FILENAME ];   then
      sudo chmod a+x $mount_FILENAME
      bash ./$mount_FILENAME
fi
cd rl_baselines_ad

#pip install --ignore-installed -e ./gym/


#pip install -r requirements.txt


bash ./run_baseline.sh $worker_numbers

# Create a Volume where we will clone all the code
#sudo docker volume create --name rl_baselines_v



#docker volume create my-vol
# if [ "$1" != "" ]; then
#     echo "Docker Image ID is:" $1
#     docker cp docker-script.sh $1:/docker-script.sh
#     docker run --entrypoint "/bin/bash" -it munirjojoverge/rl_baselines
# else
#     echo "Docker Image ID is empty"
# fi


