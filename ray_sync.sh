#!/bin/sh

if [ $# \> 0 ]
  then
    ray_yaml_file=$1
  else
    ray_yaml_file="Ray-Cluster.yaml"
fi

LOCAL_DOWNLOAD_PATH="$PWD/ray_results/"
echo "LOCAL_DOWNLOAD_PATH" $LOCAL_DOWNLOAD_PATH

REMOTE_DOWNLOAD_PATH="/mnt/datastore/groups/behavior/Pinaki/rl_baselines/rl_baselines_ad/ray_results/"
ray exec $ray_yaml_file "sudo chmod -R a+rwx $REMOTE_DOWNLOAD_PATH"
ray rsync_down $ray_yaml_file $REMOTE_DOWNLOAD_PATH $LOCAL_DOWNLOAD_PATH # rsync_up will upload
