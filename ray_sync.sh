#!/bin/sh

if [ $# \> 0 ]
  then
    ray_yaml_file=$1
  else
    ray_yaml_file="Ray-Cluster.yaml"
fi

# Automatically parse the yaml file and get the remote mount path
source yaml.sh
yaml_key_val="$(parse_yaml "$ray_yaml_file" )"
#echo $yaml_key_val
while IFS=' ' read -ra ADDR; do
     	for i in "${ADDR[@]}"; do
              if [[ $i == *"/mnt"* ]]; then
                 IFS=':' read -ra REMOTE_DOWNLOAD_PATH <<< "$i"
              fi
      done
 done <<< "$yaml_key_val"
REMOTE_DOWNLOAD_PATH="$REMOTE_DOWNLOAD_PATH/"

LOCAL_DOWNLOAD_PATH="$PWD/ray_results/"
echo "LOCAL_DOWNLOAD_PATH" $LOCAL_DOWNLOAD_PATH

# You can manually override the remote mount path here 
#REMOTE_DOWNLOAD_PATH="/mnt/datastore/groups/behavior/Pinaki/rl_baselines/rl_baselines_ad/ray_results/"  
echo "REMOTE_DOWNLOAD_PATH" $REMOTE_DOWNLOAD_PATH

ray exec $ray_yaml_file "sudo chmod -R a+rwx $REMOTE_DOWNLOAD_PATH"
ray rsync_down $ray_yaml_file $REMOTE_DOWNLOAD_PATH $LOCAL_DOWNLOAD_PATH # rsync_up will upload
