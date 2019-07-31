#!/bin/sh

if [ $# \> 0 ]
  then
    ray_yaml_file=$1
  else
    ray_yaml_file="Ray-Cluster.yaml"
fi

ray up -y $ray_yaml_file
ray exec --docker $ray_yaml_file "cd rl_baselines_ad;./run_baseline.sh"
bash ray_sync.sh $ray_yaml_file
ray down -y $ray_yaml_file
