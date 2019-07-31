#!/bin/sh

if [ $# \> 0 ]
  then
    ray_yaml_file=$1
  else
    ray_yaml_file="Ray-Cluster.yaml"
fi

outputfile="output_cluster.txt"

ray up -y $ray_yaml_file 2>&1 | tee  $outputfile
sleep 500
ray exec --docker $ray_yaml_file "docker attach $(docker ps -aqf "name=ray_docker");cd rl_baselines_ad;./run_baseline.sh" 2>&1 | tee  $outputfile
bash ray_sync.sh $ray_yaml_file 
ray down -y $ray_yaml_file 
