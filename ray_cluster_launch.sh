#!/bin/sh

if [ $# \> 0 ]
  then
    ray_yaml_file=$1
  else
    ray_yaml_file="Ray-Cluster.yaml" # Your default cluster yaml file
fi

if [ $# \> 1 ]
  then
    exec_cmd=$2
  else
    exec_cmd="cd BehaviorRL;./run_baseline.sh" # Your default ray exec cmnd
fi

#copying from the volatile docker container to the persistent mount drive
sync_cmd="rsync -a  /BehaviorRL/ray_results/   /rl_baselines_ad/ray_results/" 


outputfile="output_cluster.txt"
dockerfiles/docker-ini-script.sh # script for custom docker setup. Not needed for 
#general usage
ray up -y $ray_yaml_file
#head_ip=ray get_head_ip $ray_yaml_file
# Python API ray_cluster_status_check() checks against ray_yaml_file to see if 
# min number of nodes are up or waits (embedded within the exec cmd step)
# Ideally we want to take this decision in shell before launching the ray exec
# command. Otherwise this will need to be put inside the dev code (ray_cluster_status_check())
ray exec --docker $ray_yaml_file "$exec_cmd"  #2>&1 | tee  $outputfile 
ray exec --docker $ray_yaml_file "$sync_cmd" &&
bash ray_sync.sh $ray_yaml_file  &&
ray down -y $ray_yaml_file 
