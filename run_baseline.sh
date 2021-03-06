#!/bin/sh
outputfile="output.txt"
runfile="baselines_run.py"
pdb_commands="-m pdb -c continue"

# Automatically parse the yaml file and get the remote mount path
source yaml.sh
yaml_key_val="$(parse_yaml "$ray_yaml_file" )" #parse_yaml script parses the ray yaml file
while IFS=' ' read -ra ADDR; do
     	for i in "${ADDR[@]}"; do
              if [[ $i == *"min_workers"* ]]; then
                 IFS='min_workers: ' read min_cluster_nodes_i <<< "$i"
              fi
      done
 done <<< "$yaml_key_val"

min_cluster_nodes=$(sed -e 's/min_workers=(\(.*\))/\1/' <<< "$min_cluster_nodes_i")


if [ $# \> 0 ]
  then
    worker_numbers=$1
  else
    echo "MPI not running"
    python -W ignore  $runfile  2>&1 | tee  $outputfile
    return
fi

echo "number of workers are " "$worker_numbers"



# mpirun -bind-to none -np 4  python -W ignore baselines_run.py  2>&1 | tee  output.txt
echo "Running all planning modules ... " &
#xterm -sl 10000  -geometry 60x10+10+10  -fa 'Monospace' -fs 10 -title "RL" -hold -e " mpirun -bind-to none -np 10  python -W ignore ./baselines_run.py  2>&1 | tee  output.txt " ;
export OMP_NUM_THREADS=1;
export USE_SIMPLE_THREADED_LEVEL=1;


if (( $worker_numbers > 1 ));  then
    echo "MPI running"
    mpirun  -bind-to none -np $worker_numbers --allow-run-as-root  python -W ignore $runfile  2>&1 | tee  $outputfile
  else
    echo "MPI not running. This can be because Ray is running or there is only 1 cpu allocated to this machine"
    python -W ignore  $runfile  min_cluster_nodes=$min_cluster_nodes init_cluster_nodes=$min_cluster_nodes 2>&1 | tee  $outputfile
fi


