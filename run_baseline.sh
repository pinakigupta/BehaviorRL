#!/bin/sh


if [ $# > 0 ]
  then
    worker_numbers=$1
  else
    worker_numbers=1 
fi

'''
if [ $# > 1 ]
  then
     conda_env=$2
     eval "$(conda shell.bash hook)"
     conda activate $conda_env
  else
     echo "Conda env not automatically updated"
fi'''

outputfile="output.txt"
runfile="./baselines_run.py"
pdb_commands="-m pdb -c continue"

# mpirun -bind-to none -np 4  python -W ignore baselines_run.py  2>&1 | tee  output.txt
echo "Running all planning modules ... " &
#xterm -sl 10000  -geometry 60x10+10+10  -fa 'Monospace' -fs 10 -title "RL" -hold -e " mpirun -bind-to none -np 10  python -W ignore ./baselines_run.py  2>&1 | tee  output.txt " ;
export OMP_NUM_THREADS=1;
export USE_SIMPLE_THREADED_LEVEL=1;

if [ $worker_numbers -eq 1 ]
  then
    python -W ignore  $runfile  2>&1 | tee  $outputfile
  else
    mpirun  -np $worker_numbers  python -W ignore $runfile  2>&1 | tee  $outputfile
fi


