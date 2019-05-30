#!/bin/sh
if [ $# -eq 1 ]
  then
    worker_numbers=$1
  else
    worker_numbers=1 # backup map file
fi

# mpirun -bind-to none -np 4  python -W ignore baselines_run.py  2>&1 | tee  output.txt
echo "Running all planning modules ... " &
#xterm -sl 10000  -geometry 60x10+10+10  -fa 'Monospace' -fs 10 -title "RL" -hold -e " mpirun -bind-to none -np 10  python -W ignore ./baselines_run.py  2>&1 | tee  output.txt " ;
mpirun  -np $worker_numbers  python -W ignore ./baselines_run.py  2>&1 | tee  output.txt

