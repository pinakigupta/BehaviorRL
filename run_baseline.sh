#!/bin/sh

# mpirun -bind-to none -np 4  python -W ignore baselines_run.py  2>&1 | tee  output.txt
OPENAI_LOGDIR=$HOME/logs/two-way-ppo2 OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' mpirun -bind-to none -np 1  python -W ignore baselines_run.py  2>&1 | tee  output.txt
