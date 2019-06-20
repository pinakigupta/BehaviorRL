# rl_baselines for Autonomous Driving (AD)
Author: **Munir Jojo-Verge**

## Introduction

Reinforcement Learning Baselines (from OpenAI) applied to Autonomous Driving

This Research is aiming to address RL approaches to solve Urban driving scenarios such as (but not limited ): Roundabout, Merging, Urban/Street navigation, Two way navigation (pass over the opposite direction lane), self parking, etc...

The objective is to compare performances of well known RL algorithms (from Open AI baselines) in these environments and use these results as a foundation for further studies of the potencial use of RL for AD at production level.

The research will also address the differences (advantages and disadvantages) between Continuous and Discrete Action Spaces. 

The Original version of these environments were created by **Edouard Leurent** and can be found in https://github.com/eleurent/highway-env

## Getting Started (Docker)
0) This code requires (Don't install anything yet):

   * Python 3
   * OpenAI Gym (https://github.com/openai/gym)
   * OpenAI Baselines (https://github.com/openai/baselines)

1) **Docker Installation**
   1) For a clean and encapsulated installation we will use Docker. Follow the instructions on https://docs.docker.com/install/linux/docker-ce/ubuntu/
      1) Uninstall old versions
Older versions of Docker were called docker, docker.io , or docker-engine. If these are installed, uninstall them:
            ```bash
            sudo apt-get remove docker docker-engine docker.io containerd runc
            ```
        2) Update the apt package index.
            ```bash
            sudo apt-get update
            ```
        3) Install the latest version of Docker CE and containerd, or go to docker page mentioned above to install a specific version:
            ```bash
            sudo apt-get install docker-ce docker-ce-cli containerd.io
            ```
        4) Verify that Docker CE is installed correctly by running the hello-world image.
            ```bash
            sudo docker run hello-world
            ```

1) Clone this repository and make sure you include the openai baselines submodules by typing the following command
```bash
git clone --recurse-submodules  https://gitlab.com/havalus-gemini/behavior/rl_baselines_ad.git
```
2) Navigate to the folder where you cloned this repo. 
```bash
   cd rl_baselines_ad
```
3) Run the following command to build the Docker image. This wil take some time, be patient!
```bash
docker build -t munirjojoverge/rl_baselines .
```
4) Run the following command to start the Docker. Make sure you can run it by makeint executable:   
```bash
chmod +x docker-start.sh
./docker-start.sh
```
5) If everything went ok, you should see something like this: 
![Inside docker](/img/DockerStart.png)

6) Run the following command to install openai gym and all the required libraries to run.
```bash
./docker-ini-script.sh
```

## Arguments and Config Files
**baselines_run.py** script uses **2** clear defined sections to setup all the main OpenAI Baselines arguments.
```python
    ###############################################################
    #        DEFINE YOUR "BASELINE" (AGENT) PARAMETERS HERE       
    ###############################################################
    train_env_id =  'merge-v0'
    play_env_id = ''
    alg = 'ppo2'
    network = 'mlp'
    num_timesteps = '1e4'
    load_file_name = '20190511-121635' # 'merge-v0'    
    #################################################################        
```

 **settings.py** is used to specify the folders where all the training logs and final/trained agent weights will be saved to. 

## Hyperparameter Tuning
Check OpenAI Baselines for more information on how to change their default hyperparameters. This will let you specify a set of hyperparameters to test different from the default ones.

## Benchmark Results
By default the training results will be placed on

**/run/models/"algorithm"/"network"**

### A2C
To reproduce my results, run ***baselines_run.py*** use the following setup:
```python
    ##########################################################
    #            DEFINE YOUR "BASELINES" PARAMETERS HERE 
    ##########################################################
    train_env_id =  'merge-v0'
    play_env_id = ''
    alg = 'a2c'
    network = 'mlp'
    num_timesteps = '1e4'
    load_file_name = '20190511-121635' # 'merge-v0' 
    ##########################################################
        
```
#### NOTE: 
1) If you run it for the first time, make sure to comment out the loading path argument (see below), since you don't have any initial weights to load.
2) If you want to use the "default" network (mlp) you can comment out the "--network" argument (see below)
3) logger_path will only work with some of the OpenAI baseline algorithms. If you chose one algortihm and it throws you an error regarding the "logger" just comment out the argument (see below)
4) For more information about what these arguments do and if there are more arguments that you can add to "tune" your agent, please refer to OpenAI baselines README.MD files for the algorithm/agent you are using.

```python
DEFAULT_ARGUMENTS = [
        '--env=' + env,
        '--alg=' + alg,
    #    '--network=' + network,
        '--num_timesteps=' + num_timesteps,    
    #    '--num_env=0',
        '--save_path=' + save_file,
    #    '--load_path=' + load_path,
    #    '--logger_path=' + logger_path,
        '--play'
    ]
```

### ACKTR
To reproduce my results, follow the same instructions as above. As it is obvious, in this case the algorithm argument is now
```python
alg = 'acktr'
```

### PPO2
To reproduce my results, follow the same instructions as above. As it is obvious, in this case the algorithm argument is now
```python
alg = 'ppo2'
```
### TRPO_MPI
To reproduce my results, follow the same instructions as above. As it is obvious, in this case the algorithm argument is now
```python
alg = 'trpo_mpi'
```

### ON-POLICY Hindsight Experience Replay (HER) over DDPG
To reproduce my results, follow the same instructions as above. As it is obvious, in this case the algorithm argument is now
```python
alg = 'her'
```
#### NOTE:
1) To know which algorithms you can use, simply take a look at the /open_ai_baselines/baselines folder.
2) Some algorithms will fail since are NOT suited for your problem. For example, DDPG was implemented for discrete actions spaces and will not take a "BOX" as an action space. Try the one you are interested in and find out why it will not run. Sometimes it will take just a few changes and other times, as metioned before, it might not even be meant for this type of problem. 

Note: "Save path" does not work on DDPG unless you add to the ddpg class the following: 

```python
from baselines.common import tf_util

def save(self, save_path):

​        tf_util.save_variables(save_path)
```

Comment the lines on DEFAULT_ARGUMENTS as shown below if you want to skip certain parameters

```python
DEFAULT_ARGUMENTS = [
​        '--env=' + env,
​        '--alg=' + alg,
​    #    '--network=' + network,
​        '--num_timesteps=' + num_timesteps,    
​    #    '--num_env=0',
​        '--save_path=' + save_file,
​    #    '--load_path=' + load_path,
​        '--logger_path=' + logger_path,
​        '--play'
​    ]
```

if you have a s3 Drive mounted uncomment the s3pathname with the full path to the mounted drive. The load model method would automatically prioritize this path.


## How to Run

On the command prompt run:

```bash
python baselines_run.py
```
OR 

./run_baseline.sh 70 (= number of workers/environments you want to parallely process. The parallelization uses MPI calls, which the baseline code supports)

