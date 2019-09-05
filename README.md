# Reinforcement Learning based Autonomous Driving (AD)


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
   * RLLIB (https://ray.readthedocs.io/en/latest/rllib.html)

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
git clone  https://github.com/pinakigupta/BehaviorRL.git
```
2) Navigate to the folder where you cloned this repo. 
```bash
   cd BehaviorRL
```
3) Run the following command to build the Docker image. This wil take some time, be patient!
```bash
docker build -t pinakigupta/rl_baselines dockerfiles/
```
Alternatively you can pull the latest docker image pushed to my public repo on docker hub. 
```bash
docker pull pinakigupta/rl_baselines 
```
4) Run the following command to start the Docker. Make sure you can run it by make it executable:   
```bash
chmod +x dockerfiles/docker-start.sh
dockerfiles/docker-start.sh
```
This will launch a docker container named "ray_docker_local". You can change the container name in the docker-start.sh script.
Additional disclaimer!!! This step will also try to copy your aws credentials from .aws and .ssh folders presuming they are located in the ~/ directory.
This step is important to be able to launch a cluster on the aws cloud from your docker container. 
If you have ray (or any similar 3rd party cloud compute package) installed on your local machine you may not need to launch compute clusters from the docker container.
But launching it from the docker container is easy, no local installation required and gurantees version compatibility. 

5) If everything went ok, you should see something like this: 
![Inside docker](/img/DockerStart.png)

6) Run the following command to install openai gym and all the required libraries to run. This step additionally moves the aws credentials to docker ~/ and assigns root
uid/gid to the credential files.
```bash
cd rl_baselines_ad
dockerfiles/docker-ini-script.sh
```
# RLLIB/RAY
**raylibs.py** script contains two main methods ray_train(...) and ray_play() along with helper methods. 
For details about using ray visit https://ray.readthedocs.io/en/latest/rllib.html

## How to Run on local machine

On the command prompt run:

```bash
python baselines_run.py
python baselines_run.py  predict=True (To run only prediction)
python baselines_run.py  predict=False (To run only training)
```
OR

```bash
./run_baseline.sh
```

## How to Run on aws cluster

Run the following command to  
    &nbsp;&nbsp;&nbsp;&nbsp;a) launch cluster in aws (cluster parameters can be changed in Ray-Cluster.yaml file). In aws cluster HEAD node a docker container "ray docker" will be launched.  
    &nbsp;&nbsp;&nbsp;&nbsp;b)launches the main script within the ray_docker container (which uses the cluster WORKER nodes, and auto scales/descales according to compute need). This step 
    runs the ./run_baseline.sh script on the HEAD node docker  
    &nbsp;&nbsp;&nbsp;&nbsp;c)copies result files to the local machine (using script ray_sync.sh).  
    &nbsp;&nbsp;&nbsp;&nbsp;d)deletes the cluster in aws(this process is sometimes buggy. If cluster is not down type manually "ray down Ray-Cluster.yaml" to shutdown the cluster)  
    
```bash
bash ray_cluster_launch.sh
```
## Training configurations

Training can be done using ray Experiment API or ray tune API. Both examples have been provided in the ray_train(...) method with the ray tune API be the active choice. This 
is because tune builds on Experiment and allows hyper parameter tuning.  
    &nbsp;&nbsp;algo = "PPO" # RL Algorithm of choice  
    &nbsp;&nbsp;LOAD_MODEL_FOLDER = "20190828-201729" # Location of previous model (if needed) for training  
    &nbsp;&nbsp;RESTORE_COND = "NONE" # RESTORE: Use a previous model to start new training  
    &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;# RESTORE_AND_RESUME: Use a previous model to finish previous unfinished training  
    &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;# NONE: Start fresh  
    Also the following config options are self explanatory  
```bash
                "config": {
                    "num_gpus_per_worker": 0,
                    #"num_cpus_per_worker": 1,
                    # "gpus": 0,
                    "gamma": 0.85,
                    "num_workers": delegated_cpus,
                    "num_envs_per_worker": 2,
                    "env": train_env_id,
                    "remote_worker_envs": False,
                    "model": {
                                #    "use_lstm": True,
                                    "fcnet_hiddens": [256, 256, 256],
                            },
                     }   
```   
## Prediction configurations
Prediction is using the ray_play() method defined  in raylibs.py. As it stands most configurations are directly extracted from the save model. So the only config parameter provided is the 
model folder lcoation.  
LOAD_MODEL_FOLDER = "20190903-200633" # Location of previous model for prediction 

## Hyperparameter Tuning
Ray tune API provides configuration choices for hyper parameter tuning. For details please visit https://ray.readthedocs.io/en/latest/tune.html  
For using the current setup following lines can be  unhighlghted to get a population based training schedular. 

```bash
            # scheduler=pbt,
            # trial_executor=RayTrialExecutor(),
            # resources_per_trial={"cpu": delegated_cpus, "gpu": 0},

                    # Following are some params to be tuned
            
                    # These params are tuned from a fixed starting value.
                    # "lambda": 0.95,
                    # "clip_param": 0.2,
                    # "lr": 1e-4,
                    
                    # These params start off randomly drawn from a set.
                    # "num_sgd_iter": sample_from(lambda spec: random.choice([10, 20, 30])),
                    # "sgd_minibatch_size": sample_from(lambda spec: random.choice([128, 512, 2048])),
                    # "train_batch_size": sample_from(lambda spec: random.choice([10000, 20000, 40000])),
```   


## Benchmark Results
By default the training results will be placed on

**./ray_results/"%Y%m%d-%H%M%S"/...**

# OPEN AI BASELINES
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

**./run/models/"algorithm"/"network"**

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
python baselines_run.py  predict=True (To run only prediction)
python baselines_run.py  predict=False (To run only training)
```
OR 

```bash
./run_baseline.sh 70 (= number of workers/environments you want to parallely process. The parallelization uses MPI calls, which the baseline code supports)
```bash
