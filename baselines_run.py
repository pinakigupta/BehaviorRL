######################################################################
#          Deep Reinforcement Learning for Autonomous Driving
#                  Created/Modified on: February 5, 2019
#                      Author: Munir Jojo-Verge
#######################################################################

import sys, os
from os.path import dirname, abspath
import time
import gym
import numpy as np
import glob
####################
pathname = os.getcwd()
print("current directory is : " + pathname)
foldername = os.path.basename(pathname)
print("Directory name is : " + foldername)
####################

open_ai_baselines_dir = pathname + '/open_ai_baselines'
print(open_ai_baselines_dir)

urban_AD_env_path = pathname + '/urban_env/envs'
print(urban_AD_env_path)

sys.path.append(open_ai_baselines_dir)
sys.path.append(urban_AD_env_path)

import baselines.run as run
from baselines import logger

import urban_env

from settings import req_dirs, models_folder

###############################################################
#        DEFINE YOUR "BASELINE" (AGENT) PARAMETERS HERE 
###############################################################
train_env_id =  'two-way-v0' 
play_env_id = 'two-way-v0'
alg = 'ppo2'
network = 'mlp'
num_timesteps = '1e1'
#################################################################

def create_dirs(req_dirs):
    for dirName in req_dirs:
        if not os.path.exists(dirName):
            os.mkdir(dirName)
            print("Directory " , dirName ,  " Created ")
        else:    
            print("Directory " , dirName ,  " already exists")


def default_args():    
    create_dirs(req_dirs)

    currentDT = time.strftime("%Y%m%d-%H%M%S")
    ####################################################################
    # DEFINE YOUR SAVE FILE, LOAD FILE AND LOGGING FILE PARAMETERS HERE 
    ####################################################################        
    save_folder = models_folder + '/' + train_env_id +'/'+ alg + '/' + network 
    save_file = save_folder + '/' + str(currentDT)
    logger_path = save_file + '_log'
    list_of_file = glob.glob('save_folder/*')
    print("list_of_file",list_of_file)
  
    ###############################################################
        
    try:  
        os.mkdir(save_folder)
    except OSError:  
        print ("Creation of the save path %s failed. It might already exist" % save_folder)
    else:  
        print ("Successfully created the save path folder %s " % save_folder)

    DEFAULT_ARGUMENTS = [
        '--env=' + train_env_id,
        '--alg=' + alg,
    #    '--network=' + network,
        '--num_timesteps=' + num_timesteps,    
    #    '--num_env=0',
        '--save_path=' + save_file,        
    #    '--logger_path=' + logger_path,
        '--play'
    ]

    if list_of_file:
        latest_file = max( list_of_file, key=os.path.getctime)
        load_path = save_folder +'/'+ latest_file #her_default_20190212-141935' # Good with just Ego  
        DEFAULT_ARGUMENTS.append('--load_path=' + load_path) 

    return DEFAULT_ARGUMENTS


def play(env_id, policy):
    
    env = gym.make(env_id)
    
    logger.configure()
    logger.log("Running trained model")
    obs = env.reset()

    state = policy.initial_state if hasattr(policy, 'initial_state') else None
    dones = np.zeros((1,))

    episode_rew = 0
    while True:
        if state is not None:
            actions, _, state, _ = policy.step(obs,S=state, M=dones)
        else:
            actions, _, _, _ = policy.step(obs)

        obs, rew, done, _ = env.step(actions)
        episode_rew += rew
        env.render()
        done = done.any() if isinstance(done, np.ndarray) else done
        if done:
            print('episode_rew={}'.format(episode_rew))
            episode_rew = 0
            obs = env.reset()

if __name__ == "__main__":
    args = sys.argv
    if len(args) <= 1:
        args = default_args()
    policy = run.main(args)
    #play(play_env_id,policy)          

