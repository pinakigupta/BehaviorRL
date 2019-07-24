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
import warnings


####################
pathname = os.getcwd()
#print("current directory is : " + pathname)
####################

open_ai_baselines_dir = pathname + '/open_ai_baselines'
print(open_ai_baselines_dir)

urban_AD_env_path = pathname + '/urban_env/envs'
print(urban_AD_env_path)

sys.path.append(open_ai_baselines_dir)
sys.path.append(urban_AD_env_path)

import baselines.run as run
from baselines import logger
from baselines.common.vec_env import  VecEnv
from baselines.common import tf_util
import tensorflow as tf


import urban_env

from settings import req_dirs, models_folder
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings("ignore") 

###############################################################
#        DEFINE YOUR "BASELINE" (AGENT) PARAMETERS HERE 
###############################################################
#train_env_id =  'merge-v0'
#train_env_id =  'roundabout-v0'
#train_env_id =  'two-way-v0'
#train_env_id =  'parking_2outs-v0'
train_env_id =  'multilane-v0'

play_env_id = ''
alg = 'ppo2'
network = 'mlp'
num_timesteps = '3e4'
#load_file_name = '20190511-121635' # 'merge-v0'
#load_file_name = '20190510-100005' # 'roundabout-v0'
#load_file_name = '20190511-180238' # 'two-way-v0'
#load_file_name = '20190506-082121' # 'parking_2outs-v0'
load_file_name = 'empty' # 'multilane-v0'
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
    save_folder = models_folder + '/' + train_env_id +'/'+ alg + '/' + network 
    save_file = save_folder + '/' + str(currentDT)
    logger_path = save_file + '_log'
    list_of_file = glob.glob(save_folder+'/*')
        
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

    if load_file_name == 'empty':
        print(" list_of_file empty in load path ", save_folder)
        print('We will train from scratch')       
    elif load_file_name != '':
        load_path = load_file_name
        print("load_path",load_path)
        DEFAULT_ARGUMENTS.append('--load_path=' + load_path) 
    else:
        latest_file = max( list_of_file, key=os.path.getctime)
        load_path = latest_file 
        print("load_path",load_path)
        DEFAULT_ARGUMENTS.append('--load_path=' + load_path)       

    return DEFAULT_ARGUMENTS


def play(env, policy):
    
    # env = gym.make(env_id)
    
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

        obs, rew, done, _ = env.step(actions[0])
        episode_rew += rew
        env.render()
        done = done.any() if isinstance(done, np.ndarray) else done
        if done:
            print('episode_rew={}'.format(episode_rew))
            episode_rew = 0
            env.close()
            break

if __name__ == "__main__":    
    args = sys.argv
    itr = 1
    while itr<20:
        if len(args) <= 1:
            args = default_args()
        
        policy = run.main(args)
        print(" Batch iteration ", itr)
        itr += 1
        if play_env_id != 'None':
            if play_env_id == '':
                play_env_id = train_env_id
            play_env = gym.make(play_env_id)
            play(play_env,policy)
        sess = tf_util.get_session()
        sess.close()