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
import subprocess


from baselines.common.vec_env import  VecEnv
from baselines.common import tf_util
import tensorflow as tf

####################
pathname = os.getcwd()
#print("current directory is : " + pathname)
foldername = os.path.basename(pathname)
#print("Directory name is : " + foldername)
####################

open_ai_baselines_dir = pathname + '/open_ai_baselines'
#print(open_ai_baselines_dir)

urban_AD_env_path = pathname + '/urban_env/envs'
#print(urban_AD_env_path)

sys.path.append(open_ai_baselines_dir)
sys.path.append(urban_AD_env_path)

import baselines.run as run
from baselines import logger

import urban_env

from settings import req_dirs, models_folder
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings("ignore") 





###############################################################
#        DEFINE YOUR "BASELINE" (AGENT) PARAMETERS HERE 
###############################################################
train_env_id =  'two-way-v0' 
play_env_id = 'two-way-v0'
alg = 'ppo2'
network = 'mlp'
num_timesteps = '1e0'
#################################################################

def create_dirs(req_dirs):
    for dirName in req_dirs:
        if not os.path.exists(dirName):
            os.mkdir(dirName)
            #print("Directory " , dirName ,  " Created ")
        #else:    
            #print("Directory " , dirName ,  " already exists")

InceptcurrentDT = time.strftime("%Y%m%d-%H%M%S")


def default_args(save_in_sub_folder=None):    
    create_dirs(req_dirs)

    currentDT = time.strftime("%Y%m%d-%H%M%S")
    
    ####################################################################
    # DEFINE YOUR SAVE FILE, LOAD FILE AND LOGGING FILE PARAMETERS HERE 
    #################################################################### 
    save_folder = pathname + '/' + models_folder + '/' + train_env_id +'/'+ alg + '/' + network 
    if save_in_sub_folder is not None:
       save_folder +=  '/' + str(save_in_sub_folder)
    save_file = save_folder + '/' + str(currentDT)
    list_of_file = glob.glob(save_folder+'/*')

    # Specifiy log directories for open AI 
    ''' logger_path = save_folder + '_log/'
    tb_logger_path = logger_path + '/tb'
    os.environ['OPENAI_LOGDIR'] = logger_path
    os.environ['OPENAI_LOG_FORMAT'] = 'stdout,tensorboard'''
  
    ###############################################################
        
    try:  
        os.mkdir(save_folder)
    except OSError:  
        #print ("Creation of the save path %s failed. It might already exist" % save_folder)
        a=1
    else:  
        print ("Successfully created the save path folder %s " % save_folder)

    DEFAULT_ARGUMENTS = [
        '--env=' + train_env_id,
        '--alg=' + alg,
        '--network=' + network,
        '--num_timesteps=' + num_timesteps,    
    #    '--num_env=0',
    #    '--save_path=' + save_file,        
    #     '--tensorboard --logdir=' + tb_logger_path,
    #    '--play'
    #    '--num_env=8' 
    ]

    def save_model(save_file = None):
        if save_file is not None:
            DEFAULT_ARGUMENTS.append('--save_path=' + save_file)
        return 

    def load_model(load_file = None):
        if load_file is not None:
            DEFAULT_ARGUMENTS.append('--load_path=' + load_file)
        return 

    loadlatestfileforplay = False
    if (float(num_timesteps) == 1):
       loadlatestfileforplay = True

    if list_of_file: # is there anything in the save directory
        latest_file_or_folder = max( list_of_file, key=os.path.getctime) 
#        print(" latest_file_or_folder " ,latest_file_or_folder)
        if os.path.isdir(latest_file_or_folder): # search for the latest file (to load) from the latest folder\
            latest_folder = latest_file_or_folder
            list_of_files = glob.glob(latest_folder+'/*')
            if list_of_files and (save_in_sub_folder is not None) :
               if save_in_sub_folder in latest_folder:
                  latest_file = max( list_of_files, key=os.path.getctime)
                  load_model(load_file=latest_file)
                  print(" load_path " ,latest_file)

            save_model(save_file=save_file)
#            print(" save_path " ,save_file)
        else: # got the latest file (to load)
            latest_file = latest_file_or_folder
            if loadlatestfileforplay:
                load_model(load_file=latest_file) 
            elif save_in_sub_folder is not None :
                load_model(load_file=latest_file) 
                save_model(save_file=save_file)
            else:
                save_model(save_file=save_file)
    else : 
        print(" list_of_file empty in load path ", save_folder)
        if not loadlatestfileforplay:
                save_model(save_file=save_file)

    #print(" DEFAULT_ARGUMENTS ", DEFAULT_ARGUMENTS)

    return DEFAULT_ARGUMENTS


def play(env, policy):
    
    # env = gym.make(env_id)
    
    logger.configure()
    #logger.log("Running trained model")
    obs = env.reset()

    state = policy.initial_state if hasattr(policy, 'initial_state') else None
    dones = np.zeros((1,))

    episode_rew = 0
    episode_len = 0
    while True:
        if state is not None:
            actions, _, state, _ = policy.step(obs,S=state, M=dones)
        else:
            actions, _, _, _ = policy.step(obs)

        obs, rew, done, _ = env.step(actions[0])
        episode_rew += rew
        episode_len += 1
        env.render()
        done = done.any() if isinstance(done, np.ndarray) else done
        if episode_len%10 ==0:
            print('episode_rew={}'.format(episode_rew), '  episode_len={}'.format(episode_len))
        if done:
            print('episode_rew={}'.format(episode_rew))
            print('episode_len={}'.format(episode_len))
            episode_rew = 0
            episode_len = 0
            env.close()
            break


if __name__ == "__main__":
    


    itr = 1
    sys_args = sys.argv

    
    max_iteration = 1
    while itr<=max_iteration and float(num_timesteps)>1:
        play_env = gym.make(play_env_id)
        print(" Batch iteration ", itr)
        if len(sys_args) <= 1:
            save_in_sub_folder = None
            if max_iteration > 1:
               save_in_sub_folder = InceptcurrentDT
            args = default_args(save_in_sub_folder)

        policy = run.main(args)

        #print("policy training args ", args,"\n\n")
        itr += 1
        #play(play_env,policy)
        #from tensorboard import main as tb
        #tb.main()
        sess = tf_util.get_session()
        sess.close()
        tf.reset_default_graph()

    '''try:
        subprocess.call(["rsync", "-avu", "--delete","../", "localhost:~/Documents/aws_sync"])
    except:
        print("Rsync didn't work")'''

    try:
        # Just try Play
        play_env = gym.make(play_env_id)
        policy = run.main(default_args())
        while True:
            play(play_env,policy)
            '''sess = tf_util.get_session()
            sess.close()
            tf.reset_default_graph()'''
    except Exception as e:
        print("Could not play the prediction due to error ",e)
