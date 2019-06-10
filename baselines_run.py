######################################################################
#          Deep Reinforcement Learning for Autonomous Driving
#                  Created/Modified on: February 5, 2019
#                      Author: Munir Jojo-Verge
#######################################################################

from settings import req_dirs, models_folder
import sys
import os
from os.path import dirname, abspath
import time
import pprint
pp = pprint.PrettyPrinter(indent=4)
####################
pathname = os.getcwd()
# print("current directory is : " + pathname)
foldername = os.path.basename(pathname)
# print("Directory name is : " + foldername)
####################

urban_AD_env_path = pathname + '/urban_env/envs'
# print(urban_AD_env_path)

open_ai_baselines_dir = pathname + '/open_ai_baselines'
# print(open_ai_baselines_dir)

sys.path.append(open_ai_baselines_dir)
sys.path.append(urban_AD_env_path)


import gym
import urban_env
import numpy as np
import glob
import warnings
import subprocess

import tensorflow as tf
from shutil import copyfile
from mpi4py import MPI
from urban_env.envs.abstract import AbstractEnv


from baselines.common.cmd_util import common_arg_parser, parse_unknown_args
from baselines.results_plotter import plot_results
from baselines.common.vec_env import VecEnv
from baselines.common import tf_util,mpi_util
from baselines import logger
import baselines.run as run



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")


###############################################################
#        DEFINE YOUR "BASELINE" (AGENT) PARAMETERS HERE
###############################################################
train_env_id = 'two-way-v0'
play_env_id = 'two-way-v0'
alg = 'ppo2'
network = 'mlp'
num_timesteps = '1e0'
#################################################################
first_MPI_call  = True
LOAD_PREV_MODEL = True

def create_dirs(req_dirs):
    for dirName in req_dirs:
        if not os.path.exists(dirName):
            os.mkdir(dirName)
            # print("Directory " , dirName ,  " Created ")
        # else:
            # print("Directory " , dirName ,  " already exists")



def is_master():
    return MPI.COMM_WORLD.Get_rank()==0


if is_master():
    InceptcurrentDT = time.strftime("%Y%m%d-%H%M%S")
else:
    InceptcurrentDT = None

InceptcurrentDT = MPI.COMM_WORLD.bcast(InceptcurrentDT, root=0)

def is_predict_only():
    return float(num_timesteps) == 1
gym.Env.metadata['_predict_only'] = is_predict_only()

def default_args(save_in_sub_folder=None):
    create_dirs(req_dirs)
    currentDT = time.strftime("%Y%m%d-%H%M%S")
    global first_MPI_call 
    ####################################################################
    # DEFINE YOUR SAVE FILE, LOAD FILE AND LOGGING FILE PARAMETERS HERE
    ####################################################################
    save_folder = pathname + '/' + models_folder + \
        '/' + train_env_id + '/' + alg + '/' + network

    if first_MPI_call :
        list_of_file = glob.glob(save_folder+'/*')
        if save_in_sub_folder is not None:
            save_folder += '/' + str(save_in_sub_folder)
        save_file = save_folder + '/' + str(currentDT)
        first_MPI_call_Trigger  = True
    else:
        if save_in_sub_folder is not None:
            save_folder += '/' + str(save_in_sub_folder)
        save_file = save_folder + '/' + str(currentDT)
        list_of_file = glob.glob(save_folder+'/*')
        first_MPI_call_Trigger = False

    # Specifiy log directories for open AI
    '''logger_path = save_folder + '/log/'
    tb_logger_path = save_folder + '/tb/'
    os.environ['OPENAI_LOGDIR'] = logger_path
    os.environ['OPENAI_LOG_FORMAT'] = 'stdout,tensorboard'''

    ###############################################################



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


    def copy_terminal_output_file():
        src = os.getcwd() + '/' + terminal_output_file_name
        dst = save_folder + '/' + terminal_output_file_name
        if not os.path.exists(save_folder):
           os.mkdir(save_folder)
        copyfile(src, dst)

    def create_save_folder(save_folder):
        try:
            os.makedirs(save_folder)
        except OSError:
            # print ("Creation of the save path %s failed. It might already exist" % save_folder)
            a = 1
        else:
            print("Successfully created the save path folder %s " % save_folder)

    def save_model(save_file=None):
        if save_file is not None:
            if not is_predict_only():
                if MPI is None or is_master():
                    create_save_folder(save_folder=save_folder)
                    DEFAULT_ARGUMENTS.append('--save_path=' + save_file)
                    print("Saving file", save_file)
                    copy_terminal_output_file()
                    # DEFAULT_ARGUMENTS.append('--tensorboard --logdir=' + tb_logger_path)
        return

    def load_model(load_file=None):
        if load_file is not None:
           if (not LOAD_PREV_MODEL) and first_MPI_call :
              return
           DEFAULT_ARGUMENTS.append('--load_path=' + load_file)
           print("Loading file", load_file)
        return

    terminal_output_file_name = 'output.txt'

    def is_empty_directory(directorypath):
        if not os.path.isdir(directorypath):
               return False
        if not os.listdir(directorypath):
               return True
        return False

    def filetonum(filename):
        try:
            return int(filename.split('/')[-1].replace('-',''))
        except:
            return -1

    def purge_names_not_matching_pattern(list_of_file_or_folders):
        if not list_of_file_or_folders:
            return None
        for fileorfoldername in list_of_file_or_folders:
            if '.' in fileorfoldername:
               list_of_file_or_folders.remove(fileorfoldername)
            elif is_empty_directory(directorypath=fileorfoldername): # remove empty directories
                 list_of_file_or_folders.remove(fileorfoldername)
        return list_of_file_or_folders


    def latest_model_file_from_list_of_files_and_folders(list_of_files):
        list_of_file_or_folders = purge_names_not_matching_pattern(list_of_file_or_folders =list_of_files )
        if not list_of_file_or_folders :
            return None
        latest_file_or_folder = max(list_of_file_or_folders, key=filetonum)
        if os.path.isdir(latest_file_or_folder) :
            list_of_files_and_folders_in_subdir = glob.glob(latest_file_or_folder+'/*')
            latest_model_file_in_subdir = \
                    latest_model_file_from_list_of_files_and_folders(list_of_files_and_folders_in_subdir)
            if latest_model_file_in_subdir is None:
                list_of_file_or_folders.remove(latest_file_or_folder)
                return latest_model_file_from_list_of_files_and_folders(list_of_file_or_folders)
            else:
                return latest_model_file_in_subdir
        return latest_file_or_folder # must be a file

    if list_of_file:  # is there anything in the save directory
       if save_in_sub_folder is  None:
          load_last_model = LOAD_PREV_MODEL
       else:
          load_last_model = LOAD_PREV_MODEL or not first_MPI_call 

       if load_last_model:
          latest_file = latest_model_file_from_list_of_files_and_folders(list_of_files=list_of_file)
          load_model(load_file=latest_file)
          save_model(save_file=save_file)
       else:
          save_model(save_file=save_file)
    else:
        print(" list_of_file empty in load path ", save_folder)
        save_model(save_file=save_file)

    # print(" DEFAULT_ARGUMENTS ", DEFAULT_ARGUMENTS)

    if first_MPI_call_Trigger:
        first_MPI_call = False

    return DEFAULT_ARGUMENTS


def play(env, policy):

    # env = gym.make(env_id)

    logger.configure()
    # logger.log("Running trained model")
    obs = env.reset()

    state = policy.initial_state if hasattr(policy, 'initial_state') else None
    dones = np.zeros((1,))

    def print_action_and_obs():
        print('episode_rew={}'.format(episode_rew), '  episode_len={}'.format(episode_len),\
                'episode_travel = ', episode_travel)
        print("obs space ")
        obs_format = pp.pformat(np.round(np.reshape(obs,(6, 5)),3))
        obs_format = obs_format.rstrip("\n")
        extra_obs = pp.pformat(info["extra_obs"])
        print(obs_format)
        print(extra_obs)

                
        print("Optimal action ",AbstractEnv.ACTIONS[actions[0]], "\n" )

    episode_rew = 0 
    episode_len = 0
    while True:
        if state is not None:
            actions, _, state, _ = policy.step(obs, S=state, M=dones)
        else:
            actions, _, _, _ = policy.step(obs)

        obs, rew, done, info = env.step(actions[0])
        episode_travel = env.vehicle.position[0]-env.ego_x0
        if episode_len%10 ==0 and is_predict_only():
            print_action_and_obs()  
       # print(env._max_episode_step)
        episode_rew += rew
        episode_len += 1
        env.render()
        done = done.any() if isinstance(done, np.ndarray) else done
  
        if done:
            print_action_and_obs()
            episode_rew = 0
            episode_len = 0
            env.close()
            break


if __name__ == "__main__":

    itr = 1
    sys_args = sys.argv

    policy = None
    play_env = None
    max_iteration = 1
    if not is_predict_only():
        while itr <= max_iteration:
            sess = tf_util.get_session()
            sess.close()
            tf.reset_default_graph()
            play_env = gym.make(play_env_id)
            print(" Batch iteration ", itr)
            print("(rank , size) = ",mpi_util.get_local_rank_size(MPI.COMM_WORLD))
            if len(sys_args) <= 1:
                save_in_sub_folder = None
                if max_iteration > 1:
                    save_in_sub_folder = InceptcurrentDT
                args = default_args(save_in_sub_folder=save_in_sub_folder)

            policy = run.main(args)
            MPI.COMM_WORLD.barrier()

            # print("policy training args ", args,"\n\n")
            itr += 1
            print()

            '''try:
                play(play_env, policy)
            except Exception as e:
                print("Could not play the prediction after training due to error  ", e)'''
            # from tensorboard import main as tb
            # tb.main()

    else:
        DFLT_ARGS = default_args()
        loaded_file_correctly = ('load_path' in stringarg for stringarg in DFLT_ARGS)
        policy = run.main(DFLT_ARGS)
        play_env = gym.make(play_env_id)
        # Just try to Play
        while loaded_file_correctly:
            play(play_env, policy)


    

