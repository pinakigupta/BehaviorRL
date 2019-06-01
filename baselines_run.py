######################################################################
#          Deep Reinforcement Learning for Autonomous Driving
#                  Created/Modified on: February 5, 2019
#                      Author: Munir Jojo-Verge
#######################################################################

from settings import req_dirs, models_folder
import urban_env
from baselines import logger
import baselines.run as run
import sys
import os
from os.path import dirname, abspath
import time
import gym
import numpy as np
import glob
import warnings
import subprocess
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args
from baselines.results_plotter import plot_results
from baselines.common.vec_env import VecEnv
from baselines.common import tf_util
import tensorflow as tf
from shutil import copyfile
from mpi4py import MPI

####################
pathname = os.getcwd()
# print("current directory is : " + pathname)
foldername = os.path.basename(pathname)
# print("Directory name is : " + foldername)
####################

open_ai_baselines_dir = pathname + '/open_ai_baselines'
# print(open_ai_baselines_dir)

urban_AD_env_path = pathname + '/urban_env/envs'
# print(urban_AD_env_path)

sys.path.append(open_ai_baselines_dir)
sys.path.append(urban_AD_env_path)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")


###############################################################
#        DEFINE YOUR "BASELINE" (AGENT) PARAMETERS HERE
###############################################################
train_env_id = 'two-way-v0'
play_env_id = 'two-way-v0'
alg = 'ppo2'
network = 'mlp'
num_timesteps = '0.6e5'
#################################################################
first_call = True


def create_dirs(req_dirs):
    for dirName in req_dirs:
        if not os.path.exists(dirName):
            os.mkdir(dirName)
            # print("Directory " , dirName ,  " Created ")
        # else:
            # print("Directory " , dirName ,  " already exists")



def is_master():
    return MPI.COMM_WORLD.Get_rank()==0


InceptcurrentDT = time.strftime("%Y%m%d-%H%M%S")

def is_predict_only():
    return float(num_timesteps) == 1

LOAD_PREV_MODEL = False
def default_args(save_in_sub_folder=None):
    create_dirs(req_dirs)
    currentDT = time.strftime("%Y%m%d-%H%M%S")
    global first_call
    ####################################################################
    # DEFINE YOUR SAVE FILE, LOAD FILE AND LOGGING FILE PARAMETERS HERE
    ####################################################################
    save_folder = pathname + '/' + models_folder + \
        '/' + train_env_id + '/' + alg + '/' + network

    if first_call:
        list_of_file = glob.glob(save_folder+'/*')
        if save_in_sub_folder is not None:
            save_folder += '/' + str(save_in_sub_folder)
        save_file = save_folder + '/' + str(currentDT)
        first_call = False
    else:
        if save_in_sub_folder is not None:
            save_folder += '/' + str(save_in_sub_folder)
        save_file = save_folder + '/' + str(currentDT)
        list_of_file = glob.glob(save_folder+'/*')

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
        copyfile(src, dst)

    def create_save_folder(save_folder):
        try:
            os.mkdir(save_folder)
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
           if not LOAD_PREV_MODEL and first_call:
              return
           DEFAULT_ARGUMENTS.append('--load_path=' + load_file)
           print("Loading file", load_file)
        return

    terminal_output_file_name = 'output.txt'

    def get_latest_file_or_folder(list_of_files):
        for fileorfoldername in list_of_files:
            if '.' in fileorfoldername:
               list_of_files.remove(fileorfoldername)
            elif os.path.isdir(fileorfoldername): # remove empty directories
                if not os.listdir(fileorfoldername):
                    list_of_files.remove(fileorfoldername)

        if not list_of_files:
            return None
        if list_of_files is None:
            return None
        def keyfn(filename):
            return int(filename.split('/')[-1].replace('-',''))
        return max(list_of_files, key=keyfn)

    def latest_model_file_from_list_of_files_and_folders(list_of_files):
        while list_of_files:
            latest_file_or_folder = get_latest_file_or_folder(list_of_files = list_of_files)
            if latest_file_or_folder is None:
                return None
            if os.path.isfile(latest_file_or_folder) :
                #print(" load_path ", latest_file)
                return latest_file
            list_of_files.remove(latest_file_or_folder)  # directory
        return None

    if list_of_file:  # is there anything in the save directory
        latest_file_or_folder = get_latest_file_or_folder(list_of_files = list_of_file)
#        print(" latest_file_or_folder " ,latest_file_or_folder)
        if latest_file_or_folder is None:
           print("latest_file_or_folder is None in list_of_file",list_of_file)
        elif os.path.isdir(latest_file_or_folder):  # search for the latest file (to load) from the latest folder\
            latest_folder = latest_file_or_folder
            list_of_files = glob.glob(latest_folder+'/*')
            if list_of_files and save_in_sub_folder is not None :
                if save_in_sub_folder in latest_folder:
                    latest_file = latest_model_file_from_list_of_files_and_folders(list_of_files)
                    load_model(load_file=latest_file)
            elif list_of_files and is_predict_only():
                latest_file = latest_model_file_from_list_of_files_and_folders(list_of_files)
                load_model(load_file=latest_file)

            save_model(save_file=save_file)
        else:  # got the latest file (to load)
            latest_file = latest_file_or_folder
            if save_in_sub_folder is not None:
                load_model(load_file=latest_file)
                save_model(save_file=save_file)
            else:
                save_model(save_file=save_file)
    else:
        print(" list_of_file empty in load path ", save_folder)
        save_model(save_file=save_file)

    # print(" DEFAULT_ARGUMENTS ", DEFAULT_ARGUMENTS)

    return DEFAULT_ARGUMENTS


def play(env, policy):

    # env = gym.make(env_id)

    logger.configure()
    # logger.log("Running trained model")
    obs = env.reset()

    state = policy.initial_state if hasattr(policy, 'initial_state') else None
    dones = np.zeros((1,))

    episode_rew = 0
    episode_len = 0
    while True:
        if state is not None:
            actions, _, state, _ = policy.step(obs, S=state, M=dones)
        else:
            actions, _, _, _ = policy.step(obs)

        obs, rew, done, _ = env.step(actions[0])
        episode_rew += rew
        episode_len += 1
        env.render()
        done = done.any() if isinstance(done, np.ndarray) else done
        # if episode_len%100 ==0:
        # print('episode_rew={}'.format(episode_rew), '  episode_len={}'.format(episode_len))
        if done:
            print('episode_rew={}'.format(episode_rew), '  episode_len={}'.format(episode_len),
                  'episode travel = ', env.vehicle.position[0]-env.ego_x0)
            episode_rew = 0
            episode_len = 0
            env.close()
            break


if __name__ == "__main__":

    itr = 1
    sys_args = sys.argv

    policy = None
    play_env = None
    max_iteration = 10
    if not is_predict_only():
        while itr <= max_iteration:
            sess = tf_util.get_session()
            sess.close()
            tf.reset_default_graph()
            play_env = gym.make(play_env_id)
            print(" Batch iteration ", itr)
            if len(sys_args) <= 1:
                save_in_sub_folder = None
                if max_iteration > 1:
                    save_in_sub_folder = InceptcurrentDT
                args = default_args(save_in_sub_folder=save_in_sub_folder)

            policy = run.main(args)

            # print("policy training args ", args,"\n\n")
            itr += 1

            '''try:
                play(play_env, policy)
            except Exception as e:
                print("Could not play the prediction after training due to error  ", e)'''
            # from tensorboard import main as tb
            # tb.main()

    else:
        policy = run.main(default_args())
        play_env = gym.make(play_env_id)

    '''try:
        subprocess.call(["rsync", "-avu", "--delete","../", "localhost:~/Documents/aws_sync"])
    except:
        print("Rsync didn't work")'''

    try:
        # Just try to Play
        while True:
            play(play_env, policy)
            '''sess = tf_util.get_session()
            sess.close()
            tf.reset_default_graph()'''
    except Exception as e:
        print("Could not play the prediction due to error ", e)
