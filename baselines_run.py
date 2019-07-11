######################################################################
#          Deep Reinforcement Learning for Autonomous Driving
#                  Created/Modified on: February 5, 2019
#                      Author: Munir Jojo-Verge
#######################################################################
from settings import req_dirs, models_folder
from urban_env.envs.two_way_env import TwoWayEnv
from urban_env.envs.abstract import AbstractEnv
from mpi4py import MPI
from shutil import copyfile
import subprocess
import warnings
import glob
import numpy as np
import urban_env
import gym
import tensorflow as tf

from settings import req_dirs, models_folder
import sys
import os
from os.path import dirname, abspath
import time
import pprint
import ray
from ray.tune import run_experiments, register_env
#from ray.rllib.agents import a3c
pp = pprint.PrettyPrinter(indent=4)
####################
pathname = os.getcwd()
homepath = os.path.expanduser("~")
#s3pathname = homepath+'/s3-drive/groups/Behavior/Pinaki'



urban_AD_env_path = pathname + '/urban_env/envs'
# print(urban_AD_env_path)

open_ai_baselines_dir = pathname + '/open_ai_baselines'
# print(open_ai_baselines_dir)

sys.path.append(open_ai_baselines_dir)
sys.path.append(urban_AD_env_path)


####################

from baselines.common import tf_util, mpi_util
from baselines.common.vec_env import VecEnv
from baselines import logger
import baselines.run as run

####################

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")


###############################################################
#        DEFINE YOUR "BASELINE" (AGENT) PARAMETERS HERE
###############################################################
'''
train_env_id = 'parking_2outs-v0'
play_env_id = 'parking_2outs-v0' 
alg = 'her'
network = 'mlp'
num_timesteps = '1e0'

'''
train_env_id = 'two-way-v0'
play_env_id = 'two-way-v0'
alg = 'ppo2'
network = 'mlp'
num_timesteps = '10000'

redis_add = ray.services.get_node_ip_address() + ":6379"
try:
    ray.init(redis_add)
except:
    ray.shutdown()
    ray.init()
register_env(train_env_id, lambda _: TwoWayEnv)
#################################################################
first_default_args_call = True
LOAD_PREV_MODEL = True


def create_dirs(req_dirs):
    for dirName in req_dirs:
        if not os.path.exists(dirName):
            os.mkdir(dirName)
            # print("Directory " , dirName ,  " Created ")
        # else:
            # print("Directory " , dirName ,  " already exists")


def is_master():
    return MPI.COMM_WORLD.Get_rank() == 0


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
    global first_default_args_call
    ####################################################################
    # DEFINE YOUR SAVE FILE, LOAD FILE AND LOGGING FILE PARAMETERS HERE
    ####################################################################
    modelpath = pathname
    try:
        if os.path.exists(s3pathname):
            modelpath = s3pathname
    except:
        print("s3 pathname doesn't exist")

    save_folder = modelpath + '/' + models_folder + \
        '/' + train_env_id + '/' + alg + '/' + network
    load_folder = modelpath + '/' + models_folder + \
        '/' + train_env_id + '/' + alg + '/' + network

    if first_default_args_call:
        list_of_file = glob.glob(load_folder+'/*')
        if save_in_sub_folder is not None:
            save_folder += '/' + str(save_in_sub_folder)
        save_file = save_folder + '/' + str(currentDT)
        first_default_args_call_Trigger = True
    else:
        if save_in_sub_folder is not None:
            save_folder += '/' + str(save_in_sub_folder)
        save_file = save_folder + '/' + str(currentDT)
        list_of_file = glob.glob(save_folder+'/*')
        first_default_args_call_Trigger = False

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

    DEFAULT_ARGUMENTS_DICT = {
        'env': train_env_id,
        'alg': alg,
        'network': network,
        'num_timesteps': num_timesteps
    }

    def copy_terminal_output_file():
        src = os.getcwd() + '/' + terminal_output_file_name
        dst = save_folder + '/' + terminal_output_file_name
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        if os.path.exists(src):
            copyfile(src, dst)
        else:
            print("out put file ",terminal_output_file_name,"doesn't exist")

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
                    DEFAULT_ARGUMENTS_DICT['save_path'] = save_file
                    print("Saving file", save_file)
                    copy_terminal_output_file()
                    # DEFAULT_ARGUMENTS.append('--tensorboard --logdir=' + tb_logger_path)
        return

    def load_model(load_file=None):
        if load_file is not None:
            if (not LOAD_PREV_MODEL) and first_default_args_call:
                return
            DEFAULT_ARGUMENTS.append('--load_path=' + load_file)
            DEFAULT_ARGUMENTS_DICT['load_path'] = load_file
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
            return int(filename.split('/')[-1].replace('-', ''))
        except:
            return -1

    def purge_names_not_matching_pattern(list_of_file_or_folders):
        if not list_of_file_or_folders:
            return None
        for fileorfoldername in list_of_file_or_folders:
            if '.' in fileorfoldername:
                list_of_file_or_folders.remove(fileorfoldername)
            # remove empty directories
            elif is_empty_directory(directorypath=fileorfoldername):
                list_of_file_or_folders.remove(fileorfoldername)
        return list_of_file_or_folders

    def latest_model_file_from_list_of_files_and_folders(list_of_files):
        list_of_file_or_folders = purge_names_not_matching_pattern(
            list_of_file_or_folders=list_of_files)
        if not list_of_file_or_folders:
            return None
        latest_file_or_folder = max(list_of_file_or_folders, key=filetonum)
        if os.path.isdir(latest_file_or_folder):
            list_of_files_and_folders_in_subdir = glob.glob(
                latest_file_or_folder+'/*')
            latest_model_file_in_subdir = \
                latest_model_file_from_list_of_files_and_folders(
                    list_of_files_and_folders_in_subdir)
            if latest_model_file_in_subdir is None:
                list_of_file_or_folders.remove(latest_file_or_folder)
                return latest_model_file_from_list_of_files_and_folders(list_of_file_or_folders)
            else:
                return latest_model_file_in_subdir
        return latest_file_or_folder  # must be a file

    if list_of_file:  # is there anything in the save directory
        if save_in_sub_folder is None:
            load_last_model = LOAD_PREV_MODEL
        else:
            load_last_model = LOAD_PREV_MODEL or not first_default_args_call

        if load_last_model:
            latest_file = latest_model_file_from_list_of_files_and_folders(
                list_of_files=list_of_file)
            load_model(load_file=latest_file)
            save_model(save_file=save_file)
        else:
            save_model(save_file=save_file)
    else:
        print(" list_of_file empty in load path ", load_folder)
        save_model(save_file=save_file)

    # print(" DEFAULT_ARGUMENTS ", DEFAULT_ARGUMENTS)

    if first_default_args_call_Trigger:
        first_default_args_call = False

    return DEFAULT_ARGUMENTS, DEFAULT_ARGUMENTS_DICT


def play(env, policy):

    # env = gym.make(env_id)

    logger.configure()
    # logger.log("Running trained model")
    obs = env.reset()

    state = policy.initial_state if hasattr(policy, 'initial_state') else None
    dones = np.zeros((1,))

    def print_action_and_obs():
        print('episode_rew={}'.format(episode_rew), '  episode_len={}'.format(episode_len),
              'episode_travel = ', episode_travel)
        env.print_obs_space()
        if "extra_obs" in info:
            extra_obs = pp.pformat(info["extra_obs"])
            print(extra_obs)

        #print("Optimal action ",AbstractEnv.ACTIONS[actions[0]], "\n")

    episode_rew = 0
    episode_len = 0
    ego_x0 = None
    while True:
        if state is not None:
            actions, _, state, _ = policy.step(obs, S=state, M=dones)
        else:
            actions, _, _, _ = policy.step(obs)

        if len(actions) == 1:
            actions = actions[0]
        obs, rew, done, info = env.step(actions)
        if ego_x0 is None:
            ego_x0 = env.vehicle.position[0]
        episode_travel = env.vehicle.position[0]-ego_x0
        if (episode_len % 10 == 0) and is_predict_only():
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


def ray_train(save_in_sub_folder=None):
    run_experiments({
                        "pygame-ray": {
                                        "run": "PPO",
                                        "env": TwoWayEnv,
                                        "stop": {"training_iteration": int(num_timesteps)},
                                        "checkpoint_at_end": True,
                                        "checkpoint_freq": 100,
                                        "config": {
                                                    # "env_config": env_config,
                                                    "num_gpus_per_worker": 0.2,
                                                    "num_cpus_per_worker": 2,
                                                    "gamma": 0.85,
                                                    "num_workers": 5,
                                                  },
                                        "local_dir": save_in_sub_folder,
                                      },
                     },
        resume=False,
        reuse_actors=False,
                   )


if __name__ == "__main__":

    mega_batch_itr = 1
    sys_args = sys.argv

    TRAIN_WITH_RAY = True
    policy = None
    play_env = None
    max_iteration = 1
    if not is_predict_only():
        while mega_batch_itr <= max_iteration:
            sess = tf_util.get_session()
            sess.close()
            tf.reset_default_graph()
            print(" Batch iteration ", mega_batch_itr)
            #gym.Env.metadata['_mega_batch_itr'] = mega_batch_itr
            print("(rank , size) = ", mpi_util.get_local_rank_size(MPI.COMM_WORLD))

            if TRAIN_WITH_RAY:
                save_in_sub_folder = InceptcurrentDT
                args, args_dict = default_args(save_in_sub_folder=save_in_sub_folder)
                ray_train(save_in_sub_folder=args_dict['save_path'])
            else:
                print("(rank , size) = ", mpi_util.get_local_rank_size(MPI.COMM_WORLD))
                if len(sys_args) <= 1:
                    save_in_sub_folder = None
                    if max_iteration > 1:
                        save_in_sub_folder = InceptcurrentDT
                    args, args_dict = default_args(save_in_sub_folder=save_in_sub_folder)
                policy = run.main(args)
                MPI.COMM_WORLD.barrier()


            # print("policy training args ", args,"\n\n")
            mega_batch_itr += 1

            play_env = gym.make(play_env_id)
            '''try:
                play(play_env, policy)
            except Exception as e:
                print("Could not play the prediction after training due to error  ", e)'''
            # from tensorboard import main as tb
            # tb.main()

    else:
        DFLT_ARGS = default_args()
        loaded_file_correctly = (
            'load_path' in stringarg for stringarg in DFLT_ARGS)
        #play_env = gym.make(play_env_id)
        #policy = run.main(DFLT_ARGS)
        # Just try to Play
        while loaded_file_correctly:
            play(play_env, policy)
