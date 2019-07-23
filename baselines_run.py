import urban_env
from urban_env.envs.two_way_env import TwoWayEnv
from urban_env.envs.abstract import AbstractEnv
from mpi4py import MPI
import subprocess
import warnings
import glob
import numpy as np
import urban_env
import tensorflow as tf
import random
from settings import req_dirs, models_folder, ray_folder
import sys
import os
from os.path import dirname, abspath
import time
import pprint
pp = pprint.PrettyPrinter(indent=4)
####################



####################

pathname = os.getcwd()
open_ai_baselines_dir = pathname + '/open_ai_baselines'
sys.path.append(open_ai_baselines_dir)


from baselines.common import tf_util, mpi_util
from baselines.common.vec_env import VecEnv
import baselines.run as run

####################



from handle_model_files import create_dirs, req_dirs, models_folder, makedirpath, is_master, is_predict_only, default_args
from handle_model_files import train_env_id, play_env_id, alg, network, num_timesteps, homepath, RUN_WITH_RAY, InceptcurrentDT
import gym
#gym.Env.metadata['_predict_only'] = is_predict_only()

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

#################################################################


if __name__ == "__main__":

    mega_batch_itr = 1
    sys_args = sys.argv

    
    policy = None
    play_env = None
    max_iteration = 1
    if not is_predict_only():
        while mega_batch_itr <= max_iteration:
            sess = tf_util.get_session()
            sess.close()
            tf.reset_default_graph()
            print(" Batch iteration ", mega_batch_itr)
            #gym.Env.metadata['DIFFICULTY_LEVELS'] = mega_batch_itr
            print("(rank , size) = ", mpi_util.get_local_rank_size(MPI.COMM_WORLD))

            if RUN_WITH_RAY:
                from raylibs import ray_train
                ray_train(save_in_sub_folder=pathname + "/" + ray_folder + "/" + InceptcurrentDT)
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
        if RUN_WITH_RAY:
            from raylibs import ray_play
            ray_play()
        else:
            from baselinelibs import baselines_play
            play_env = gym.make(play_env_id)
            DFLT_ARGS, _ = default_args()
            loaded_file_correctly = ('load_path' in stringarg for stringarg in DFLT_ARGS)
            policy = run.main(DFLT_ARGS)
            # Just try to Play
            while loaded_file_correctly:
                baselines_play(play_env, policy)
