import gym
from handle_model_files import train_env_id, play_env_id, alg, network, num_timesteps, homepath, RUN_WITH_RAY, InceptcurrentDT
from handle_model_files import create_dirs, req_dirs, models_folder, makedirpath, is_master, is_predict_only, default_args
import urban_env
from urban_env.envs.two_way_env import TwoWayEnv
from urban_env.envs.abstract import AbstractEnv
from mpi4py import MPI
import subprocess
import warnings
import glob
import numpy as np
import tensorflow as tf
import random
from settings import req_dirs, models_folder, ray_folder
import sys
import os
from os.path import dirname, abspath
import time
import pprint
import atexit
from color import color
from ray.tune import register_env


def exit_handler():
    subprocess.run(["chmod", "-R", "a+rwx", "."])


atexit.register(exit_handler)

pp = pprint.PrettyPrinter(indent=4)
####################
try:
    import git
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    print("git HEAD sha is ", sha)
    print("git HEAD message is ", repo.head.object.message)
    print("git branch is ", repo.active_branch)
except:
    print("git python import not working")


####################

pathname = os.getcwd()
open_ai_baselines_dir = pathname + '/open_ai_baselines'
sys.path.append(open_ai_baselines_dir)


####################


# gym.Env.metadata['_predict_only'] = is_predict_only()

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

#################################################################


def main(mainkwargs):
    predict_only = is_predict_only(**mainkwargs)
    mega_batch_itr = 1
    sys_args = sys.argv
    policy = None
    play_env = None
    max_iteration = 1

    config = {
                "LOAD_MODEL_FOLDER": "20200111-051102",
                "RESTORE_COND": "RESTORE", 
                "MODEL":        {
                                #    "use_lstm": True,
                                     "fcnet_hiddens": [256, 128, 128],
                                #     "fcnet_activation": "relu",
                                 }, 
                #"num_workers": 12,                                 
             }    
    if not predict_only:
        if RUN_WITH_RAY:
            register_env('multilane-v0', lambda config: urban_env.envs.MultilaneEnv(config))
            register_env('merge-v0', lambda config: urban_env.envs.MergeEnv(config))
            register_env('roundabout-v0', lambda config: urban_env.envs.RoundaboutEnv(config))
            register_env('two-way-v0', lambda config: urban_env.envs.TwoWayEnv(config))
            register_env('parking-v0', lambda config: urban_env.envs.ParkingEnv(config))
            register_env('parking_2outs-v0', lambda config: urban_env.envs.ParkingEnv_2outs(config))
            register_env('LG-SIM-ENV-v0', lambda config: urban_env.envs.LG_Sim_Env(config))
            register_env('multitask-v0', lambda config: MultiTaskEnv(config))   
                     
            from raylibs import ray_train, ray_init
            from ray_rollout import ray_retrieve_agent
            from settings import update_policy
            available_cluster_cpus, available_cluster_gpus = ray_init(**mainkwargs)
            #play_env = gym.make(play_env_id)
            '''retrieved_agent = ray_retrieve_agent(play_env_id)
            retrieved_agent_policy = retrieved_agent.get_policy()
            update_policy(retrieved_agent_policy)'''
            save_in_sub_folder = pathname + "/" + ray_folder + "/" + InceptcurrentDT
            print("save_in_sub_folder is ", save_in_sub_folder)
            mainkwargs = {**mainkwargs, 
                          **{"save_in_sub_folder": save_in_sub_folder, 
                             "available_cluster_cpus": available_cluster_cpus,
                             "available_cluster_gpus": available_cluster_gpus,
                            }
                         }
            ray_train(config=config, **mainkwargs)
        else:
            while mega_batch_itr <= max_iteration:
                from baselines.common import tf_util, mpi_util
                from baselines.common.vec_env import VecEnv

                sess = tf_util.get_session()
                sess.close()
                tf.reset_default_graph()
                print(" Batch iteration ", mega_batch_itr)
                print("(rank , size) = ", mpi_util.get_local_rank_size(MPI.COMM_WORLD))

                # import baselines.run as run
                print("(rank , size) = ", mpi_util.get_local_rank_size(MPI.COMM_WORLD))
                if len(sys_args) <= 1:
                    save_in_sub_folder = None
                    if max_iteration > 1:
                        save_in_sub_folder = InceptcurrentDT
                    args, args_dict = default_args(
                        save_in_sub_folder=save_in_sub_folder)
                policy = run.main(args)
                MPI.COMM_WORLD.barrier()

                # print("policy training args ", args,"\n\n")
                mega_batch_itr += 1

                play_env = gym.make(play_env_id)
                # from tensorboard import main as tb
                # tb.main()

        print(color.BOLD + 'Successfully ended Training!' + color.END)

    else:
        if RUN_WITH_RAY:
            from raylibs import ray_play, ray_init
            from ray_rollout import ray_retrieve_agent
            from settings import update_policy
            ray_init(**mainkwargs)
            #play_env = gym.make(play_env_id)
            #config=play_env.config
            retrieved_agent = ray_retrieve_agent(env_id=play_env_id, config=config)
            retrieved_agent_policy = retrieved_agent.get_policy()
            update_policy(retrieved_agent_policy)
            print("entering ray play")
            ray_play(env_id=play_env_id, config=config, agent=retrieved_agent)
        else:
            from baselines.common import tf_util, mpi_util
            from baselines.common.vec_env import VecEnv
            # import baselines.run as run
            from baselinelibs import baselines_play
            play_env = gym.make(play_env_id)
            DFLT_ARGS, _ = default_args()
            loaded_file_correctly = ('load_path' in stringarg for stringarg in DFLT_ARGS)
            policy = run.main(DFLT_ARGS)
            # Just try to Play
            while loaded_file_correctly:
                baselines_play(play_env, policy)


if __name__ == "__main__":
    argdict = dict(arg.split('=') for arg in sys.argv[1:])
    argdict = {**argdict,
               **{
                   "LOCAL_MODE": False,
                 }
               }
    main(argdict)
