######################################################################
#          Deep Reinforcement Learning for Autonomous Driving
#                  Created/Modified on: February 5, 2019
#                      Author: Munir Jojo-Verge
#######################################################################
import urban_env
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



####################

pathname = os.getcwd()
open_ai_baselines_dir = pathname + '/open_ai_baselines'
sys.path.append(open_ai_baselines_dir)


from baselines.common import tf_util, mpi_util
from baselines.common.vec_env import VecEnv
from baselines import logger
import baselines.run as run

####################

from handle_model_files import create_dirs, is_master, is_predict_only, default_args
from handle_model_files import train_env_id, play_env_id, alg, network, num_timesteps

gym.Env.metadata['_predict_only'] = is_predict_only()

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")




register_env(train_env_id, lambda _: TwoWayEnv)
redis_add = ray.services.get_node_ip_address() + ":6379"
try:
    ray.init(redis_add)
except:
    ray.shutdown()
    ray.init()
#################################################################
first_default_args_call = True
LOAD_PREV_MODEL = True





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
                                        "stop": {"training_iteration": int(100)},
                                        "checkpoint_at_end": True,
                                        "checkpoint_freq": 1,
                                        "config": {
                                                    # "env_config": env_config,
                                                    "num_gpus_per_worker": 0,
                                                    "num_cpus_per_worker": 1,
                                                    "gamma": 0.85,
                                                    "num_workers": 2,
                                                  },
                                        #"local_dir": save_in_sub_folder,
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
                #save_in_sub_folder = InceptcurrentDT
                #args, args_dict = default_args(save_in_sub_folder=save_in_sub_folder)
                #ray_train(save_in_sub_folder=args_dict['save_path'])
                ray_train()
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
        play_env = gym.make(play_env_id)
        if TRAIN_WITH_RAY:
            subprocess.run(["rllib", "rollout", homepath+"/ray_results/pygame-ray/PPO_TwoWayEnv_0_2019-07-12_21-57-55a2nhe7zt/checkpoint_1/checkpoint-1",\
                     "--run", "PPO", "--env", play_env_id, "--steps", "10000"])
        else:
            DFLT_ARGS = default_args()
            loaded_file_correctly = ('load_path' in stringarg for stringarg in DFLT_ARGS)
            policy = run.main(DFLT_ARGS)
        # Just try to Play
            while loaded_file_correctly:
                play(play_env, policy)
