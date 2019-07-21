import urban_env
from urban_env.envs.two_way_env import TwoWayEnv
from urban_env.envs.abstract import AbstractEnv
from mpi4py import MPI
import subprocess
import warnings
import glob
import numpy as np
import urban_env
import gym
from gym.envs.registration import registry
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
from baselines import logger
import baselines.run as run

####################



from handle_model_files import create_dirs, req_dirs, models_folder, makedirpath, is_master, is_predict_only, default_args
from handle_model_files import train_env_id, play_env_id, alg, network, num_timesteps, homepath, RUN_WITH_RAY, InceptcurrentDT

gym.Env.metadata['_predict_only'] = is_predict_only()

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

#################################################################





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
            #gym.Env.metadata['_mega_batch_itr'] = mega_batch_itr
            print("(rank , size) = ", mpi_util.get_local_rank_size(MPI.COMM_WORLD))

            if RUN_WITH_RAY:
                #save_in_sub_folder = InceptcurrentDT
                #args, args_dict = default_args(save_in_sub_folder=save_in_sub_folder)
                #ray_train(save_in_sub_folder=args_dict['save_path'])
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
            subprocess.run(["chmod", "-R", "a+rwx", ray_folder + "/"])
            checkpt = 100 # which checkpoint file to play
            results_folder = "PPO_two-way-v0_0_2019-07-15_17-11-45avp2pc6k"
            results_folder = pathname + "/" + ray_folder + "/" + "pygame-ray/" + results_folder + \
                "/checkpoint_" + str(checkpt) +"/checkpoint-" + str(checkpt)
            print("results_folder = ", results_folder)
            subprocess.run(["rllib", "rollout", results_folder, "--run", "PPO", "--env", play_env_id, "--steps", "10000"])
            subprocess.run(["chmod", "-R", "a+rwx", ray_folder + "/"])
        else:
            play_env = gym.make(play_env_id)
            DFLT_ARGS, _ = default_args()
            loaded_file_correctly = ('load_path' in stringarg for stringarg in DFLT_ARGS)
            policy = run.main(DFLT_ARGS)
        # Just try to Play
            while loaded_file_correctly:
                play(play_env, policy)
