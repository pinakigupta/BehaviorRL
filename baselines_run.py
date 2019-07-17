######################################################################
#          Deep Reinforcement Learning for Autonomous Driving
#                  Created/Modified on: February 5, 2019
#                      Author: Munir Jojo-Verge
#######################################################################
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
import ray
from ray.tune import Experiment, run_experiments, register_env, sample_from
from ray.tune.schedulers import PopulationBasedTraining, AsyncHyperBandScheduler
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

register_env(train_env_id, lambda config: TwoWayEnv(config))
redis_add = ray.services.get_node_ip_address() + ":6379"
try:
    ray.init(redis_add)
except:
    subprocess.run(["sudo", "pkill", "redis-server"]) # Kill the redis-server. This seems the surest way to kill it
    ray.shutdown()
    ray.init()
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


def explore(config):
        # ensure we collect enough timesteps to do sgd
        if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
            config["train_batch_size"] = config["sgd_minibatch_size"] * 2
        # ensure we run at least one sgd iter
        if config["num_sgd_iter"] < 1:
            config["num_sgd_iter"] = 1
        return config


pbt = PopulationBasedTraining(
                                        time_attr="training_iteration",
                                        metric="episode_reward_mean",
                                        mode="max",
                                        perturbation_interval=25,
                                        resample_probability=0.25,
                                        # Specifies the mutations of these hyperparams
                                        hyperparam_mutations={
                                                                "lambda": lambda: random.uniform(0.9, 1.0),
                                                                "clip_param": lambda: random.uniform(0.01, 0.5),
                                                                "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
                                                                "num_sgd_iter": lambda: random.randint(1, 30),
                                                                "sgd_minibatch_size": lambda: random.randint(128, 16384),
                                                                "train_batch_size": lambda: random.randint(2000, 160000),
                                                             },
                                        custom_explore_fn=explore
                                      )

def ray_train(save_in_sub_folder=None):
    subprocess.run(["chmod", "-R", "a+rwx", save_in_sub_folder + "/"])
    # Postprocess the perturbed config to ensure it's still valid



    s3pathname = 's3://datastore-s3/groups/Behavior/Pinaki'                                 
    upload_dir_path = s3pathname + "/" + ray_folder + '/' + InceptcurrentDT
    if save_in_sub_folder is not None:
        local_dir_path = save_in_sub_folder
            #makedirpath(upload_dir_path)
    '''ray_experiment = Experiment(name=None,
                                run="PPO",
                                stop={"training_iteration": int(num_timesteps)},
                                checkpoint_at_end=True,
                                checkpoint_freq=5,
                                local_dir=local_dir_path,
                                #upload_dir=upload_dir_path,
                                config={
                                            "num_gpus_per_worker": 0,
                                            "num_cpus_per_worker": 2,
                                            "gamma": 0.85,
                                            "num_workers": 1,
                                            "env": train_env_id,
                                            # These params are tuned from a fixed starting value.
                                            #"lambda": 0.95,
                                            #"clip_param": 0.2,
                                            #"lr": 1e-4,
                                            # These params start off randomly drawn from a set.
                                            #"num_sgd_iter": sample_from(lambda spec: random.choice([10, 20, 30])),
                                            #"sgd_minibatch_size": sample_from(lambda spec: random.choice([128, 512, 2048])),
                                            #"train_batch_size": sample_from(lambda spec: random.choice([10000, 20000, 40000])),
                                       },
                                )


    run_experiments(ray_experiment,
                    resume=False,
                    reuse_actors=False,
                    scheduler=pbt,
                    verbose=False,
                    ) '''

    ray.tune.run(
                    "PPO",
                    name="pygame-ray",
                    stop={"training_iteration": int(num_timesteps)},
                    scheduler=pbt,
                    checkpoint_freq=20,
                    checkpoint_at_end=True,
                    local_dir=local_dir_path,
                    #upload_dir=upload_dir_path,
                    verbose=True,
                    queue_trials=False,
                    **{
                        #"env": train_env_id,
                        "num_samples": 8,
                        "config" :{
                                    "num_gpus_per_worker": 0,
                                    "num_cpus_per_worker": 1,
                                    "gamma": 0.85,
                                    "num_workers": 50,
                                    "env": train_env_id,
                                    # These params are tuned from a fixed starting value.
                                    "lambda": 0.95,
                                    "clip_param": 0.2,
                                    "lr": 1e-4,
                                    # These params start off randomly drawn from a set.
                                    "num_sgd_iter": sample_from(lambda spec: random.choice([10, 20, 30])),
                                    "sgd_minibatch_size": sample_from(lambda spec: random.choice([128, 512, 2048])),
                                    "train_batch_size": sample_from(lambda spec: random.choice([10000, 20000, 40000])),
                                },
                    }
                    

                )                       


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
        else:
            play_env = gym.make(play_env_id)
            DFLT_ARGS, _ = default_args()
            loaded_file_correctly = ('load_path' in stringarg for stringarg in DFLT_ARGS)
            policy = run.main(DFLT_ARGS)
        # Just try to Play
            while loaded_file_correctly:
                play(play_env, policy)
