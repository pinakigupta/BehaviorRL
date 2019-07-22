import urban_env
import subprocess
import tensorflow as tf
import random
from settings import req_dirs, models_folder, ray_folder
import sys
import os
import time
import ray
from ray.tune import Experiment, Trainable, run_experiments, register_env, sample_from
from ray.tune.schedulers import PopulationBasedTraining, AsyncHyperBandScheduler
from ray.tune.ray_trial_executor import RayTrialExecutor
from ray.rllib.agents.trainer_template import build_trainer

import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.impala as impala


from handle_model_files import train_env_id, play_env_id, alg, network, num_timesteps, homepath, RUN_WITH_RAY, InceptcurrentDT, is_predict_only
from handle_model_files import pathname
from urban_env.envs.two_way_env import TwoWayEnv

DEFAULT_CONFIG = {
    "observation": {
        "type": "Kinematics",
        "features": ['x', 'y', 'vx', 'vy', 'psi'],
        "vehicles_count": 6
    },
    "other_vehicles_type": "urban_env.vehicle.behavior.IDMVehicle",
    "centering_position": [0.3, 0.5],
    "duration": 20,
    "_predict_only": is_predict_only(),
    "_mega_batch_itr": 3,
}


register_env(train_env_id, lambda config: TwoWayEnv(config))
#register_env(play_env_id, lambda config: TwoWayEnv(config))
redis_add = ray.services.get_node_ip_address() + ":6379"

if is_predict_only():
    ray.init(num_gpus=0, local_mode=False)
else:
    try:
        ray.init(redis_add)
    except:
        # Kill the redis-server. This seems the surest way to kill it
        subprocess.run(["sudo", "pkill", "redis-server"])
        ray.shutdown()
        ray.init(num_gpus=0, local_mode=False)


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
    perturbation_interval=7,
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
        # makedirpath(upload_dir_path)
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

    from ray.rllib.agents.impala.vtrace_policy import VTraceTFPolicy

    impala_config = impala.DEFAULT_CONFIG.copy()
    impala_config["num_gpus"] = 0
    algo = "IMPALA"
    ImpalaTrainer = build_trainer(name="IMPALA",
                                  default_config=impala_config,
                                  default_policy=VTraceTFPolicy,
                                  validate_config=impala.impala.validate_config,
                                  get_policy_class=impala.impala.choose_policy,
                                  make_workers=impala.impala.defer_make_workers,
                                  make_policy_optimizer=impala.impala.make_aggregators_and_optimizer,
                                  mixins=[impala.impala.OverrideDefaultResourceRequest])

    checkpt = 2800
    available_cluster_cpus = int(ray.cluster_resources().get("CPU"))
    if is_predict_only():
        delegated_cpus=1
    else:
        delegated_cpus=available_cluster_cpus-2

    ray.tune.run(
        ImpalaTrainer,
        name="pygame-ray",
        stop={"training_iteration": int(num_timesteps)},
        # scheduler=pbt,
        checkpoint_freq=int(num_timesteps)//10,
        checkpoint_at_end=True,
        local_dir=local_dir_path,
        # upload_dir=upload_dir_path,
        verbose=True,
        queue_trials=False,
        resume=True,
        # trial_executor=RayTrialExecutor(),
        #resources_per_trial = {"cpu": 216, "gpu": 0},
        #restore=pathname + "/" + ray_folder + "/" + "20190721-021730"+"/pygame-ray/"+algo+"_"+train_env_id +
        #"_0_"+"2019-07-21_02-17-42lcyu3tu7" + "/checkpoint_" + str(checkpt) + "/checkpoint-" + str(checkpt),
        **{
            "num_samples": 1,
            "config": {
                "num_gpus_per_worker": 0,
                "num_cpus_per_worker": 1,
                # "gpus": 0,
                "gamma": 0.85,
                "num_workers": delegated_cpus,
                "env": train_env_id,
                # These params are tuned from a fixed starting value.
                # "lambda": 0.95,
                # "clip_param": 0.2,
                # "lr": 1e-4,
                # These params start off randomly drawn from a set.
                # "num_sgd_iter": sample_from(lambda spec: random.choice([10, 20, 30])),
                # "sgd_minibatch_size": sample_from(lambda spec: random.choice([128, 512, 2048])),
                # "train_batch_size": sample_from(lambda spec: random.choice([10000, 20000, 40000])),
            },
        }
    )

    subprocess.run(["chmod", "-R", "a+rwx", ray_folder + "/"])


def ray_play():
    import gym
    gym.make(play_env_id).reset()
    subprocess.run(["chmod", "-R", "a+rwx", ray_folder + "/"])
    algo = "IMPALA"
    checkpt = 629  # which checkpoint file to play
    results_folder = pathname + "/" + ray_folder + "/" + "20190721-021730"+"/pygame-ray/"+algo+"_"+play_env_id + \
        "_0_"+"2019-07-21_02-17-42lcyu3tu7" + "/checkpoint_" + \
        str(checkpt) + "/checkpoint-" + str(checkpt)
    #results_folder = "PPO_two-way-v0_0_2019-07-15_17-11-45avp2pc6k"
    # results_folder = pathname + "/" + ray_folder + "/" + "pygame-ray/" + results_folder + \
    #    "/checkpoint_" + str(checkpt) +"/checkpoint-" + str(checkpt)
    print("results_folder = ", results_folder)
    subprocess.run(["rllib", "rollout", results_folder, "--run", algo, "--env", play_env_id, "--steps", "10000"])
    subprocess.run(["chmod", "-R", "a+rwx", ray_folder + "/"])
