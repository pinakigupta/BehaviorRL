import urban_env
import subprocess
import tensorflow as tf
import random
from settings import req_dirs, models_folder, ray_folder
import sys, os, time
import ray
from ray.tune import Experiment, Trainable, run_experiments, register_env, sample_from
from ray.tune.schedulers import PopulationBasedTraining, AsyncHyperBandScheduler
from ray.tune.ray_trial_executor import RayTrialExecutor


from handle_model_files import train_env_id, play_env_id, alg, network, num_timesteps, homepath, RUN_WITH_RAY, InceptcurrentDT


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
                    resume=True,
                    trial_executor=RayTrialExecutor(),
                    **{
                        #"env": train_env_id,
                        "num_samples": 8,
                        "config" :{
                                    "num_gpus_per_worker": 0,
                                    "num_cpus_per_worker": 1,
                                    "gamma": 0.85,
                                    "num_workers": 1,
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
