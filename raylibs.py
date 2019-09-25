import urban_env
import subprocess
import tensorflow as tf
import random
from settings import req_dirs, models_folder, ray_folder
import sys
import os
import time
import glob
import redis
import ray
from ray.tune import Experiment, Trainable, run_experiments, register_env, sample_from
from ray.tune.schedulers import PopulationBasedTraining, AsyncHyperBandScheduler
from ray.tune.ray_trial_executor import RayTrialExecutor
from ray.rllib.agents.trainer_template import build_trainer


import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.impala as impala


from handle_model_files import train_env_id, play_env_id, alg, network, num_timesteps, RUN_WITH_RAY, InceptcurrentDT, is_predict_only
from handle_model_files import pathname, copy_terminal_output_file, terminal_output_file_name
import handle_model_files
from urban_env.envs.two_way_env import TwoWayEnv
from urban_env.envs.multilane_env import MultilaneEnv
from urban_env.envs.multitask_env import MultiTaskEnv
from urban_env.envs.abstract import AbstractEnv

from ray_rollout import retrieve_ray_folder_info, ray_retrieve_agent, filetonum, rollout

register_env('multilane-v0', lambda config: urban_env.envs.MultilaneEnv(config))
register_env('merge-v0', lambda config: urban_env.envs.MergeEnv(config))
register_env('roundabout-v0', lambda config: urban_env.envs.RoundaboutEnv(config))
register_env('two-way-v0', lambda config: urban_env.envs.TwoWayEnv(config))
register_env('parking-v0', lambda config: urban_env.envs.ParkingEnv(config))
register_env('parking_2outs-v0', lambda config: urban_env.envs.ParkingEnv_2outs(config))
register_env('LG-SIM-ENV-v0', lambda config: urban_env.envs.LG_Sim_Env(config))
register_env('multitask-v0', lambda config: MultiTaskEnv(config))

redis_add = ray.services.get_node_ip_address() + ":6379"

def get_immediate_subdirectories(a_dir):
    if not a_dir:
        return
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def purge_ray_dirs():
    all_folders = glob.glob(pathname + "/" + ray_folder+'/*')
    for folder in all_folders:
        results_folder = folder + "/pygame-ray"
        if not os.path.exists(results_folder):
            continue
        subdirs = get_immediate_subdirectories(results_folder)
        if not subdirs:
            continue
        subdir = subdirs[0]
        results_folder = results_folder + "/" + subdir 
        all_checkpt_folders = get_immediate_subdirectories(results_folder)
        if all_checkpt_folders:
            continue
        import shutil
        print("purging folder ", folder)
        shutil.rmtree(folder)
purge_ray_dirs()

def ray_node_ips():
    @ray.remote
    def f():
        time.sleep(0.01)
        return ray.services.get_node_ip_address()

    # Get a list of the IP addresses of the nodes that have joined the cluster.
    list_of_ips = set(ray.get([f.remote() for _ in range(1000)]))
    return list_of_ips


def ray_cluster_status_check(ray_yaml_file="Ray-Cluster.yaml" , initial_workers_check=True):
    import yaml
    with open(ray_yaml_file, 'r') as stream:
        try:
            yaml_data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    min_cluster_nodes = yaml_data['min_workers']+1 #+1 for head node, parse yaml file for min workers
    init_cluster_nodes = yaml_data['initial_workers']+1 #+1 for head node, , parse yaml file for initial workers
    if initial_workers_check:
        min_cluster_nodes = max(min_cluster_nodes,init_cluster_nodes)
    while True: #run waiting for the entire cluster to be initialized (or something else is wrong ?)
        available_nodes = len(ray.nodes()) # gives all available nodes "ready" for compute (ex: not initializing)
        if available_nodes >= min_cluster_nodes:
            print("All nodes available. min_cluster_nodes count ", min_cluster_nodes,
                  "available_nodes count ", available_nodes)
            break
        else:
            print("available nodes count ", available_nodes," min cluster nodes required",
            min_cluster_nodes)
            print("ray nodes  ", ray.nodes())
            print("cluster_resources ", ray.cluster_resources())
            print("available_resources ", ray.available_resources())



LOCAL_MODE = False  #Use local mode for debug purposes
if is_predict_only():
    try:
        subprocess.run(["sudo", "pkill", "redis-server"])
        subprocess.run(["sudo", "pkill", "ray_RolloutWork"])
    except:
        print("ray process not running")
    LOCAL_MODE = True    
    ray.init(num_gpus=0, local_mode=True)
else:
    try: # to init in the cluster
        ray.init(redis_add)
        ray_cluster_status_check()
        available_cluster_cpus = int(ray.cluster_resources().get("CPU"))
        LOCAL_MODE = False
    except: # try to init in your machine/isolated compute instance
        # Kill the redis-server. This seems the surest way to kill it
        subprocess.run(["sudo", "pkill", "redis-server"])
        subprocess.run(["sudo", "pkill", "ray_RolloutWork"])
        try: # shutting down for a freash init assuming ray init process started when attempted to run in a cluster
            ray.shutdown()
        except:
            print("ray shutdown failed. Perhaps ray was not initialized ?")

        ray.init(num_gpus=0, local_mode=LOCAL_MODE)
        if not LOCAL_MODE:
            available_cluster_cpus = int(ray.available_resources().get("CPU"))
            print("ray nodes  ", ray.nodes())
            print("cluster_resources ", ray.cluster_resources())
            print("available_resources ", ray.available_resources())
            

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

def on_episode_start(info):
    print(info.keys())  # -> "env", 'episode"



def on_train_result(info):
    result = info["result"]
    if result["episode_reward_mean"] > 1200:
        curriculam = 4
    else:
        curriculam = 2
    trainer = info["trainer"]
    trainer.workers.foreach_worker(
        lambda ev: ev.foreach_env(
            lambda env: env.set_curriculam(curriculam)))

def ray_train(save_in_sub_folder=None):
    subprocess.run(["chmod", "-R", "a+rwx", save_in_sub_folder + "/"])
    # Postprocess the perturbed config to ensure it's still valid

    s3pathname = 's3://datastore-s3/groups/Behavior/Pinaki'
    upload_dir_path = s3pathname + "/" + ray_folder + '/' + InceptcurrentDT
    if save_in_sub_folder is not None:
        local_dir_path = save_in_sub_folder
        # makedirpath(upload_dir_path)


    from ray.rllib.agents.impala.vtrace_policy import VTraceTFPolicy

    if is_predict_only() or LOCAL_MODE:
        delegated_cpus = 1
    else:
        delegated_cpus = available_cluster_cpus-2

    impala_config = impala.DEFAULT_CONFIG.copy()
    impala_config["num_gpus"] = 0
    ImpalaTrainer = build_trainer(name="IMPALA",
                                  default_config=impala_config,
                                  default_policy=VTraceTFPolicy,
                                  validate_config=impala.impala.validate_config,
                                  get_policy_class=impala.impala.choose_policy,
                                  make_workers=impala.impala.defer_make_workers,
                                  make_policy_optimizer=impala.impala.make_aggregators_and_optimizer,
                                  mixins=[impala.impala.OverrideDefaultResourceRequest]
                                 )
    
    from ray.rllib.agents.ppo import PPOTrainer
    from ray.rllib.optimizers import AsyncGradientsOptimizer

    def make_async_optimizer(workers, config):
        return AsyncGradientsOptimizer(workers, grads_per_step=100)

    CustomTrainer = PPOTrainer.with_updates(
        make_policy_optimizer=make_async_optimizer)
                                

    restore_folder=None
    algo = "PPO" # RL Algorithm of choice
    LOAD_MODEL_FOLDER = "20190828-201729" # Location of previous model (if needed) for training 
    RESTORE_COND = "NONE" # RESTORE: Use a previous model to start new training 
                          # RESTORE_AND_RESUME: Use a previous model to finish previous unfinished training 
                          # NONE: Start fresh
    if RESTORE_COND == "RESTORE_AND_RESUME":
        restore_folder, local_restore_path, _ = retrieve_ray_folder_info(LOAD_MODEL_FOLDER)
        local_dir=local_restore_path
        resume=True
    elif RESTORE_COND == "RESTORE":
        restore_folder, local_restore_path, _ = retrieve_ray_folder_info(LOAD_MODEL_FOLDER)
        local_dir=local_dir_path
        resume=False
    else:
        local_dir=local_dir_path
        resume=False

    checkpoint_freq=int(num_timesteps)//min(int(num_timesteps), 20)

    import settings
    retrieved_agent_policy = settings.retrieved_agent_policy

    ray_trials = ray.tune.run(
            CustomTrainer,
            name="pygame-ray",
            stop={"training_iteration": int(num_timesteps)},
            checkpoint_freq=checkpoint_freq,
            checkpoint_at_end=True,
            local_dir=local_dir,
            # upload_dir=upload_dir_path,
            verbose=True,
            queue_trials=False,
            resume=resume,
            # scheduler=pbt,
            # trial_executor=RayTrialExecutor(),
            # resources_per_trial={"cpu": delegated_cpus, "gpu": 0},
            restore=restore_folder,
            **{
                "num_samples": 1,
                "config": {
                    "num_gpus_per_worker": 0,
                    #"num_cpus_per_worker": 1,
                    # "gpus": 0,
                    "gamma": 0.85,
                    "num_workers": delegated_cpus,
                    "num_envs_per_worker": 2,
                    "env": train_env_id,
                    "remote_worker_envs": False,
                    "model": {
                                #    "use_lstm": True,
                                    "fcnet_hiddens": [256, 256, 256],
                            }, 
                    "env_config": {
                                    "retrieved_agent_policy": 1,
                                  },               
                    #"callbacks": {
                                #  "on_episode_start": ray.tune.function(on_episode_start),
                    #             },
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
    copy_terminal_output_file(save_folder=local_dir_path, terminal_output_file_name=terminal_output_file_name)
    subprocess.run(["chmod", "-R", "a+rwx", ray_folder + "/"])


def ray_play():
    #subprocess.run(["chmod", "-R", "a+rwx", ray_folder + "/"])
    agent=ray_retrieve_agent()

    rollout(agent=agent,
            env_name=None,
            num_steps=10000,
            no_render=False,
            out=None,
            predict=False)
    #subprocess.run(["chmod", "-R", "a+rwx", ray_folder + "/"])
