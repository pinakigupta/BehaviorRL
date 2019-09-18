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

filename = '/usr/local/lib/python3.6/dist-packages/ray/rllib/rollout.py'
with open(filename, 'r') as original: 
    original_data = original.read()
    linebylinecontent = original.readlines()

#print(linebylinecontent)
#for linecontent in linebylinecontent:
    #print(linecontent)
    #if "import ray" in linecontent :
    #    exit("Exiting code")


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
    "DIFFICULTY_LEVELS": 3,
}

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

def filetonum(filename):
    try:
        return int(filename.split('_')[-1])
    except:
        return -1


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

def retrieve_ray_folder_info(target_folder):
    local_restore_path = pathname + "/" + ray_folder + "/" + target_folder #"20190805-132549"
    restore_folder = local_restore_path + "/pygame-ray/"
    #checkpt = 2800
    subdir = next(os.walk(restore_folder))[1][0]
    restore_folder = restore_folder + subdir + "/" 
    all_checkpt_folders = glob.glob(restore_folder+'/*')
    last_checkpt_folder = max(all_checkpt_folders, key=filetonum)
    if 'checkpt' not in locals():    
        checkpt = filetonum(last_checkpt_folder)
    restore_folder = restore_folder + "checkpoint_" + str(checkpt) + "/checkpoint-" + str(checkpt)
    assert(os.path.exists(restore_folder))
    if 'algo' not in locals():
        algo = subdir.split('_')[0]
    return restore_folder, local_restore_path, algo

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
    ImpalaTrainer = build_trainer(name="IMPALA",
                                  default_config=impala_config,
                                  default_policy=VTraceTFPolicy,
                                  validate_config=impala.impala.validate_config,
                                  get_policy_class=impala.impala.choose_policy,
                                  make_workers=impala.impala.defer_make_workers,
                                  make_policy_optimizer=impala.impala.make_aggregators_and_optimizer,
                                  mixins=[impala.impala.OverrideDefaultResourceRequest])
    
    if is_predict_only() or LOCAL_MODE:
        delegated_cpus = 1
    else:
        delegated_cpus = available_cluster_cpus-2

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

    checkpoint_freq=int(num_timesteps)//min(int(num_timesteps),20)

    ray_trials = ray.tune.run(
            algo,
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



import collections
from ray.rllib.rollout import default_policy_agent_mapping, DefaultMapping
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.env import MultiAgentEnv
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
import pickle
import os
import copy

def rollout(agent, env_name, num_steps, out=None, no_render=True):
    policy_agent_mapping = default_policy_agent_mapping

    if hasattr(agent, "workers"):
        env = agent.workers.local_worker().env
        multiagent = isinstance(env, MultiAgentEnv)
        if agent.workers.local_worker().multiagent:
            policy_agent_mapping = agent.config["multiagent"][
                "policy_mapping_fn"]

        policy_map = agent.workers.local_worker().policy_map
        state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
        use_lstm = {p: len(s) > 0 for p, s in state_init.items()}
        action_init = {
            p: m.action_space.sample()
            for p, m in policy_map.items()
        }
    else:
        env = gym.make(env_name)
        multiagent = False
        use_lstm = {DEFAULT_POLICY_ID: False}

    if out is not None:
        rollouts = []
    steps = 0
    while steps < (num_steps or steps + 1):
        mapping_cache = {}  # in case policy_agent_mapping is stochastic
        if out is not None:
            rollout = []
        obs = env.reset()
        agent_states = DefaultMapping(
            lambda agent_id: state_init[mapping_cache[agent_id]])
        prev_actions = DefaultMapping(
            lambda agent_id: action_init[mapping_cache[agent_id]])
        prev_rewards = collections.defaultdict(lambda: 0.)
        done = False
        reward_total = 0.0
        while not done and steps < (num_steps or steps + 1):
            multi_obs = obs if multiagent else {_DUMMY_AGENT_ID: obs}
            action_dict = {}
            for agent_id, a_obs in multi_obs.items():
                if a_obs is not None:
                    policy_id = mapping_cache.setdefault(
                        agent_id, policy_agent_mapping(agent_id))
                    p_use_lstm = use_lstm[policy_id]
                    if p_use_lstm:
                        a_action, p_state, _ = agent.compute_action(
                            a_obs,
                            state=agent_states[agent_id],
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id)
                        agent_states[agent_id] = p_state
                    else:
                        a_action = agent.compute_action(
                            a_obs,
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id)
                    action_dict[agent_id] = a_action
                    prev_actions[agent_id] = a_action
            action = action_dict




            action = action if multiagent else action[_DUMMY_AGENT_ID]
            next_obs, reward, done, _ = env.step(action)
            if multiagent:
                for agent_id, r in reward.items():
                    prev_rewards[agent_id] = r
            else:
                prev_rewards[_DUMMY_AGENT_ID] = reward

            predict_env = copy.deepcopy(env)
            predict_env.DEFAULT_CONFIG["_predict_only"] = True
            pred_actions = []
            pred_steps = 0
            pred_obs = multi_obs[_DUMMY_AGENT_ID]
            pred_done = False
            pred_action = action
            pred_reward = reward
            policy_id = mapping_cache.setdefault(
                        _DUMMY_AGENT_ID, policy_agent_mapping(_DUMMY_AGENT_ID))
            while not pred_done and pred_steps < 4:
                pred_action = agent.compute_action(
                            pred_obs,
                            prev_action=pred_action,
                            prev_reward=pred_reward,
                            policy_id=policy_id)
                pred_obs, pred_reward, pred_done, _ = predict_env.step(pred_action)
                pred_actions.append(pred_action)
                pred_steps += 1

            env.set_actions(pred_actions)

            if multiagent:
                done = done["__all__"]
                reward_total += sum(reward.values())
            else:
                reward_total += reward
            if not no_render:
                env.render()
            if out is not None:
                rollout.append([obs, action, next_obs, reward, done])
            steps += 1
            obs = next_obs
        if out is not None:
            rollouts.append(rollout)
        print("Episode reward", reward_total)

    if out is not None:
        pickle.dump(rollouts, open(out, "wb"))


def ray_play():
    import gym
    #env = gym.make(play_env_id).reset()
    subprocess.run(["chmod", "-R", "a+rwx", ray_folder + "/"])
    subprocess.run(["xhost", "+"], shell=True)
    LOAD_MODEL_FOLDER = "20190916-031304" # Location of previous model for prediction 
    results_folder, _ , algo = retrieve_ray_folder_info(LOAD_MODEL_FOLDER)
    print("results_folder = ", results_folder) 
    print("algo = ", algo)
    
    config_path = os.path.join(results_folder, "params.pkl")
    if not os.path.exists(config_path):
        two_up = os.path.abspath(os.path.join(results_folder ,"../.."))
        config_path = os.path.join(two_up, "params.pkl")
    with open(config_path, "rb") as f:
        config = pickle.load(f)
    if "num_workers" in config:
        config["num_workers"] = min(2, config["num_workers"])

    cls = get_agent_class(algo)
    agent = cls(env=play_env_id, config=config) 
    agent.restore(results_folder)
    rollout(agent=agent,
            env_name=None,
            num_steps=10000,
            no_render=False,
            out=None)
    subprocess.run(["chmod", "-R", "a+rwx", ray_folder + "/"])


