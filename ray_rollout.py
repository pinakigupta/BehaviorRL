import collections
import pickle
import os
import copy
import subprocess
import glob
import ray
import gym
from multiprocessing import Process

from torch import double



import time
import zmq, zlib, pickle
import numpy as np

print(" print(pickle.format_version)" , pickle.format_version)

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

def send_zipped_pickle(socket, obj, flags=0, protocol=-1):
    """pickle an object, and zip the pickle before sending it"""
    p = pickle.dumps(obj, protocol)
    z = zlib.compress(p)
    return socket.send(z, flags=flags)

def recv_zipped_pickle(socket, flags=0, protocol=-1):
    """inverse of send_zipped_pickle"""
    z = socket.recv(flags)
    p = zlib.decompress(z)
    return pickle.loads(p)

# obj=np.random.rand(2, 30)

# from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID


from settings import req_dirs, models_folder, ray_folder
from handle_model_files import train_env_id, play_env_id, alg, network, num_timesteps, RUN_WITH_RAY, InceptcurrentDT, is_predict_only
from handle_model_files import pathname, copy_terminal_output_file, terminal_output_file_name


def filetonum(filename):
    try:
        return int(filename.split('_')[-1])
    except:
        return -1


def dirsearch(resultstr):
    print("resultstr ", resultstr)
    for dirname, dirnames, filenames in os.walk('/'):
        if '.git' in dirnames:
            # don't go into any .git directories.
            dirnames.remove('.git')
        for subdirname in dirnames: 
            if 'ray_results' in dirname:         
                print(" subdirname ", subdirname, " , dirname ", dirname)
            if resultstr in subdirname:
                print("resultstr ", resultstr, " subdirname ", subdirname, " dirname ", dirname)
                return dirname+"/" + subdirname         
        for subdirname in filenames:            
            if resultstr in subdirname:
                print("resultstr ", resultstr, " subdirname ", subdirname, " dirname ", dirname)
                return dirname+"/"+subdirname
                return(os.path.join(dirname, subdirname))



def retrieve_ray_folder_info(target_folder, checkpt=None):
    local_restore_path = dirsearch(target_folder)
    restore_folder = local_restore_path + "/pygame-ray/"
    print("target_folder = ", target_folder , " ", "restore_folder = ", restore_folder)
    subdir = next(os.walk(restore_folder))[1][0]
    restore_folder = restore_folder + subdir + "/"
    all_checkpt_folders = glob.glob(restore_folder+'/*')
    last_checkpt_folder = max(all_checkpt_folders, key=filetonum)
    if checkpt is None:
        checkpt = filetonum(last_checkpt_folder)
    restore_folder = restore_folder + "checkpoint_" + \
        str(checkpt) + "/checkpoint-" + str(checkpt)
    assert(os.path.exists(restore_folder))
    if 'algo' not in locals():
        algo = subdir.split('_')[0]
    return restore_folder, local_restore_path, algo


def rollout(agent, env_name, num_steps, out=None, no_render=True, intent_predict=False):
    from ray.rllib.rollout import default_policy_agent_mapping, DefaultMapping
    from ray.rllib.env import MultiAgentEnv
    from ray.rllib.env.base_env import _DUMMY_AGENT_ID
    policy_agent_mapping = default_policy_agent_mapping


    

    '''if env_name is not None:
        env = gym.make(env_name)
        multiagent = False
        use_lstm = {DEFAULT_POLICY_ID: False}'''

    if hasattr(agent, "workers"):
        env = gym.make(
            env_name) if env_name is not None else agent.workers.local_worker().env
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
        raise ValueError('Env name/id is None and agent has no workers')

    print("action_init ", action_init)

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
        # rollout one trajectory
        from urban_env.utils import print_execution_time
        import time
        current_wall_time = time.time()
        prev_step_time = 0
        prev_action_time = 0
        USE_OFFLINE_MODEL = False 
        try:
            while not done and steps < (num_steps):

                current_wall_time = time.time()
                sim_loop_time = current_wall_time-prev_step_time
                action_loop_time =  current_wall_time-prev_action_time
                
                if sim_loop_time < 1/env.config["SIMULATION_FREQUENCY"]:
                    #print("loop time (in ms) ", round(1e3*sim_loop_time, 2))
                    continue

                multi_obs = obs if multiagent else {_DUMMY_AGENT_ID: obs}                                  
                if action_loop_time > 1/env.config["POLICY_FREQUENCY"]:
                    if USE_OFFLINE_MODEL:
                        # import torch
                        # torch_policy = torch.jit.load('policy.pt')
                        # torch_obs = multi_obs[_DUMMY_AGENT_ID]
                        # print("torch_obs ", torch_obs , type(torch_obs))
                        # torch_tensor_obs = torch.from_numpy(torch_obs)
                        # torch_tensor_obs_double = torch_tensor_obs.double()
                        # torch_tensor_obs_double.type(torch.DoubleTensor)
                        # print(" torch_tensor_obs_double ", torch_tensor_obs_double)
                        # action = torch_policy(torch_tensor_obs_double)
                        # print("action ", action, type(action))
                        message = recv_zipped_pickle(socket, protocol=4)
                        action = message[0]
                        print("Received request: %s" % action)
                        send_zipped_pickle(socket, multi_obs[_DUMMY_AGENT_ID], protocol=4)                    
                    else:
                        action = act(   
                                        multi_obs,
                                        agent,
                                        multiagent,
                                        prev_actions,
                                        prev_rewards,
                                        policy_agent_mapping,
                                        mapping_cache,
                                        use_lstm
                                        )

                        # print("action ", action, type(action))

                #current_wall_time = print_execution_time(current_wall_time, "After calculating action ")
                next_obs, reward, done, _ = env.step(action)
                prev_step_time = time.time()
                if multiagent:
                    for agent_id, r in reward.items():
                        prev_rewards[agent_id] = r
                else:
                    prev_rewards[_DUMMY_AGENT_ID] = reward

                policy_id = mapping_cache.setdefault(
                    _DUMMY_AGENT_ID, policy_agent_mapping(_DUMMY_AGENT_ID))

                #current_wall_time = print_execution_time(current_wall_time, "Before intent pred ")
                if intent_predict:

                    projections = predict_one_step_of_rollout(
                                                                env,
                                                                agent,
                                                                multi_obs,
                                                                action,
                                                                reward,
                                                                policy_id,
                                                                False
                                                            )
                    env.vehicle.projection = projections
                    # env.intent_pred = True
                    no_render = False
                #current_wall_time = print_execution_time(current_wall_time, "After intent pred ")    

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
                import generate_env_data
                print("trying to generate env data at step ", steps)
                generate_env_data.main(["--env", "env"], rollouts)
                    # print("Finished adding env data")
            print("Episode reward", reward_total)
        except Exception as e:
            print("Error ", e , " at step ", steps)
            if out is not None:
                import generate_env_data
                print("trying to generate env data")
                generate_env_data.main(["--env", "env"], rollouts)
                print("Finished adding env data")
            pass


    # conn.close()

        # pickle.dump(rollouts, open(out, "wb"))


def predict_one_step_of_rollout(env, agent, obs, action, reward, policy_id, no_render=True):
    from ray.rllib.env.base_env import _DUMMY_AGENT_ID
    predict_env = copy.deepcopy(env)
    for v in predict_env.road.vehicles:
        v.projection = []
        v.is_projection = True
    predict_env.intent_pred = True
    predict_env.DEFAULT_CONFIG["_predict_only"] = True
    pred_steps = 0
    pred_obs = obs[_DUMMY_AGENT_ID]
    pred_done = False
    pred_action = action
    pred_reward = reward
    max_pred_steps = int(
        predict_env.config["TRAJECTORY_HORIZON"]*predict_env.config["POLICY_FREQUENCY"])
    while not pred_done and pred_steps < max_pred_steps:
        pred_action = agent.compute_action(
            pred_obs,
            prev_action=pred_action,
            prev_reward=pred_reward,
            policy_id=policy_id)
        pred_obs, pred_reward, pred_done, _ = predict_env.step(pred_action)
        pred_steps += 1
    #print("Total pred_steps ", pred_steps)
    return predict_env.vehicle.projection


def act(multi_obs, agent, multiagent, prev_actions, prev_rewards, policy_agent_mapping, mapping_cache, use_lstm):
    from ray.rllib.env.base_env import _DUMMY_AGENT_ID
    action_dict = {}
    # print(" multi_obs.items() ", multi_obs.items())
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
    return action


def ray_retrieve_agent(env_id=play_env_id, config=None):
    # if config is None:
    #    config = gym.make(env_id).config
    LOAD_MODEL_FOLDER = config["LOAD_MODEL_FOLDER"]
    print("LOAD_MODEL_FOLDER", LOAD_MODEL_FOLDER)
    results_folder, _, algo = retrieve_ray_folder_info(LOAD_MODEL_FOLDER)
    print("results_folder = ", results_folder)
    print("algo = ", algo)

    config_path = os.path.join(results_folder, "params.pkl")
    if not os.path.exists(config_path):
        two_up = os.path.abspath(os.path.join(results_folder, "../.."))
        config_path = os.path.join(two_up, "params.pkl")
    with open(config_path, "rb") as f:
        config = pickle.load(f)
    if "num_workers" in config:
        config["num_workers"] = min(2, config["num_workers"])

    from ray.rllib.agents.registry import get_agent_class
    cls = get_agent_class(algo)
    agent = cls(env=None, config=config)
    agent.restore(results_folder)
    policy = agent.get_policy()
    return agent
