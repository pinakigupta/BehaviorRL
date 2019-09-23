######################################################################
#          Deep Reinforcement Learning for Autonomous Driving
#                  Created/Modified on: January 10, 2019
#                      Author: Munir Jojo-Verge
#######################################################################
from ray.cloudpickle import cloudpickle
import pickle
ray_folder = 'ray_results'
run_folder = 'run/'
logs_folder = run_folder + 'logs'
models_folder = run_folder + 'models'

req_dirs = [ray_folder,run_folder, logs_folder, models_folder]

retrieved_agent = None
retrieved_agent_policy = None

def update_policy(new_policy):

    global retrieved_agent_policy 
    retrieved_agent_policy = new_policy



