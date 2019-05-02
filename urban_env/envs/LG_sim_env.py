######################################################################
#          Deep Reinforcement Learning for Autonomous Driving
#                  Created/Modified on: April 17, 2019
#                      Author: Munir Jojo-Verge
#######################################################################

from __future__ import division, print_function, absolute_import
import numpy as np
import pandas
import os
import time
import copy
import gym

from gym import GoalEnv, logger
from gym.spaces import Discrete, Box, Tuple
from gym.utils import colorize, seeding


import lgsvl
from lgsvl.utils import *


from gym import GoalEnv, spaces

VELOCITY_EPSILON = 0.1

class LG_Sim_Env(GoalEnv):
    """       
        LG Simulator Openai Compliyant Environment Class
    """

    COLLISION_REWARD     = -1.0            
    OVER_OTHER_PARKING_SPOT_REWARD = -0.9
    REVERSE_REWARD = -0.2    

    PARKING_MAX_VELOCITY = 7.0 # m/s

    OBS_SCALE = 100
    REWARD_SCALE = np.absolute(COLLISION_REWARD)

    REWARD_WEIGHTS = [7/100, 7/100, 6/100, 6/100, 9/10, 9/10]
    SUCCESS_THRESHOLD = 0.33


    # REWARD_WEIGHTS = [5/100, 5/100, 1/100, 1/100, 5/10, 5/10]
    # SUCCESS_THRESHOLD = 0.27    

    ACTIONS_HOLD_TIME = 1.0
    
    DEFAULT_CONFIG = {        
        "map": 'ParkingLot',
        "observation": {            
            "features": ['x', 'z', 'vx', 'vz', 'cos_h', 'sin_h'],
            "scale": 100,
            "normalize": False
        },
        "parking_spots": 15, #'random', # Parking Spots Per side
        "vehicles_count": 0, #'random', # Total number of cars in the parking (apart from Ego)    
    }    


    def __init__(self):
        
        self.config = self.DEFAULT_CONFIG.copy()
        self.map = self.config["map"]

        self.sim = lgsvl.Simulator(os.environ.get("SIMULATOR_HOST", "127.0.0.1"), 8181) 
        self.control = lgsvl.VehicleControl()

        if self.sim.current_scene == self.map:
            self.sim.reset()
        else:
            self.sim.load(self.map)       

        
        if self.config["parking_spots"] == 'random':
            self.parking_spots = np.random.randint(1,21)
        else:
            self.parking_spots = self.config["parking_spots"]

        if self.config["vehicles_count"] == 'random':
            self.vehicles_count = np.random.randint(self.parking_spots) * 2
        else:
            self.vehicles_count = self.config["vehicles_count"]
        assert (self.vehicles_count < self.parking_spots*2)
        
        self.features = self.config["observation"]["features"]  # Observation features

        # Spaces        
        self.define_spaces()        
                

        self.crashed = False
        
        # self.action_space = spaces.Box(-1., 1., shape=(2,), dtype=np.float32)
        self.REWARD_WEIGHTS = np.array(self.REWARD_WEIGHTS)
        

    def define_spaces(self):
        # Let;s define the action space as: 
        # Throttle: [0 to 1], 
        # Brake: [0 to 1] 
        # Steering Angle: [-1 to 1],
        # reverse: True/False (Discrete 0,1) NOT IN USE FOR NOW
        self.action_space = Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 1]), dtype=np.float32)
        self.previous_action = np.array([0.0, 0.0, False])


        obs = self.reset()
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs["desired_goal"].shape, dtype=np.float32),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype=np.float32),
            observation=spaces.Box(-np.inf, np.inf, shape=obs["observation"].shape, dtype=np.float32),
        ))        

    def reset(self):
        self.crashed = False
        self.sim.reset()                
        self._populate_scene()        
        
        return self.observe()


    def step(self, action):                                
        velocity = np.sqrt(self.ego.state.velocity.x**2 + self.ego.state.velocity.z**2)
        ################################                
        prev_in_reverse = self.previous_action[2].item()
        cmd_reverse = action[2].item()

        allow_switch_gear = np.abs(velocity) < VELOCITY_EPSILON
        if ( (prev_in_reverse and bool(cmd_reverse < 0.0) ) or
             (not prev_in_reverse and bool(cmd_reverse < 0.0) and allow_switch_gear)
        ):
            reverse = True
        elif ((prev_in_reverse >= 0.0 and bool(cmd_reverse >= 0.0) ) or
              (prev_in_reverse <  0.0 and bool(cmd_reverse >= 0.0)  and allow_switch_gear)
        ):
            reverse = False
        else: # You are trying to make an illegal gear switch and you should stay on whatever your last gear is
            reverse = bool(prev_in_reverse < 0.0)
        
        action[2] = reverse

        throttle_brake = -action[0].item()
        if throttle_brake < 0.0: # Only Braking
            self.control.throttle = 0.0
            self.control.breaking = np.abs(throttle_brake)
        else: # Only Throttle
            self.control.throttle = throttle_brake
            self.control.breaking = 0.0
                
        self.control.steering = -action[1].item()
        
        self.control.reverse  = reverse

        self.ego.apply_control(self.control, True)

        self.sim.run(time_limit = self.ACTIONS_HOLD_TIME)
        #self.sim.run()

        print("curr_act, speed:", action, velocity)

        self.previous_action = action

        obs = self.observe()
        info = {
            "is_success": self._is_success(obs['achieved_goal'], obs['desired_goal']),
            "is_collision": int(self.crashed),
            "is_reverse": int(self.control.reverse)
        }

        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], info)
        terminal = self._is_terminal()
        return obs, reward, terminal, info

    def observe(self):                
        
        d = {            
            'x': self.ego.state.transform.position.x,
            'z': self.ego.state.transform.position.z,
            'vx': self.ego.state.velocity.x,
            'vz': self.ego.state.velocity.z,
            'cos_h': np.cos(np.deg2rad(-self.ego.state.transform.rotation.y)),
            'sin_h': np.sin(np.deg2rad(-self.ego.state.transform.rotation.y))
        }
        obs = np.ravel(pandas.DataFrame.from_records([d])[self.features])
        goal = np.ravel(pandas.DataFrame.from_records([self.goal])[self.features])
        obs = {
            "observation": obs / self.OBS_SCALE,
            "achieved_goal": obs / self.OBS_SCALE,
            "desired_goal": goal / self.OBS_SCALE
        }
        return obs

    def _populate_scene(self):
        """
            Create some new random vehicles of a given type, and add them on the road.
        """
        spawns = self.sim.get_spawn()

        state = lgsvl.AgentState()
        state.transform = spawns[0]
        self.ego = self.sim.add_agent("XE_Rigged-apollo", lgsvl.AgentType.EGO, state)
        self.ego.on_collision(self.on_collision)

        # PLACE OTHER VEHICLES ON THE ROAD
        # 10 meters ahead
        sx = spawns[1].position.x - 10.0        
        sz = spawns[1].position.z

        # side = False
        # for i, name in enumerate(["Sedan", "SUV", "Jeep", "HatchBack", "SchoolBus"]):
        #     state = lgsvl.AgentState()
        #     state.transform = spawns[1]

        #     state.transform.position.x = sx - (5 * i)
        #     state.transform.position.z = sz - (int(side) * 8.0)
        #     state.transform.rotation.y = 0
        #     self.sim.add_agent(name, lgsvl.AgentType.NPC, state)

        #     side = not side
        
        #### TESTING THE GOAL LOCATION BY PLACING A BUS
        state = lgsvl.AgentState()
        state.transform = spawns[1]

        state.transform.position.x = sx - 17.0
        state.transform.position.z = sz + 15.0
        state.transform.rotation.y = 0
        self.sim.add_agent("SchoolBus", lgsvl.AgentType.NPC, state)

        # SET THE GOAL LOCATION
        self.goal = {            
            'x': sx - 17.0,
            'z': sz - 8.0,
            'vx': 0,
            'vz': 0,
            'cos_h': 1,
            'sin_h': 0
        }
    
    def distance_2_goal_reward(self, achieved_goal, desired_goal, p=0.5):
        return - np.power(np.dot(self.OBS_SCALE * np.abs(achieved_goal - desired_goal), self.REWARD_WEIGHTS), p)        

    def compute_reward(self, achieved_goal, desired_goal, info, p=0.5):
        """
            Proximity to the goal is rewarded

            We use a weighted p-norm
        :param achieved_goal: the goal that was achieved
        :param desired_goal: the goal that was desired
        :param info: any supplementary information
        :param p: the Lp^p norm used in the reward. Use p<1 to have high kurtosis for rewards in [0, 1]
        :return: the corresponding reward
        """
        
        # return - np.power(np.dot(self.OBS_SCALE * np.abs(achieved_goal - desired_goal), self.REWARD_WEIGHTS), p)

        # DISTANCE TO GOAL
        distance_to_goal_reward = self.distance_2_goal_reward(achieved_goal, desired_goal, p)
        
        # OVER OTHER PARKING SPOTS REWARD        
        # over_other_parking_spots_reward = self.OVER_OTHER_PARKING_SPOT_REWARD * np.squeeze(info["is_over_others_parking_spot"])

        # COLLISION REWARD
        collision_reward = self.COLLISION_REWARD * np.squeeze(info["is_collision"])

         # REVERESE DRIVING REWARD
        reverse_reward = self.REVERSE_REWARD * np.squeeze(info["is_reverse"])

        # REACHING THE GOAL REWARD
        # reaching_goal_reward = self.REACHING_GOAL_REWARD *  np.squeeze(info["is_success"])

        reward = (distance_to_goal_reward + \
                  collision_reward + \
                  reverse_reward)  

                 # over_other_parking_spots_reward + \
                 # reverse_reward + \
                 # against_traffic_reward + \
                 # moving_reward +\
                 # reaching_goal_reward + \
                 # collision_reward)

        reward /= self.REWARD_SCALE
        #print(reward)
        return reward 


    def _reward(self, action):
        raise NotImplementedError

    def _is_success(self, achieved_goal, desired_goal):
        # DISTANCE TO GOAL
        distance_to_goal_reward = self.distance_2_goal_reward(achieved_goal, desired_goal)
        
        self.is_success = (distance_to_goal_reward > -self.SUCCESS_THRESHOLD)
        return self.is_success       
        

    def _is_terminal(self):
        """
            The episode is over if the ego vehicle crashed or the goal is reached.
        """
        # The episode cannot terminate unless all time steps are done. The reason for this is that HER + DDPG uses constant
        # length episodes. If you plan to use other algorithms, please uncomment this line
        #if info["is_collision"] or info["is_success"]:

        if self.crashed: # or self.vehicle.is_success:
            self.reset()
        return False # self.vehicle.crashed or self._is_success(obs['achieved_goal'], obs['desired_goal'])
    

    def on_collision(self, agent1, agent2, contact):
        self.crashed = True


    def render(self, mode='human'):
        """
            Render the environment.

            Create a viewer if none exists, and use it to render an image.
        :param mode: the rendering mode
        """
        pass

    def close(self):
        """
            Close the environment.

            Will close the environment viewer if it exists.
        """
        self.sim.stop()
