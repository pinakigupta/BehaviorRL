######################################################################
#          Deep Reinforcement Learning for Autonomous Driving
#                  Created/Modified on: April 17, 2019
#                      Author:   Pinaki Gupta
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


from PythonAPI import lgsvl
from PythonAPI.lgsvl.utils import *

from gym import GoalEnv, spaces

from urban_env.envs.parking_env_2outs import ParkingEnv_2outs as ParkingEnv


VELOCITY_EPSILON = 0.1


class LG_Sim_Env(ParkingEnv):
    """       
        LG Simulator Openai Compliyant Environment Class
    """
    ACTIONS_HOLD_TIME = 1.0

    sim = None
    
    DEFAULT_CONFIG = {**ParkingEnv.DEFAULT_CONFIG,
        **{
            "OVER_OTHER_PARKING_SPOT_REWARD": -10,
            "VELOCITY_REWARD": 2,
            "COLLISION_REWARD": -750,
            "TERM_REWARD": -400,
            "REVERSE_REWARD": -1,
            "GOAL_REWARD": 2000,
            "CURRICULAM_REWARD_THRESHOLD": 0.9,
            "SUCCESS_THRESHOLD": 0.0015,
            "REWARD_WEIGHTS": np.array([15/100, 15/100, 1/100, 1/100, 2/100, 2/100]),
        },
        **{
            "LOAD_MODEL_FOLDER": "20191203-232528",
            "RESTORE_COND": "RESTORE", 
            "MODEL":             {
                                #    "use_lstm": True,
                                     "fcnet_hiddens": [256, 128, 128],
                                #     "fcnet_activation": "relu",
                                 }, 
        },
        **{
            "observation": {
                "type": "KinematicsGoal",
                "features": ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
                "relative_features": ['x', 'y'],
                "scale": 100,
                "obs_size": 10,
                "obs_count": 10,
                "goals_size": 10,
                "goals_count": 10,
                "constraints_count": 5,
                           },
            "other_vehicles_type": "urban_env.vehicle.behavior.IDMVehicle",
            "DIFFICULTY_LEVELS": 4,
            "OBS_STACK_SIZE": 1,
            "vehicles_count": 'random',
            "goals_count": 'all',
            "pedestrian_count": 0,
            "SIMULATION_FREQUENCY": 5,  # The frequency at which the system dynamics are simulated [Hz]
            "POLICY_FREQUENCY": 1,  # The frequency at which the agent can take actions [Hz]
            "x_position_range": ParkingEnv.DEFAULT_PARKING_LOT_WIDTH,
            "y_position_range": ParkingEnv.DEFAULT_PARKING_LOT_LENGTH,
            "velocity_range": 1.5*ParkingEnv.PARKING_MAX_VELOCITY,
            "MAX_VELOCITY": ParkingEnv.PARKING_MAX_VELOCITY,
            "closest_lane_dist_thresh": 500,
            "map": 'InterchangeDrive',
            },
        **{
            "PARKING_LOT_WIDTH": ParkingEnv.DEFAULT_PARKING_LOT_WIDTH,
            "PARKING_LOT_LENGTH": ParkingEnv.DEFAULT_PARKING_LOT_LENGTH,
            "parking_spots": 'random',  # Parking Spots per side            
            "parking_angle": 'random',  # Parking angle in deg           
          }
    }


    def __init__(self, config=None):
        if self.sim is not None:
            return

        #self.config = self.DEFAULT_CONFIG.copy()
        if config is None:
            config = self.DEFAULT_CONFIG
        else:
            config = {**self.DEFAULT_CONFIG, **config}

        super(LG_Sim_Env, self).__init__(config)

        #self.sim = lgsvl.Simulator(address=os.environ.get("SIMULATOR_HOST", "127.0.0.1"), port=8080) 
        
        self.sim = lgsvl.Simulator(address="127.0.0.1", port=8181) 

        self.control = lgsvl.VehicleControl()

        '''if self.sim.current_scene == self.config["map"]:
            self.sim.reset()
        else:'''
        self.sim.load(self.config["map"])  
        self.agents = {}     
        self._populate_scene()

        # Spaces        
        #self.define_spaces()        
                

        #self.crashed = False
        
        # self.action_space = spaces.Box(-1., 1., shape=(2,), dtype=np.float32)
        #self.REWARD_WEIGHTS = np.array(self.REWARD_WEIGHTS)
        

    def step(self, action): 
        obs, reward, done, info = super(LG_Sim_Env, self).step(action)
        print("stepping")
        return obs, reward, done, info

    def step1(self, action):                                
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

    '''def observe(self):                
        
        d = {            
            'x': self.ego.state.transform.position.x,
            'z': self.ego.state.transform.position.z,
            'vx': self.ego.state.velocity.x,
            'vz': self.ego.state.velocity.z,
            'cos_h': np.cos(np.deg2rad(-self.ego.state.transform.rotation.y)),
            'sin_h': np.sin(np.deg2rad(-self.ego.state.transform.rotation.y))
        }
        obs = np.ravel(pandas.DataFrame.from_records([d])[self.features])
        goal = np.ravel(pandas.DataFrame.from_records([self.road.goals])[self.features])
        obs = {
            "observation": obs / self.OBS_SCALE,
            "achieved_goal": obs / self.OBS_SCALE,
            "desired_goal": goal / self.OBS_SCALE
        }
        return obs'''

    def _populate_scene(self):
        """
            Create some new random vehicles of a given type, and add them on the road.
        """

        for v in self.road.vehicles:
            if v.is_ego_vehicle:
                self._setup_agent(v, "jaguar2015xe",  lgsvl.AgentType.EGO)
            elif v in self.road.virtual_vehicles:
                self._setup_agent(v, "BoxTruck",  lgsvl.AgentType.NPC)
            else:
                self._setup_agent(v, "Sedan",  lgsvl.AgentType.NPC)

        #self.ego.on_collision(self.on_collision)


        
    def _setup_agent(self, v, agent_name="Jeep", agent_type=lgsvl.AgentType.NPC):
        state = lgsvl.AgentState()
        state.transform.position = lgsvl.Vector(v.position[0], 0, v.position[1])
        state.transform.rotation.y = v.heading
        #state.velocity = v.velocity
        self.sim.add_agent(agent_name, agent_type, state)

    def on_collision(self, agent1, agent2, contact):
        self.crashed = True

    def reset(self):
        return super(LG_Sim_Env, self).reset()
        

    def close(self):
        """
            Close the environment.

            Will close the environment viewer if it exists.
        """
        self.sim.stop()
