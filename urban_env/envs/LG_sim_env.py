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
from math import sin, cos
import random

from gym import GoalEnv, logger
from gym.spaces import Discrete, Box, Tuple
from gym.utils import colorize, seeding


import lgsvl
from lgsvl.utils import *

from gym import GoalEnv, spaces

from urban_env.envs.parking_env_2outs import ParkingEnv_2outs as ParkingEnv
from urban_env.vehicle.dynamics import Obstacle, Pedestrian


VELOCITY_EPSILON = 0.1

HAVAL_PARKING_LOT = {
                        "parking_angle": 0,
                        "parking_spots": 10,
#                        "map_offset": [-40, -4],
                        "aisle_width": 6.5, 
                        "width": 3,
                    }


class ObstacleLG(Obstacle):
    """
        A motionless obstacle at a given position.
    """

    def __init__(self, road, position, heading=0, config=None, color=None, **kwargs):
        super(ObstacleLG, self).__init__(
                                       road=road, 
                                       position=position, 
                                       heading=heading, 
                                       color=color, 
                                       config=config, 
                                       **kwargs)
        self.LGAgent = None

    def step(self, dt):
        
        if self.LGAgent is not None:
            self.position = [self.LGAgent.state.transform.position.x, self.LGAgent.state.transform.position.z]



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
            "RESTORE_COND": None,
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
            #"obstacle_type": "urban_env.envs.LG_sim_env.ObstacleLG",
            "DIFFICULTY_LEVELS": 4,
            "OBS_STACK_SIZE": 1,
            "vehicles_count": 'random',
            "goals_count": 'all',
            "pedestrian_count": 'random',
            "SIMULATION_FREQUENCY": 10,  # The frequency at which the system dynamics are simulated [Hz]
            "POLICY_FREQUENCY": 10,  # The frequency at which the agent can take actions [Hz]
            "velocity_range": 1.5*ParkingEnv.PARKING_MAX_VELOCITY,
            "MAX_VELOCITY": ParkingEnv.PARKING_MAX_VELOCITY,
            "closest_lane_dist_thresh": 500,
            "map": 'WideFlatMap',
            "map_offset": [0, 0, 0],
            },
        **{
            "PARKING_LOT_WIDTH": ParkingEnv.DEFAULT_PARKING_LOT_WIDTH,
            "PARKING_LOT_LENGTH": ParkingEnv.DEFAULT_PARKING_LOT_LENGTH,
            "parking_spots": 'random',  # Parking Spots per side            
            "parking_angle": 'random',  # Parking angle in deg 
            "x_position_range": ParkingEnv.DEFAULT_PARKING_LOT_WIDTH,
            "y_position_range": ParkingEnv.DEFAULT_PARKING_LOT_LENGTH,                      
          },
          **HAVAL_PARKING_LOT
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
        


        '''if self.sim.current_scene == self.config["map"]:
            self.sim.reset()
        else:'''
         
    def step(self, action): 
        obs, reward, done, info = super(LG_Sim_Env, self).step(action)
        velocity = np.sqrt(self.ego.state.velocity.x**2 + self.ego.state.velocity.z**2)
        allow_switch_gear = np.abs(velocity) < VELOCITY_EPSILON
        #reverse = False

        ##############################################
        #acceleration = action[0].item() * self.vehicle.config['max_acceleration']
        #steering = action[1].item() * self.vehicle.config['max_steer_angle']
        ###############################################

        throttle_brake = action[0].item()

        if velocity > self.config["MAX_VELOCITY"]:
            throttle_brake = min(throttle_brake, 1.0*(self.config["MAX_VELOCITY"] - velocity))
        elif velocity < -self.config["MAX_VELOCITY"]:
            throttle_brake = max(throttle_brake, 1.0*(self.config["MAX_VELOCITY"] - velocity))

        if throttle_brake < 0.0: # Fwd Braking or Reverse Throttle
            if velocity > VELOCITY_EPSILON:
                self.control.throttle = 0.0
                self.control.braking = np.abs(throttle_brake)
                self.vehicle.reverse = False
            elif velocity < -VELOCITY_EPSILON or self.control.reverse:
                self.control.throttle = np.abs(throttle_brake)
                self.control.braking = 0.0
                self.vehicle.reverse = True
            else: # self.control.reverse is False
                self.control.throttle = 0.0
                self.control.braking = 0.0
                self.vehicle.reverse = True

        else: # Fwd Throttle or Reverse Braking
            if velocity > VELOCITY_EPSILON or (not self.control.reverse): # FWD throttle
                self.control.braking = 0.0
                self.control.throttle = np.abs(throttle_brake)
                self.vehicle.reverse = False
            elif self.control.reverse: # reverse brake
                self.control.braking = np.abs(throttle_brake)
                self.control.throttle = 0.0
                self.vehicle.reverse = True
            else:# self.control.reverse is True
                self.control.throttle = 0.0
                self.control.braking = 0.0
                self.vehicle.reverse = False

        self.control.steering = action[1].item() * np.rad2deg(self.vehicle.config['max_steer_angle']) / 39.4
        self.control.reverse = self.vehicle.reverse
        self.ego.apply_control(self.control, True)
        self.sim.run(time_limit=1/self.config["POLICY_FREQUENCY"])

        print(" reverse ", self.vehicle.reverse,
              " velocity ", "{0:.2f}".format(velocity), 
              " steer ", "{0:.2f}".format(self.control.steering), 
              " throttle ", "{0:.2f}".format(self.control.throttle), 
              " braking ",  "{0:.2f}".format(self.control.braking), 
              " throttle_brake ", "{0:.2f}".format(throttle_brake))

        return obs, reward, done, info


    def _populate_scene(self):
        """
            Create some new random vehicles of a given type, and add them on the road.
        """

        for v in self.road.vehicles:
            if v.is_ego_vehicle:
                v.LGAgent = self._setup_agent(v, "jaguar2015xe",  lgsvl.AgentType.EGO)
                self.ego = v.LGAgent
            elif isinstance(v, Pedestrian):
                pedestrian = random.choice(["Bob", "Howard", "Johny", "Pamela", "Presley", "Red", "Robin", "Stephen", "Zoe"])
                v.LGAgent = self._setup_agent(v, pedestrian,  lgsvl.AgentType.PEDESTRIAN)
            elif v in self.road.virtual_vehicles:
                #self._setup_agent(v, "BoxTruck",  lgsvl.AgentType.NPC)
                pass
            else:
                v.LGAgent = self._setup_agent(v, "Sedan",  lgsvl.AgentType.NPC)

        #self.ego.on_collision(self.on_collision)


        
    def _setup_agent(self, v, agent_name="Jeep", agent_type=lgsvl.AgentType.NPC):
        state = lgsvl.AgentState()
        x = v.position[0] + self.config["map_offset"][0]
        y = 0.0
        z = v.position[1] + self.config["map_offset"][1]

        vx = v.velocity*cos(v.heading)
        vy = 0.0
        vz = v.velocity*sin(v.heading)

        state.transform.position = lgsvl.Vector(z, y, x)
        state.transform.rotation = lgsvl.Vector(0.0, np.rad2deg(v.heading) + self.config["map_offset"][2], 0.0)
        state.velocity = lgsvl.Vector(vz, vy, vx)
        state.angular_velocity = lgsvl.Vector(0.0, 0.0, 0.0)
        return self.sim.add_agent(agent_name, agent_type, state)

    def on_collision(self, agent1, agent2, contact):
        self.crashed = True

    def reset(self):
        obs = super(LG_Sim_Env, self).reset()
        if self.sim is not None:
            self.sim.reset()
        else:
            self.sim = lgsvl.Simulator(address="127.0.0.1", port=8181) 
            self.control = lgsvl.VehicleControl()
            print("self.sim.current_scene ", self.sim.current_scene)
            if self.sim.current_scene == self.config["map"]:
                self.sim.reset()
            else:
                self.sim.load(self.config["map"]) 
        self._populate_scene()
        return obs


    def close(self):
        """
            Close the environment.

            Will close the environment viewer if it exists.
        """
        self.sim.stop()
