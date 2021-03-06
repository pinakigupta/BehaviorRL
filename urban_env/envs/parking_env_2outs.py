######################################################################
#          Deep Reinforcement Learning for Autonomous Driving
#                  Created/Modified on: February 5, 2019
#                      Author:   Pinaki Gupta
#######################################################################


from __future__ import division, print_function, absolute_import
import numpy as np
from gym import GoalEnv
from gym.spaces import Dict, Discrete, Box, Tuple
import copy
import sys
from random import choice
from numpy import linalg as la
import os
import time

from urban_env.envs.abstract import AbstractEnv
from urban_env.road.lane import StraightLane, LineType, AbstractLane
from urban_env.envs.graphics import EnvViewer
from urban_env.road.lane import StraightLane, LineType
from urban_env.road.road import Road, RoadNetwork
from urban_env.vehicle.dynamics import Vehicle, Obstacle, Pedestrian
from urban_env.vehicle.control import MDPVehicle
from urban_env.vehicle.behavior import IDMVehicle
from urban_env.utils import *
from handle_model_files import is_predict_only
from urban_env.envdict import WHITE, RED, BLACK, GREY

import pprint

HAVAL_PARKING_LOT = {
                        "parking_angle": 90,
                        "parking_spots": 10,
                        "map_offset": [-4, -40 , 0],
                        "ego_initial_pose": 'random',
                        "summon_pose": None, #[4, 40, -90], # where to summon 
                        "aisle_width": 6.5, 
                        "width": 3,
                        "map": "ParkingLot",
                        "PARKING_LOT_WIDTH": 25,
                        "PARKING_LOT_LENGTH": 100,
                        "screen_height": 1200,
                    }


class ParkingEnv_2outs(AbstractEnv, GoalEnv):
    """
        A continuous control environment.

        It implements a reach-type task, where the agent observes their position and velocity and must
        control their acceleration and steering so as to reach a given goal.

    """
    PARKING_MAX_VELOCITY = 7.0  # m/s
    DEFAULT_PARKING_LOT_WIDTH = 90
    DEFAULT_PARKING_LOT_LENGTH = 70

    DEFAULT_CONFIG = {**AbstractEnv.DEFAULT_CONFIG,
        **{ # Reward related
            "OVER_OTHER_PARKING_SPOT_REWARD": -10,
            "VELOCITY_REWARD": 2,
            "COLLISION_REWARD": -750,
            "TERM_REWARD": -400,
            "REVERSE_REWARD": -1,
            "GOAL_REWARD": 2000,
            "CURRICULAM_REWARD_THRESHOLD": 0.9,
            "SUCCESS_THRESHOLD": 0.0015,
            "REWARD_WEIGHTS": np.array([15/100, 15/100, 1/100, 1/100, 2/100, 2/100]),
            "ACTION_CHANGE_REWARD": 0.0,
        },
        **{ # RL Policy model related
            "LOAD_MODEL_FOLDER": "20191216-141903",
            "RESTORE_COND": "RESTORE", 
            "MODEL":             {
                                #    "use_lstm": True,
                                     "fcnet_hiddens": [256, 128, 128],
                                #     "fcnet_activation": "relu",
                                 }, 
            "retrieved_agent_policy": 0,
        },
        **{ # observation space related
            "observation": {
                "type": "KinematicsGoal",
                "features":  ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
                "constraint_features":  ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
                "pedestrian_features": ['x', 'y', 'vx', 'vy'],
                "relative_features": ['x', 'y'],
                "scale": 100,
                "obs_size": 10,
                "obs_count": 10,
                "goals_size": 10,
                "goals_count": 10,
                "pedestrians_size": 0,
                "pedestrians_count": 0,
                "constraints_count": 5,
                           },
            # general config
            "obstacle_type": "urban_env.vehicle.dynamics.Obstacle",
            "duration": 100,
            "_predict_only": is_predict_only(),
            "screen_width": 800,
            "screen_height": 800,
            "DIFFICULTY_LEVELS": 6,
            "OBS_STACK_SIZE": 1,
            "vehicles_count": 'random',
            "goals_count": 'all',
            "pedestrian_count": 0,
            "constraints_count": 4,
            "x_position_range": DEFAULT_PARKING_LOT_WIDTH,
            "y_position_range": DEFAULT_PARKING_LOT_LENGTH,
            "velocity_range": 1.5*PARKING_MAX_VELOCITY,
            "MAX_VELOCITY": PARKING_MAX_VELOCITY,
            "closest_lane_dist_thresh": 500,
            },
        **{  # Frequency related
            "SIMULATION_FREQUENCY": 10,  # The frequency at which the system dynamics are simulated [Hz],
            "PREDICTION_SIMULATION_FREQUENCY": 10,  # The frequency at which the system dynamics are predicted [Hz],
            "POLICY_FREQUENCY": 2,  # The frequency at which the agent can take actions [Hz]
            "TRAJECTORY_FREQUENCY": 0.5, # The frequency at which the agent trajectory is generated, mainly for visualization
            "TRAJECTORY_HORIZON": 10,
          },
        **{ # Parking lot config related
            "PARKING_LOT_WIDTH": 'random',
            "PARKING_LOT_LENGTH": 'random',
            "parking_spots": 'random',  # Parking Spots per side            
            "parking_angle": 'random',  # Parking angle in deg
            "aisle_width": 'random',
            "width": 'random',
            "length": 'random',
            "ego_initial_pose": 'random',
            "summon_pose": None,
          },
          #**HAVAL_PARKING_LOT
    }

    def __init__(self, config=None):

        if config is None:
            config = self.DEFAULT_CONFIG
        else:
            config = {**self.DEFAULT_CONFIG, **config}

        # ACTION SPACE:
        # Throttle: [0 to 1],
        # Brake   : [0 to 1]
        # steering: [-1 to 1],
        # reverse : [-1 to 1] => from -1 to 0 Reverse and from 0 to 1 Forward.
        #self.action_space = Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        
        super(ParkingEnv_2outs, self).__init__(config)
        if is_predict_only():
            self.set_curriculam(15)
            self.config["SUCCESS_THRESHOLD"] *= 2.0
        #self.REWARD_WEIGHTS = np.array(self.config["REWARD_WEIGHTS"])
        self.config["REWARD_SCALE"] = np.absolute(self.config["GOAL_REWARD"])
        #self.config["closest_lane_dist_thresh"] = self.config["PARKING_LOT_WIDTH]
        EnvViewer.SCREEN_HEIGHT = self.config['screen_height']
        EnvViewer.SCREEN_WIDTH = self.config['screen_width']
        self.scene_complexity = self.config['DIFFICULTY_LEVELS']
        self.reset()
        self.prev_time = time.time()

    def step(self, action):
        from urban_env.utils import print_execution_time
        if self.prev_time is not None:
            if not self.intent_pred and is_predict_only():
                current_wall_time = print_execution_time(self.prev_time, " step " + str(self.steps) + " : " )
                self.prev_time = current_wall_time
        self.steps += 1
        ##############################################
        acceleration = action[0].item() * self.vehicle.config['max_acceleration']
        steering = action[1].item() * self.vehicle.config['max_steer_angle']
        ###############################################


        # Forward action to the vehicle
        self.vehicle.control_action = (
                                        {
                                            "acceleration": acceleration,
                                            "steering": steering                                                    
                                        }
                                      )

        '''self.other_vehicle.control_action = (
                                                {
                                                    "acceleration": acceleration,
                                                    "steering": steering                                                 
                                                }
                                            )'''

        # print("prev_act, curr_act, accel, steer, speed:", self.previous_action, action, acceleration, steering, self.vehicle.velocity)
        #self._simulate()

        #obs = self._observation()
        obs, reward, done, info = super(ParkingEnv_2outs, self).step(self.vehicle.control_action)

        #terminal = self._is_terminal()
        #self.print_obs_space(ref_vehicle=self.vehicle, obs_type="observation")
        #self.print_obs_space(ref_vehicle=self.vehicle, obs_type="desired_goal")
        #self.print_obs_space(ref_vehicle=self.vehicle, obs_type="constraint")
        
        return obs, reward, done, info

    def reset(self):
        self.steps = 0
        self.is_success = False
        self._populate_parking()
        return super(ParkingEnv_2outs, self).reset()

    def define_spaces(self):
        super(ParkingEnv_2outs, self).define_spaces()
        self.action_space = Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)

    def is_over_others_parking_spot(self, position):
        over_others_parking_spots = False
        for _from, to_dict in self.road.network.graph.items():
            for _to, lanes in to_dict.items():
                for _id, lane in enumerate(lanes):
                    if lane not in [goal.lane for goal in self.road.goals]:
                        goal_y_position = [goal.position[1] for goal in self.road.goals]
                        lane_x = lane.position(lane.length/2, 0)[0]
                        lane_y = lane.position(lane.length/2, 0)[1]
                        goal_x_offset_min = min([abs(goal.position[0]-lane_x) for goal in self.road.goals])
                        if lane_y in goal_y_position and goal_x_offset_min < 2.5*lane.width:
                            continue
                        over_others_parking_spots = lane.on_lane(position)
                    if (over_others_parking_spots):
                        return True
        return False

    def rot(self, point, angle):
        assert len(point) == 2
        x = point[0]
        y = point[1]
        cos_ang = np.cos(angle)
        sin_ang = np.sin(angle)
        x_rot = x*cos_ang - y*sin_ang
        y_rot = x*sin_ang + y*cos_ang
        return x_rot, y_rot

    def _build_parking(self):
        """
            Create a road composed of straight adjacent lanes.
            We will have 4 parking configurations based on "parking angle":
            https://www.webpages.uidaho.edu/niatt_labmanual/chapters/parkinglotdesign/theoryandconcepts/parkingstalllayoutconsiderations.htm
        """
        if self.config["PARKING_LOT_WIDTH"] == 'random':
            self.PARKING_LOT_WIDTH =  self.np_random.uniform(low=0.7*self.DEFAULT_PARKING_LOT_WIDTH, 
                                                             high=1.2*self.DEFAULT_PARKING_LOT_WIDTH)
        elif self.config["PARKING_LOT_WIDTH"] == 'default':
            self.PARKING_LOT_WIDTH =  self.DEFAULT_PARKING_LOT_WIDTH                                               
        else:
            self.PARKING_LOT_WIDTH = self.config["PARKING_LOT_WIDTH"]

        if self.config["PARKING_LOT_LENGTH"] == 'random':
            self.PARKING_LOT_LENGTH = self.np_random.uniform(low=0.7*self.DEFAULT_PARKING_LOT_LENGTH, 
                                                             high=1.2*self.DEFAULT_PARKING_LOT_LENGTH)
        elif self.config["PARKING_LOT_LENGTH"] == 'default':
            self.PARKING_LOT_LENGTH =  self.DEFAULT_PARKING_LOT_LENGTH                                                       
        else:
            self.PARKING_LOT_LENGTH = self.config["PARKING_LOT_LENGTH"]                                                

        # Defining parking spots 
        if self.config["parking_spots"] == 'random':
            self.parking_spots = self.np_random.randint(low=2, high=10)
        else:
            self.parking_spots = self.config["parking_spots"]

        if self.config["aisle_width"] == 'random':
            self.aisle_width = self.np_random.uniform(low=3, high=7)
        else:
            self.aisle_width = self.config["aisle_width"]

        if self.config["width"] == 'random':
            self.park_width = self.np_random.uniform(low=3, high=4)
        else:
            self.park_width = self.config["width"]

        if self.config["length"] == 'random':
            self.park_length = self.np_random.uniform(low=7, high=9)
        else:
            self.park_length = self.config["length"]  

        if self.config["ego_initial_pose"] == 'random':
            self.ego_initial_pose = []
            self.ego_initial_pose.append(self.np_random.uniform(low=-0.4*self.PARKING_LOT_WIDTH , 
                                                                high=0.4*self.PARKING_LOT_WIDTH ))
            self.ego_initial_pose.append(self.np_random.uniform(low=-0.4*self.PARKING_LOT_LENGTH, 
                                                                high=0.4*self.PARKING_LOT_LENGTH))
            self.ego_initial_pose.append(np.deg2rad(self.np_random.uniform(low=-90, high=90)))
        else:
            self.ego_initial_pose = self.config["ego_initial_pose"]
            self.ego_initial_pose[2] = np.deg2rad(self.ego_initial_pose[2])

        
        # Let's start by randomly choosing the parking angle
        #parking_angles = np.deg2rad([90, 75, 60, 45, 0])
        if self.config["parking_angle"] == 'random':
            #self.park_angle = parking_angles[self.np_random.randint(len(parking_angles))]
            self.park_angle =  self.np_random.uniform(low=0, high=90)
        else:
            self.park_angle = np.deg2rad(self.config["parking_angle"]) 


        # Defining pedestrian count
        if self.config["pedestrian_count"] == 'random':
            self.pedestrian_count = self.np_random.randint(0, 10)
        else:
            self.pedestrian_count =  self.config["pedestrian_count"]

        # Defining parked vehicles 
        if self.config["vehicles_count"] == 'random':
            low = 0
            high = self.parking_spots*self.scene_complexity//10
            if(low == high):
                self.vehicles_count = low
            else:
                self.vehicles_count = self.np_random.randint(low=low, high=high) 
            self.vehicles_count = min(self.vehicles_count, (self.parking_spots*2) - 1)
        elif self.config["vehicles_count"] == 'all':
            self.vehicles_count = (self.parking_spots*2) - 1
        elif self.config["vehicles_count"] == 'none':
            self.vehicles_count = 0
        elif self.config["vehicles_count"] == None:
            self.vehicles_count = 0
        else:
            self.vehicles_count = self.config["vehicles_count"]

        self.summon_pose = self.config["summon_pose"]
        self.constraints_count = self.config["constraints_count"]
        self.border_lane_count = self.constraints_count

        # Defining goals 
        if self.config["goals_count"] == 'all':
            self.goals_count = (self.parking_spots*2) -  self.vehicles_count
        else:
            self.goals_count = self.config["goals_count"]

        assert (self.vehicles_count < self.parking_spots*2)

        net = RoadNetwork()

        lt = (LineType.CONTINUOUS, LineType.CONTINUOUS)

        spots_offset = 0.0
        #aisle_width = self.np_random.randint(15, 20)
        aisle_width = self.aisle_width
        length = self.park_length
        width = self.park_width
        angle = self.park_angle 

        # Let's now build the parking lot
        for k in range(self.parking_spots):
            x1 = (k - self.parking_spots // 2) * \
                (width + spots_offset) - width / 2
            y1 = aisle_width/2
            x2 = x1
            y2 = y1 + length

            x3 = x1
            y3 = -y1
            x4 = x3
            y4 = -y2

            x1, y1 = self.rot((x1, y1), angle)
            x2, y2 = self.rot((x2, y2), angle)

            x3, y3 = self.rot((x3, y3), angle)
            x4, y4 = self.rot((x4, y4), angle)

            net.add_lane("a", "b", StraightLane(
                [x1, y1], [x2, y2], width=width, line_types=lt))
            net.add_lane("b", "c", StraightLane(
                [x3, y3], [x4, y4], width=width, line_types=lt))

        w = self.PARKING_LOT_WIDTH/2
        l = self.PARKING_LOT_LENGTH/2
        hidden = [LineType.NONE, LineType.NONE]
        borders = [([w, -l], [w, l]), ([-w, -l], [-w, l]), ([-w, -l], [w, -l]), ([-w, l], [w, l])]

        for i in range(self.border_lane_count):
            border = borders[i]
            net.add_lane("e", "f", StraightLane(*border, width=0, line_types=hidden))

        #net.add_lane("g", "h", StraightLane([-w, l/2], [w, l/2]))


        self.road = Road(network=net,
                         np_random=self.np_random,
                         config=self.config)

    def _spawn_summon_pose(self):
        self.summon = None
        if self.summon_pose is not None:
            self.summon =  Obstacle(
                                        road=self.road,
                                        position=[self.summon_pose[0], self.summon_pose[1]], 
                                        heading=np.deg2rad(self.summon_pose[2]),
                                        config={**self.config, **{"COLLISIONS_ENABLED": False}},
                                        color=RED
                                   )

    def _spawn_pedestrians(self):
        ###### ADDING PEDESTRIANS ###########
        for _ in range(self.pedestrian_count):
            rand_x = self.np_random.randint(low=10, high=self.PARKING_LOT_WIDTH//3)
            rand_y = self.np_random.randint(low=10, high=self.PARKING_LOT_LENGTH//3)
            self.Ped =  Pedestrian(
                                road=self.road, 
                                position=[self.np_random.choice([-rand_x, rand_x]), 
                                          self.np_random.choice([-rand_y, rand_y])],
                                route_lane_index=None,
                                heading=2*np.pi*self.np_random.rand(),
                                velocity=self.np_random.rand(),
                                config=self.config,
                                color=BLACK
                                )
            self.road.vehicles.append(self.Ped)
            self.road.pedestrians.append(self.Ped)


    def _spawn_EGO(self):
        ##### ADDING EGO #####
        all_lanes = self.road.network.lanes_list()
        '''for lane in all_lanes:
           if lane.distance(self.ego_initial_pose[0:1])<2.5:
                self.ego_initial_pose[0] = 0.0
                self.ego_initial_pose[0] = 0.0'''


        self.vehicle =  Vehicle(
                               road=self.road, 
                               position=[self.ego_initial_pose[0], self.ego_initial_pose[1]],
                               heading=self.ego_initial_pose[2], #2*np.pi*self.np_random.rand(),
                               velocity=0,
                               route_lane_index=None,
                               config=self.config
                               )
        self.vehicle.is_ego_vehicle = True
        self.road.vehicles.append(self.vehicle)

    def _spawn_vehicles(self):
        lane = self.np_random.choice(self.road.network.lanes_list()[:-self.border_lane_count])

        ##### ADDING OTHER VEHICLES #####
        GenericObstacle = class_from_path(self.config["obstacle_type"])
        for _ in range(self.vehicles_count):
            while lane in self.parking_spots_used:  # this loop should never be infinite since we assert that there should be more parking spots/lanes than vehicles
                # to-do: chceck for empty spots
                lane = self.np_random.choice(self.road.network.lanes_list()[:-self.border_lane_count])
            self.parking_spots_used.append(lane)

            self.road.vehicles.append(
                                      GenericObstacle(
                                                        road=self.road,
                                                        position=lane.position(lane.length/2, 0),
                                                        heading=lane.heading,
                                                        velocity=0,
                                                        config=self.config,
                                                     )
                                     )



    def _add_goals(self):
        ##### ADDING OTHER GOALS #####
        lane = self.np_random.choice(self.road.network.lanes_list()[:-self.border_lane_count])
        existing_goals_count = len(self.road.goals)
        for _ in range(self.goals_count - existing_goals_count):
            while lane in self.parking_spots_used:   
                lane = self.np_random.choice(self.road.network.lanes_list()[:-self.border_lane_count])
            self.parking_spots_used.append(lane)

            obstacle =  Obstacle(
                                road=self.road,
                                position=lane.position(lane.length/2, 0), 
                                heading=lane.heading,
                                config={**self.config, **{"COLLISIONS_ENABLED": False}},
                                #color=WHITE
                                )
            self.road.goals.append(obstacle)
            self.road.vehicles.insert(0, obstacle)
            self.road.add_virtual_vehicle(obstacle)

    def _spawn_cleanup(self):
        for v in self.road.vehicles:
            if v is self.vehicle:
                continue
            v.check_collision(self.vehicle)
            if not self.vehicle.crashed:
                continue
            if v in self.road.goals:
                continue
            elif v in self.road.virtual_vehicles: # reset ego position to gurantee there is no collision
                self.vehicle.position[0] = 0.0
                self.vehicle.position[1] = 0.0
            else: # remove the object
                self.road.vehicles.remove(v)
            self.vehicle.crashed = v.crashed = False      

        if self.config["goals_count"] == 'all':
            empty_spots_to_be_filled_with_goals = len(self.road.vehicles) + 1 < (self.parking_spots*2)
            if empty_spots_to_be_filled_with_goals > 0 : 
                self.goals_count = (self.parking_spots*2) -  self.vehicles_count
                self._add_goals()


            



    def _populate_parking(self):
        """
            Create some new random vehicles of a given type, and add them on the road.
        """

        self._build_parking()
        self.parking_spots_used = []
        self._spawn_summon_pose()
        self._spawn_pedestrians()
        self._spawn_EGO()
        self._spawn_vehicles()
        self._add_constraint_vehicles()
        self._spawn_cleanup()
        self._add_goals()

    def distance_2_goal_reward(self, achieved_goal, desired_goal, p=2):
        goal_err = achieved_goal - desired_goal
        weighed_goal_err = np.multiply(np.abs(goal_err), self.config["REWARD_WEIGHTS"])
        return np.sum(weighed_goal_err**p)**(1/p)

    def _distance_2_goal_reward(self, achieved_goal, desired_goal):
        min_distance_2_goal_reward = 1e6
        numoffeatures = len(self.config["observation"]["features"])
        numofvehicles = len(self.observations[self.vehicle].closest_vehicles()["desired_goal"])
        desired_goal_list_debug_only = []
        distance_2_goal_reward_list_debug_only = []
        for i in range(numofvehicles):
            goal_reward = self.distance_2_goal_reward(achieved_goal, desired_goal[i*numoffeatures:(i+1)*numoffeatures])
            min_distance_2_goal_reward = min(min_distance_2_goal_reward, goal_reward)
            desired_goal_list_debug_only.append(desired_goal[i*numoffeatures:(i+1)*numoffeatures])
            distance_2_goal_reward_list_debug_only.append(goal_reward)
        return -min_distance_2_goal_reward

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
            Proximity to the goal is rewarded

            We use a weighted p-norm
        :param achieved_goal: the goal that was achieved
        :param desired_goal: the goal that was desired
        :param info: any supplementary information
        :param p: the Lp^p norm used in the reward. Use p<1 to have high kurtosis for rewards in [0, 1]
        :return: the corresponding reward
        """

        # DISTANCE TO GOAL
        distance_to_goal_reward = self._distance_2_goal_reward(achieved_goal, desired_goal)
        #distance_to_goal_reward = min(-0.2, distance_to_goal_reward)
        
        #print("distance_to_goal_reward ", distance_to_goal_reward)

        # OVER OTHER PARKING SPOTS REWARD
        over_other_parking_spots_reward = self.config["OVER_OTHER_PARKING_SPOT_REWARD"] * \
            np.squeeze(info["is_over_others_parking_spot"])
        
        #print("over_other_parking_spots_reward ", over_other_parking_spots_reward)

        # COLLISION REWARD
        collision_reward = self.config["COLLISION_REWARD"] * np.squeeze(info["is_collision"])

        # REVERESE DRIVING REWARD
        reverse_reward = self.config["REVERSE_REWARD"] * np.squeeze(info["is_reverse"])
        comfort_reward =  self.config["REVERSE_REWARD"] * abs(self.vehicle.jerk)
        #print("jerk ", self.vehicle.jerk, " comfort_reward ", comfort_reward )
        velocity_reward = self.config["VELOCITY_REWARD"] * (self.vehicle.velocity - 0.5*self.PARKING_MAX_VELOCITY) / (self.PARKING_MAX_VELOCITY)
        continuous_reward = (distance_to_goal_reward + reverse_reward + comfort_reward )  # + \
        # over_other_parking_spots_reward)
        # reverse_reward + \
        # against_traffic_reward + \
        # collision_reward)
        sys.stdout.flush()
        #print("distance_to_goal_reward ", distance_to_goal_reward, " reverse_reward ", reverse_reward, \
        #    " velocity_reward ", velocity_reward, " velocity ", self.vehicle.velocity)
        goal_reward = self.config["GOAL_REWARD"]
        if self.vehicle.crashed:
            reward = collision_reward + min(0.0, continuous_reward)
        elif self.is_success:
            reward = goal_reward + continuous_reward
        else:
            reward = continuous_reward

        reward /= self.config["REWARD_SCALE"]
        #print("info", info)
        return reward

    def _reward(self, action):
        reward = self.compute_reward(self.obs['achieved_goal'], self.obs['desired_goal'], self._info())
        return reward


    def _info(self):
        info = {
            "is_success": int(self._is_success(self.obs['achieved_goal'], self.obs['desired_goal'])),
            "is_collision": int(self.vehicle.crashed),
            "is_over_others_parking_spot": int(self.is_over_others_parking_spot(self.vehicle.position)),
            "is_reverse": int(self.vehicle.velocity < 0),
            #"is_terminal": self._is_terminal()
        }
        return info

    def _is_success(self, achieved_goal, desired_goal):
        # DISTANCE TO GOAL
        distance_to_goal_reward = self._distance_2_goal_reward(
            achieved_goal, desired_goal)

        #print("desired_goal ", desired_goal)
        #print("achieved_goal ", achieved_goal)
        #print("distance_to_goal_reward ", distance_to_goal_reward)
        self.is_success = (distance_to_goal_reward > -self.config["SUCCESS_THRESHOLD"])
        return self.is_success

    def _is_terminal(self):
        """
            The episode is over if the ego vehicle crashed or the goal is reached.
            NOTE: If we use HER, we can NOT terminate the episode because they are
            supposed to have FIXED length. For this reason, we will just RESET and
            keep going.
        """
        if self.summon is not None and self.is_success and False:
            for goal in self.road.goals:
                self.road.vehicles.remove(goal)
                self.road.virtual_vehicles.remove(goal)
            self.road.goals = [self.summon]
            self.road.vehicles.insert(0, self.summon)
            self.road.add_virtual_vehicle(self.summon)
            self.steps = 0
            self.summon = None
            terminal = self.vehicle.crashed or \
                    (self.steps >= self.config["duration"])         
        else:
            terminal = self.vehicle.crashed or \
                    self.is_success or \
                    (self.steps >= self.config["duration"]) or\
                    (self.vehicle.action_validity == False)
        return terminal


    def print_obs_space(self, ref_vehicle, obs_type="observation"):
        if not ref_vehicle:
            return
        print("-------------- start  ", obs_type, ref_vehicle.Id(), "  ----------------------")
        print("obs space, step ", self.steps)
        if ref_vehicle.control_action is not None:
            print("reference accel = ", 
                    ref_vehicle.control_action['acceleration'],
                    " steering = ", ref_vehicle.control_action['steering'])
        
        #sys.stdout.flush()
        pp = pprint.PrettyPrinter(indent=4)
        numoffeatures = len(self.config["observation"]["features"])
        #numfofobs = len(self.obs[obs_type])
        #numofvehicles = numfofobs//numoffeatures
        modified_obs = self.observations[ref_vehicle].observe()[obs_type]
        close_vehicles = self.observations[ref_vehicle].closest_vehicles()[obs_type]    
        numofvehicles = len(close_vehicles)                                                  
        close_vehicle_ids = []
        for v in close_vehicles:
            close_vehicle_ids.append(int(v.Id()))
        numofclosevehicles = len(close_vehicle_ids)
        close_vehicle_ids.extend([-1]*(numofvehicles-len(close_vehicle_ids)))
        Idx = 0
        obs_Idx = 0
        while True:
            temp = copy.deepcopy(modified_obs)
            del(modified_obs)
            modified_obs = np.insert(temp, obs_Idx, close_vehicle_ids[Idx])
            del(temp)
            Idx += 1
            obs_Idx += numoffeatures+1
            if Idx >= len(close_vehicle_ids):
                break

        np.set_printoptions(precision=3, suppress=True)
        obs_format = pp.pformat(np.round(np.reshape(modified_obs[0:numofvehicles*(numoffeatures+1)], \
            (numofvehicles, numoffeatures+1 )), 3))
        obs_format = obs_format.rstrip("\n")
        print(obs_format)
        print("\n\n\n")


    def _add_constraint_vehicles(self):
        for i in range(self.border_lane_count):
            lane_index = ("e", "f", i)
            lane = self.road.network.get_lane(lane_index)
            x0 = lane.length/2
            position = lane.position(x0, 0)
            virtual_obstacle_ =     Obstacle(
                                            road=self.road,
                                            position=position,
                                            heading=lane.heading_at(x0),
                                            velocity=0,
                                            lane_index=lane_index,
                                            target_lane_index=lane_index,                                            
                                            config=self.config
                                            )
            virtual_obstacle_.virtual = True                                       
            virtual_obstacle_.LENGTH = lane.length
            self.road.add_vehicle(virtual_obstacle_)
            self.road.add_virtual_vehicle(virtual_obstacle_)
                

