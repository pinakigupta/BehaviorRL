######################################################################
#          Deep Reinforcement Learning for Autonomous Driving
#                  Created/Modified on: February 5, 2019
#                      Author: Munir Jojo-Verge, Pinaki Gupta
#######################################################################


from __future__ import division, print_function, absolute_import
import numpy as np
import pandas
from gym import GoalEnv
from gym.spaces import Dict, Discrete, Box, Tuple
import copy
import sys

from urban_env.envs.abstract import AbstractEnv
from urban_env.road.lane import StraightLane, LineType, AbstractLane
from urban_env.envs.graphics import EnvViewer
from urban_env.road.lane import StraightLane, LineType
from urban_env.road.road import Road, RoadNetwork
from urban_env.vehicle.dynamics import Vehicle, Obstacle
from urban_env.vehicle.control import MDPVehicle
from urban_env.vehicle.behavior import IDMVehicle
from urban_env.utils import *
from handle_model_files import is_predict_only
from urban_env.envdict import WHITE, RED

import pprint

VELOCITY_EPSILON = 1.0

vehicle_params = {
    'front_edge_to_center': 3.898,
    'back_edge_to_center': 0.853,
    'left_edge_to_center': 0.9655,
    'right_edge_to_center': 0.9655,

    'length': 4.751,
    'width': 1.931,
    'height': 1.655,
    'min_turn_radius': 4.63,
    'max_acceleration': 2.94,
    'max_deceleration': -6.0,
    'max_steer_angle': 0.58904875,
    'max_steer_angle_rate': 8.3733,
    'min_steer_angle_rate': 0,
    'steer_ratio': 16.00,
    'wheel_base': 2.95,
    'wheel_rolling_radius': 0.335,
    'max_abs_speed_when_stopped': 0.2
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
        **{
            "OVER_OTHER_PARKING_SPOT_REWARD": -10,
            "VELOCITY_REWARD": 2,
            "COLLISION_REWARD": -750,
            "TERM_REWARD": -400,
            "REVERSE_REWARD": -1,
            "GOAL_REWARD": 2000,
            "CURRICULAM_REWARD_THRESHOLD": 0.9,
            "SUCCESS_THRESHOLD": 0.001,
            "REWARD_WEIGHTS": np.array([15/100, 15/100, 1/100, 1/100, 2/100, 2/100]),
        },
        **{
            "LOAD_MODEL_FOLDER": "20191203-033319",
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
                "vehicles_count": 10,
                "goals_count": 10,
                "constraints_count": 5,
                           },
            "other_vehicles_type": "urban_env.vehicle.behavior.IDMVehicle",
            "duration": 100,
            "_predict_only": is_predict_only(),
            "screen_width": 1600,
            "screen_height": 900,
            "DIFFICULTY_LEVELS": 1,
            "OBS_STACK_SIZE": 1,
            "vehicles_count": 0,
            "goals_count": 1,
            "SIMULATION_FREQUENCY": 5,  # The frequency at which the system dynamics are simulated [Hz]
            "POLICY_FREQUENCY": 1,  # The frequency at which the agent can take actions [Hz]
            "x_position_range": DEFAULT_PARKING_LOT_WIDTH,
            "y_position_range": DEFAULT_PARKING_LOT_LENGTH,
            "velocity_range": 1.5*PARKING_MAX_VELOCITY,
            "MAX_VELOCITY": PARKING_MAX_VELOCITY,
            "closest_lane_dist_thresh": 500,
            },
        **{
            "PARKING_LOT_WIDTH": DEFAULT_PARKING_LOT_WIDTH,
            "PARKING_LOT_LENGTH": DEFAULT_PARKING_LOT_LENGTH,
            "parking_spots": 'random',  # Parking Spots per side            
            "parking_angle": 'random',  # Parking angle in deg           
          }
    }

    def __init__(self, config=DEFAULT_CONFIG):

        # ACTION SPACE:
        # Throttle: [0 to 1],
        # Brake   : [0 to 1]
        # steering: [-1 to 1],
        # reverse : [-1 to 1] => from -1 to 0 Reverse and from 0 to 1 Forward.
        #self.action_space = Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)

        super(ParkingEnv_2outs, self).__init__(config)
        if is_predict_only():
            self.set_curriculam(6)
            self.config["SUCCESS_THRESHOLD"] *= 2.0
        obs = self.reset()
        #self.REWARD_WEIGHTS = np.array(self.config["REWARD_WEIGHTS"])
        self.config["REWARD_SCALE"] = np.absolute(self.config["GOAL_REWARD"])
        EnvViewer.SCREEN_HEIGHT = self.config['screen_height']
        EnvViewer.SCREEN_WIDTH = self.config['screen_width']
        self.scene_complexity = self.config['DIFFICULTY_LEVELS']

    def step(self, action):
        self.steps += 1
        ##############################################
        acceleration = action[0].item() * vehicle_params['max_acceleration']
        steering = action[1].item() * vehicle_params['max_steer_angle']
        ###############################################


        # Forward action to the vehicle
        self.vehicle.control_action = (
                                        {
                                            "acceleration": acceleration,
                                            "steering": steering                                                    
                                        }
                                      )

        self.other_vehicle.control_action = (
                                                {
                                                    "acceleration":  acceleration,
                                                    "steering": steering                                                 
                                                }
                                            )

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
        self._build_parking()
        self._populate_parking()
        self.is_success = False
        self.vehicle.crashed = False
        self.define_spaces()
        self.episode_reward = 0
        obs = self.observation.observe()
        return obs
        #return self._observation()

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
            parking angle = 90, 75, 60, 45
        """
        # Defining parking spots 
        if self.config["parking_spots"] == 'random':
            self.parking_spots = self.np_random.randint(2, 15)
        else:
            self.parking_spots = self.config["parking_spots"]

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

        # Defining goals 
        if self.config["goals_count"] == 'all':
            self.goals_count = (self.parking_spots*2) -  self.vehicles_count
        else:
            self.goals_count = self.config["goals_count"]

        assert (self.vehicles_count < self.parking_spots*2)

        net = RoadNetwork()

        lt = (LineType.CONTINUOUS, LineType.CONTINUOUS)

        spots_offset = 0.0
        parking_angles = np.deg2rad([90, 75, 60, 45, 0])
        aisle_width = 20.0
        length = 8.0
        width = 4.0

        # Let's start by randomly choosing the parking angle
        angle = 0  # np.pi/3
        if self.config["parking_angle"] == 'random':
            angle = parking_angles[self.np_random.randint(len(parking_angles))]
        else:
            angle = np.deg2rad(self.config["parking_angle"])

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

        w = self.config["PARKING_LOT_WIDTH"]/2
        l = self.config["PARKING_LOT_LENGTH"]/2
        hidden = [LineType.NONE, LineType.NONE]
        net.add_lane("e", "f", StraightLane([w, -l], [w, l], width=0, line_types=hidden))
        net.add_lane("e", "f", StraightLane([-w, -l], [-w, l], width=0, line_types=hidden))
        net.add_lane("e", "f", StraightLane([-w, -l], [w, -l], width=0, line_types=hidden))
        net.add_lane("e", "f", StraightLane([-w, l], [w, l], width=0, line_types=hidden))

        #net.add_lane("g", "h", StraightLane([-w, l/2], [w, l/2]))


        self.road = Road(network=net,
                         np_random=self.np_random,
                         config=self.config)

    def _populate_parking(self):
        """
            Create some new random vehicles of a given type, and add them on the road.
        """
        self.border_lane_count = 4
        parking_spots_used = []
        '''
        lane = self.np_random.choice(self.road.network.lanes_list()[:-4])
        parking_spots_used.append(lane)
        obstacle =  Obstacle(
                                road=self.road,
                                position=lane.position(lane.length/2, 0), 
                                heading=lane.heading,
                                config={**self.config, **{"COLLISIONS_ENABLED": False}},
                                #color=WHITE
                            )
        self.road.goals.append(obstacle)
        self.road.vehicles.insert(0, obstacle)
        self.road.add_virtual_vehicle(obstacle)'''

        ##### ADDING EGO #####
        self.vehicle =  Vehicle(
                               road=self.road, 
                               position=[0, 0],
                               heading=2*np.pi*self.np_random.rand(),
                               velocity=0,
                               route_lane_index=None,
                               config=self.config
                               )
        self.vehicle.is_ego_vehicle = True
        self.road.vehicles.append(self.vehicle)
        ego_x0 = self.vehicle.position[0]

        self.other_vehicle =  Vehicle(
                               road=self.road, 
                               position=[5.0, 5.0],
                               heading=2*np.pi*self.np_random.rand(),
                               velocity=0,
                               route_lane_index=None,
                               config=self.config,
                               color=RED
                               )
        self.road.vehicles.append(self.other_vehicle)

        lane = self.np_random.choice(self.road.network.lanes_list()[:-self.border_lane_count])

        ##### ADDING OTHER VEHICLES #####
        for _ in range(self.vehicles_count):
            while lane in parking_spots_used:  # this loop should never be infinite since we assert that there should be more parking spots/lanes than vehicles
                # to-do: chceck for empty spots
                lane = self.np_random.choice(self.road.network.lanes_list()[:-self.border_lane_count])
            parking_spots_used.append(lane)

            # + self.np_random.randint(2) * np.pi
            self.road.vehicles.append(
                                      Obstacle(
                                               road=self.road,
                                               position=lane.position(lane.length/2, 0),
                                               heading=lane.heading,
                                               velocity=0,
                                               config=self.config,
                                              )
                                    )

        ##### ADDING OTHER GOALS #####
        for _ in range(self.goals_count):
            while lane in parking_spots_used:   
                lane = self.np_random.choice(self.road.network.lanes_list()[:-self.border_lane_count])
            parking_spots_used.append(lane)

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
                                           
        dyn_veh_count = 0 #np.random.randint(low=0, high=2*scene_complexity)
        lane_index = ("g", "h", 0)
        for i in range(dyn_veh_count):
            x0 = ego_x0+20+40*i + 10*self.np_random.randn()
            v =     IDMVehicle(self.road,
                               position=self.road.network.get_lane(lane_index)
                               .position(x0, 0),
                               heading=self.road.network.get_lane(lane_index).heading_at(x0),
                               velocity=max(0,10 + 2*self.np_random.randn()),
                               target_velocity=self.config["MAX_VELOCITY"]/2,
                               #target_lane_index=lane_index, 
                               #lane_index=lane_index,                             
                               enable_lane_change=False,
                               config=self.config
                               )
            self.road.add_vehicle(v)
                
        self._add_constraint_vehicles()


    def _distance_2_goal_reward(self, achieved_goal, desired_goal, p=2):
        min_distance_2_goal_reward = 1e6
        numoffeatures = len(self.config["observation"]["features"])
        #numfofobs = len(desired_goal)
        #numofvehicles = numfofobs//numoffeatures
        numofvehicles = len(self.observations[self.vehicle].closest_vehicles()["desired_goal"])
        desired_goal_list_debug_only = []
        distance_2_goal_reward_list_debug_only = []
        for i in range(numofvehicles):
            goal_err = achieved_goal - desired_goal[i*numoffeatures:(i+1)*numoffeatures]
            weighed_goal_err = np.multiply(np.abs(goal_err), self.config["REWARD_WEIGHTS"])
            distance_2_goal_reward = np.sum(weighed_goal_err**p)**(1/p)
            min_distance_2_goal_reward = min(min_distance_2_goal_reward, distance_2_goal_reward)
            desired_goal_list_debug_only.append(desired_goal[i*numoffeatures:(i+1)*numoffeatures])
            distance_2_goal_reward_list_debug_only.append(distance_2_goal_reward)
        return -min_distance_2_goal_reward

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

        # DISTANCE TO GOAL
        distance_to_goal_reward = self._distance_2_goal_reward(achieved_goal, desired_goal, p)
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
        velocity_reward = self.config["VELOCITY_REWARD"] * (self.vehicle.velocity - 0.5*self.PARKING_MAX_VELOCITY) / (self.PARKING_MAX_VELOCITY)
        continuous_reward = (distance_to_goal_reward + reverse_reward  )  # + \
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
        #elif(info["is_terminal"]):
        #    reward = self.config["TERM_REWARD"]
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
            "is_terminal": self._is_terminal()
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
        terminal = self.vehicle.crashed or \
                   self.is_success or \
                  (self.steps >= self.config["duration"]) or\
                   (self.vehicle.action_validity == False)
        # or self._is_success(obs['achieved_goal'], obs['desired_goal'])
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
            virtual_obstacle_ = Obstacle(
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
                
        '''
        lane_ids = [["a", "b" ],  ["b", "c"]]
        lane_set = []
        for lane_id in lane_ids:
            for goal in [self.road.goals[0]]:
                if list(goal.lane_index[0:2]) != lane_id:
                    lane_set.append(lane_id)
        

        print("lane_ids ",lane_ids, " lane_set ", lane_set )
        spot_idxs = [[self.config["parking_spots"]//2]]
        for lane_id in lane_set:
            for spot_idx in spot_idxs:
                lane_index = (*lane_id, *spot_idx)
                lane = self.road.network.get_lane(lane_index)
                x0 = lane.length/2
                position = lane.position(x0, 0)
                virtual_obstacle_ = Obstacle(
                                                    road=self.road,
                                                    position=position,
                                                    heading=lane.heading_at(x0)+np.pi/2,
                                                    velocity=0,
                                                    lane_index=lane_index,
                                                    target_lane_index=lane_index,                                            
                                                    config=self.config
                                            )
                #virtual_obstacle_.is_projection = True
                virtual_obstacle_.virtual = True                                       
                virtual_obstacle_.LENGTH = int(lane.width*self.config["parking_spots"])
                virtual_obstacle_.WIDTH = int(lane.length)
                #virtual_obstacle_.hidden = True
                self.road.add_vehicle(virtual_obstacle_)
                self.road.add_virtual_vehicle(virtual_obstacle_)'''


