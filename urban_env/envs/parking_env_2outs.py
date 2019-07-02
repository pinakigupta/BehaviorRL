######################################################################
#          Deep Reinforcement Learning for Autonomous Driving
#                  Created/Modified on: February 5, 2019
#                      Author: Munir Jojo-Verge
#######################################################################


from __future__ import division, print_function, absolute_import
import numpy as np
import pandas
from gym import GoalEnv
from gym.spaces import Dict, Discrete, Box, Tuple

from urban_env.envs.abstract import AbstractEnv
from urban_env.road.lane import StraightLane, LineType, AbstractLane
from urban_env.envs.graphics import EnvViewer
from urban_env.road.lane import StraightLane, LineType
from urban_env.road.road import Road, RoadNetwork
from urban_env.vehicle.dynamics import Vehicle, Obstacle
from urban_env.vehicle.control import MDPVehicle
from urban_env.utils import *

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

        Credits to Munir Jojo-Verge for the idea and initial implementation.
    """

    COLLISION_REWARD     = -1.0            
    OVER_OTHER_PARKING_SPOT_REWARD = -0.9
    REVERSE_REWARD = -0.2    

    PARKING_MAX_VELOCITY = 7.0 # m/s

    OBS_SCALE = 100
    REWARD_SCALE = np.absolute(COLLISION_REWARD)

    REWARD_WEIGHTS = [7/100, 7/100, 1/100, 1/100, 9/10, 9/10]
    SUCCESS_THRESHOLD = 0.34

    
    DEFAULT_CONFIG = {        
        "observation": {
            "type": "KinematicsGoal",
            "features": ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
            "scale": 100,
            "observation_near_ego": 0,
            "normalize": False
        },
        "other_vehicles_type": "urban_AD_env.vehicle.behavior.IDMVehicle",
        "centering_position": [0.5, 0.5],
        "parking_spots": 15, #'random', # Parking Spots Per side
        "vehicles_count": 'random', # Total number of cars in the parking (apart from Ego)
        "screen_width": 600 * 2,
        "screen_height": 300 * 2 
    }    

    def __init__(self):                
        super(ParkingEnv_2outs, self).__init__()
        self.observation_config = self.config['observation'].copy()
        obs = self.reset()
        self.observation_space = Dict(dict(
            desired_goal  = Box(-np.inf, np.inf, shape=obs["desired_goal"].shape, dtype=np.float32),
            achieved_goal = Box(-np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype=np.float32),
            observation   = Box(-np.inf, np.inf, shape=obs["observation"].shape, dtype=np.float32),
        ))
        self.DEFAULT_CONFIG["vehicles_count"]= np.random.randint(low=0,high=10)
        self._max_episode_steps = 50

        # ACTION SPACE: 
        ### Throttle: [0 to 1], 
        ### Brake   : [0 to 1] 
        ### steering: [-1 to 1],
        ### reverse : [-1 to 1] => from -1 to 0 Reverse and from 0 to 1 Forward. 
        self.action_space = Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)        

        self.REWARD_WEIGHTS = np.array(self.REWARD_WEIGHTS)
        
        
        EnvViewer.SCREEN_HEIGHT = self.config['screen_height']
        EnvViewer.SCREEN_WIDTH = self.config['screen_width']     

    def normalize(self, obs):
        """
            Normalize the observation values.
            For now, assume that the road is straight along the x axis.
        :param Dataframe obs: observation data
        """
        side_lanes = self.road.network.all_side_lanes(self.vehicle.lane_index)
        x_position_range = 5.0 * MDPVehicle.SPEED_MAX
        y_position_range = AbstractLane.DEFAULT_WIDTH * len(side_lanes)
        velocity_range = 2*MDPVehicle.SPEED_MAX
        obs['x'] = remap(obs['x'], [-x_position_range, x_position_range], [-1, 1])
        obs['y'] = remap(obs['y'], [-y_position_range, y_position_range], [-1, 1])
        obs['vx'] = remap(obs['vx'], [-velocity_range, velocity_range], [-1, 1])
        obs['vy'] = remap(obs['vy'], [-velocity_range, velocity_range], [-1, 1])
        return obs


    def _observation(self):
        ##### ADDING EGO #####
        obs = pandas.DataFrame.from_records([self.vehicle.to_dict()])[self.observation_config['features']]
        ego_obs = np.ravel(obs.copy())

        #### ADD THE GOAL ###
        #obs = obs.append(pandas.DataFrame.from_records([self.goal.to_dict()])[self.observation_config['features']])

        ##### ADDING NEARBY (TO EGO) TRAFFIC #####
        close_vehicles = self.road.closest_vehicles_to(self.vehicle, self.observation_config['observation_near_ego'])
        if close_vehicles:
            obs = obs.append(pandas.DataFrame.from_records(
                [v.to_dict(self.vehicle)
                 for v in close_vehicles])[self.observation_config['features']],
                           ignore_index=True)

        
        # Fill missing rows
        needed = self.observation_config['observation_near_ego'] + 1
        missing = needed - obs.shape[0]
        if obs.shape[0] < (needed):
            rows = -np.ones((missing, len(self.observation_config['features'])))
            obs = obs.append(pandas.DataFrame(data=rows, columns=self.observation_config['features']), ignore_index=True)

        

        # Normalize
        obs = self.normalize(obs)

        # Reorder
        obs = obs[self.observation_config['features']]
        
        # Flatten
        obs = np.ravel(obs)
        
        # Goal
        goal = np.ravel(pandas.DataFrame.from_records([self.goal.to_dict()])[self.observation_config['features']])

        # Arrange it as required by Openai GoalEnv
        obs = {
            "observation": obs / self.OBS_SCALE,
            "achieved_goal": ego_obs / self.OBS_SCALE,
            "desired_goal": goal / self.OBS_SCALE
        }
        return obs

    def step(self, action):                
        
        ##############################################           
        acceleration = action[0].item() * vehicle_params['max_acceleration']
        steering = action[1].item() * vehicle_params['max_steer_angle']                             
        ###############################################

        # Forward action to the vehicle
        self.vehicle.act({
            "acceleration": acceleration,
            "steering": steering
        })

        #print("prev_act, curr_act, accel, steer, speed:", self.previous_action, action, acceleration, steering, self.vehicle.velocity)
        self._simulate()        

        obs = self._observation()
        info = {
            "is_success": int(self._is_success(obs['achieved_goal'], obs['desired_goal'])),
            "is_collision": int(self.vehicle.crashed),
            "is_over_others_parking_spot": int(self.is_over_others_parking_spot(self.vehicle.position)),
            "is_reverse": int(self.vehicle.velocity<0)         
        }

        self.previous_action = action
        self.previous_obs = obs

        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], info)
        terminal = self._is_terminal()
        return obs, reward, terminal, info

    def reset(self):        
        self._build_parking()
        self._populate_parking()
        self.is_success = False
        self.vehicle.crashed = False
        #return super(ParkingEnv_2outs, self).reset()
        return self._observation()
    
    def is_over_others_parking_spot(self, position):        
        over_others_parking_spots = False
        for _from, to_dict in self.road.network.graph.items():
            for _to, lanes in to_dict.items():
                for _id, lane in enumerate(lanes):
                    if lane != self.goal.lane:
                        over_others_parking_spots = lane.on_lane(position)
                    if (over_others_parking_spots):
                        return True
                    
        return False

    def rot(self, point, angle):
        assert len(point)==2
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

        # Let's gather some params from the Config file
        if self.config["parking_spots"] == 'random':
            self.parking_spots = self.np_random.randint(1,21)
        else:
            self.parking_spots = self.config["parking_spots"]

        if self.config["vehicles_count"] == 'random':
            self.vehicles_count = self.np_random.randint(self.parking_spots) * 2

        elif self.config["vehicles_count"] == 'all':
            self.vehicles_count = (self.parking_spots*2) - 1 

        else:
            self.vehicles_count = self.config["vehicles_count"]

        assert (self.vehicles_count < self.parking_spots*2)

        net = RoadNetwork()
        
        lt = (LineType.CONTINUOUS, LineType.CONTINUOUS)
        
        spots_offset = 0.0
        parking_angles = np.deg2rad([90, 75, 60, 45, 0])
        aisle_width = 20.0
        length = 8.0
        width = 4.0
        
        # Let's start by randomly choosing the parking angle        
        #angle = parking_angles[self.np_random.randint(len(parking_angles))]        
        angle = 0 #np.pi/3
        # Let's now build the parking lot
        for k in range(self.parking_spots):
            x1 = (k - self.parking_spots // 2) * (width + spots_offset) - width / 2
            y1 = aisle_width/2            
            x2 = x1
            y2 = y1 + length

            x3 = x1
            y3 = -y1
            x4 = x3
            y4 = -y2

            x1, y1 = self.rot((x1,y1), angle)
            x2, y2 = self.rot((x2,y2), angle)

            x3, y3 = self.rot((x3,y3), angle)
            x4, y4 = self.rot((x4,y4), angle)
            
            net.add_lane("a", "b", StraightLane([x1, y1], [x2, y2], width=width, line_types=lt))
            net.add_lane("b", "c", StraightLane([x3, y3], [x4, y4], width=width, line_types=lt))

        self.road = Road(network=net,
                         np_random=self.np_random)

    def _populate_parking(self):
        """
            Create some new random vehicles of a given type, and add them on the road.
        """
        ##### ADDING EGO #####
        self.vehicle = Vehicle(self.road, [0, 0], heading=2*np.pi*self.np_random.rand(), velocity=0)
        self.vehicle.MAX_VELOCITY = self.PARKING_MAX_VELOCITY
        self.road.vehicles.append(self.vehicle)
        
        ##### ADDING GOAL #####
        parking_spots_used =[]
        #lane = self.np_random.choice(self.road.network.lanes_list())
        lane = self.road.network.lanes_list()[13]
        parking_spots_used.append(lane)
        goal_heading = lane.heading #+ self.np_random.randint(2) * np.pi
        self.goal = Obstacle(self.road, lane.position(lane.length/2, 0), heading=goal_heading)
        self.goal.COLLISIONS_ENABLED = False
        self.road.vehicles.insert(0, self.goal)

        ##### ADDING OTHER VEHICLES #####
        # vehicles_type = utils.class_from_path(scene.config["other_vehicles_type"])             
        for _ in range(self.vehicles_count):            
            while lane in parking_spots_used: # this loop should never be infinite since we assert that there should be more parking spots/lanes than vehicles
                lane = self.np_random.choice(self.road.network.lanes_list()) # to-do: chceck for empty spots
            parking_spots_used.append(lane)
            
            vehicle_heading = lane.heading #+ self.np_random.randint(2) * np.pi
            self.road.vehicles.append(Vehicle(self.road, lane.position(lane.length/2, 0), heading=vehicle_heading, velocity=0))


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
               
        # DISTANCE TO GOAL
        distance_to_goal_reward = self.distance_2_goal_reward(achieved_goal, desired_goal, p)
        
        # OVER OTHER PARKING SPOTS REWARD        
        over_other_parking_spots_reward = self.OVER_OTHER_PARKING_SPOT_REWARD * np.squeeze(info["is_over_others_parking_spot"])

        # COLLISION REWARD
        collision_reward = self.COLLISION_REWARD * np.squeeze(info["is_collision"])

         # REVERESE DRIVING REWARD
        reverse_reward = self.REVERSE_REWARD * np.squeeze(info["is_reverse"])
        

        reward = (distance_to_goal_reward + \
                  collision_reward + \
                  reverse_reward) #+ \
                  # over_other_parking_spots_reward)
                 # reverse_reward + \
                 # against_traffic_reward + \  
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
            NOTE: If we use HER, we can NOT terminate the episode because they are 
            supposed to have FIXED length. For this reason, we will just RESET and
            keep going.
        """
        if self.vehicle.crashed or self.is_success:
            self.reset()
        return False  # or self._is_success(obs['achieved_goal'], obs['desired_goal'])

    def print_obs_space(self):
        print("obs space ")
        '''obs_format = pp.pformat(np.round(np.reshape(self.previous_obs,(6, 5)),3))
        obs_format = obs_format.rstrip("\n")
        print(obs_format)'''
        print(self.previous_obs["observation"])
        print("achieved_goal")
        print(self.previous_obs["achieved_goal"])
        print("desired_goal")
        print(self.previous_obs["desired_goal"])
        print("actions")
        print("Optimal action ",self.previous_action, "\n")

                