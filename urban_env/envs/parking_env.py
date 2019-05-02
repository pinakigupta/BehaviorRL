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
from urban_env.envs.graphics import EnvViewer
from urban_env.road.lane import StraightLane, LineType
from urban_env.road.road import Road, RoadNetwork
from urban_env.vehicle.dynamics import Vehicle, Obstacle
import urban_env.utils as utils

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

class ParkingEnv(AbstractEnv, GoalEnv):
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

    REWARD_WEIGHTS = [7/100, 7/100, 6/100, 6/100, 9/10, 9/10]
    SUCCESS_THRESHOLD = 0.35

    
    DEFAULT_CONFIG = {        
        "observation": {
            "type": "KinematicsGoal",
            "features": ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
            "scale": 100,
            "normalize": False
        },
        "other_vehicles_type": "urban_AD_env.vehicle.behavior.IDMVehicle",
        "centering_position": [0.5, 0.5],
        "parking_spots": 15, #'random', # Parking Spots Per side
        "vehicles_count": 0, #'random', # Total number of cars in the parking (apart from Ego)
        "screen_width": 600,
        "screen_height": 300 
    }    

    def __init__(self):
        self.config = self.DEFAULT_CONFIG.copy()
        super(ParkingEnv, self).__init__()
        self.reset()

        # ACTION SPACE: 
        ### Throttle: [0 to 1], 
        ### Brake   : [0 to 1] 
        ### steering: [-1 to 1],
        ### reverse : [-1 to 1] => from -1 to 0 Reverse and from 0 to 1 Forward. 
        self.action_space = Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 1]), dtype=np.float32)
        self.previous_action = np.array([0.0, 0.0, 1.0])

        self.REWARD_WEIGHTS = np.array(self.REWARD_WEIGHTS)
        
        
        EnvViewer.SCREEN_HEIGHT = self.config['screen_height']
        EnvViewer.SCREEN_WIDTH = self.config['screen_width']     

    
    def step(self, action):                
        
        ################################   
        prev_in_reverse = self.previous_action[2].item()
        cmd_reverse = action[2].item()

        allow_switch_gear = np.abs(self.vehicle.velocity) < VELOCITY_EPSILON
        if ( (prev_in_reverse <  0.0 and bool(cmd_reverse < 0.0) ) or
             (prev_in_reverse >= 0.0 and bool(cmd_reverse < 0.0)  and allow_switch_gear)
        ):
            direction = -1
        elif ((prev_in_reverse >= 0.0 and bool(cmd_reverse >= 0.0) ) or
              (prev_in_reverse <  0.0 and bool(cmd_reverse >= 0.0)  and allow_switch_gear)
        ):
            direction = 1
        else: # You are trying to make an illegal gear switch and you should stay on whatever your last gear is
            direction = prev_in_reverse

        throttle_brake = action[0].item()
        if throttle_brake < 0.0: # Only Braking
            acceleration = np.abs(throttle_brake) * vehicle_params['max_deceleration'] * direction
            if ( (prev_in_reverse ==  1 and self.vehicle.velocity < 0) or # if Moving forward and braking (i.e acc < 0), the max acce is 0 and the min is max_decel
                 (prev_in_reverse == -1 and self.vehicle.velocity > 0)): # if we are moving in reverse..and braking (i.e acc > 0), the min acce is 0 and the max is max_accel
                acceleration = 0.0
                self.vehicle.velocity = 0.0            
        else: # Only Throttle
            acceleration = throttle_brake * vehicle_params['max_acceleration'] * direction
                
        action[2] = direction

        steering = action[1].item() * vehicle_params['max_steer_angle']
                             
        ###############################################

        # Forward action to the vehicle
        self.vehicle.act({
            "acceleration": acceleration,
            "steering": steering
        })

        print("prev_act, curr_act, accel, steer, speed:", self.previous_action, action, acceleration, steering, self.vehicle.velocity)
        self._simulate()        

        obs = self.observation.observe()
        info = {
            "is_success": self._is_success(obs['achieved_goal'], obs['desired_goal']),
            "is_collision": int(self.vehicle.crashed),
            "is_over_others_parking_spot": int(self.is_over_others_parking_spot(self.vehicle.position)),
            "is_reverse": int(self.vehicle.velocity<0),
            "action": action,
            "prev_action": self.previous_action
        }

        self.previous_action = action

        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], info)
        terminal = self._is_terminal()
        return obs, reward, terminal, info

    def reset(self):
        self._build_parking()
        self._populate_parking()
        return super(ParkingEnv, self).reset()
    
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
        else:
            self.vehicles_count = self.config["vehicles_count"]
        assert (self.vehicles_count < self.parking_spots*2)

        net = RoadNetwork()
        
        lt = (LineType.CONTINUOUS, LineType.CONTINUOUS)
        
        spots_offset = 0.0
        parking_angles = np.deg2rad([90, 75, 60, 45, 0])
        aisle_width = 10.0
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
        self.vehicle = Vehicle(self.road, [0, 0], 2*np.pi*self.np_random.rand(), velocity=0)
        self.vehicle.MAX_VELOCITY = self.PARKING_MAX_VELOCITY
        self.road.vehicles.append(self.vehicle)
        
        ##### ADDING GOAL #####
        parking_spots_used =[]
        lane = self.np_random.choice(self.road.network.lanes_list())
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
        """
        return self.vehicle.crashed  # or self._is_success(obs['achieved_goal'], obs['desired_goal'])
