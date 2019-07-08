######################################################################
#          Deep Reinforcement Learning for Autonomous Driving
#                  Created/Modified on: February 5, 2019
#                      Author: Munir Jojo-Verge
#######################################################################

from __future__ import division, print_function, absolute_import
import numpy as np
import gym

from urban_env import utils
from urban_env.envs.abstract import AbstractEnv
from urban_env.road.lane import LineType, StraightLane, SineLane
from urban_env.road.road import Road, RoadNetwork
from urban_env.envs.graphics import EnvViewer
from urban_env.vehicle.control import ControlledVehicle, MDPVehicle
from urban_env.vehicle.dynamics import Obstacle


class TwoWayEnv(AbstractEnv):
    """
        A risk management task: the agent is driving on a two-way lane with icoming traffic.
        It must balance making progress by overtaking and ensuring safety.

        These conflicting objectives are implemented by a reward signal and a constraint signal,
        in the CMDP/BMDP framework.
    """

    COLLISION_REWARD = -200
    #LEFT_LANE_CONSTRAINT = 1
    LEFT_LANE_REWARD = 0
    VELOCITY_REWARD = 5
    GOAL_REWARD = 2000
    ROAD_LENGTH = 1000

    DEFAULT_CONFIG = {
        "observation": {
            "type": "Kinematics",
            "features": ['x', 'y', 'vx', 'vy', 'psi'],
            "vehicles_count": 6
        },
        "other_vehicles_type": "urban_env.vehicle.behavior.IDMVehicle",
        "centering_position": [0.3, 0.5]
    }

    def __init__(self, config=None):
        super(TwoWayEnv, self).__init__()
        self.steps = 0
        self.reset()
        self.goal_achieved = False
        self.ego_x0 = None

        

    def step(self, action):
        self.steps += 1
        self.previous_action = action
        obs, rew, done, info = super(TwoWayEnv, self).step(action)
        self.previous_obs = obs
        return (obs, rew, done, info)


    def _reward(self, action):
        """
            The vehicle is rewarded for driving with high velocity
        :param action: the action performed
        :return: the reward of the state-action transition
        """
        lane_ID = self.vehicle.lane_index[2]
        on_route = (lane_ID==1)

        #print("self.vehicle.position  ",self.vehicle.position)
        if self.ego_x0 is not None:
            if ('_predict_only', False) in gym.Env.metadata.items():
                if '_mega_batch_itr' in gym.Env.metadata:
                    low = 60*gym.Env.metadata['_mega_batch_itr']
                    high = low + 20*gym.Env.metadata['_mega_batch_itr']
                else:
                    low = 180 
                    high = low + 40
                self.goal_achieved =  (self.vehicle.position[0] > self.ego_x0+np.random.randint(low = low,high=high))
        #neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        collision_reward = self.COLLISION_REWARD * self.vehicle.crashed
        velocity_reward = self.VELOCITY_REWARD * (self.vehicle.velocity_index -1) / (self.vehicle.SPEED_COUNT - 1)
        if (velocity_reward>0):
            velocity_reward *= on_route
        #lane_reward = 0 #self.LEFT_LANE_REWARD * (len(neighbours) - 1 - self.vehicle.target_lane_index[2]) / (len(neighbours) - 1)
        goal_reward = self.GOAL_REWARD 
        #print("collision_reward ",collision_reward, " velocity_reward ",velocity_reward, " lane_reward ",lane_reward," goal_reward ",goal_reward)
        if self.vehicle.crashed:
            reward =  collision_reward + min(0,velocity_reward)
        elif self.goal_achieved and on_route:
            reward = goal_reward + velocity_reward
        else :
            reward =   velocity_reward  
        return reward

    def _is_terminal(self):
        """
            The episode is over if the ego vehicle crashed or the time is out.
        """
        lane_ID = self.vehicle.lane_index[2]
        on_road = self.vehicle.position[0] < self.ROAD_LENGTH
        terminal = self.vehicle.crashed or self.goal_achieved or (not on_road)
        return terminal

    def _constraint(self, action):
        """
            The constraint signal is the time spent driving on the opposite lane, and occurence of collisions.
        """
        return float(self.vehicle.crashed) + float(self.vehicle.lane_index[2] == 0)/15

    def reset(self):
        self.steps = 0
        self._make_road()
        self._make_vehicles()
        return super(TwoWayEnv, self).reset()

    def _make_road(self, length = 800):
        """
            Make a road composed of a two-way road.
        :return: the road
        """
        net = RoadNetwork()
        length = self.ROAD_LENGTH
        # Lanes
        net.add_lane("a", "b", StraightLane([0, 0], [length, 0],
                                            line_types=[LineType.CONTINUOUS_LINE, LineType.STRIPED]))
        net.add_lane("a", "b", StraightLane([0, StraightLane.DEFAULT_WIDTH], [length, StraightLane.DEFAULT_WIDTH],
                                            line_types=[LineType.NONE, LineType.CONTINUOUS_LINE]))
        net.add_lane("b", "a", StraightLane([length, 0], [0, 0],
                                            line_types=[LineType.NONE, LineType.NONE]))

        road = Road(network=net, np_random=self.np_random)
        self.road = road

    def _make_vehicles(self):
        """
            Populate a road with several vehicles on the road
        :return: the ego-vehicle
        """
        road = self.road
        ego_vehicle = MDPVehicle(road, road.network.get_lane(("a", "b", 1)).position(np.random.randint(low=360,high=420), 0),\
             velocity=np.random.randint(low=15,high=35))
        road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle
        self.ego_x0 = ego_vehicle.position[0]
        #print("ego_x",self.ego_x0)
        vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        scene_complexity = 3
        if '_mega_batch_itr' in gym.Env.metadata:
            scene_complexity = gym.Env.metadata['_mega_batch_itr']

        if '_predict_only' in gym.Env.metadata:
            if gym.Env.metadata['_predict_only']:
                scene_complexity = 3
        
        # stationary vehicles
        for i in range(np.random.randint(low=0,high=scene_complexity)):
            x0 = self.ego_x0+90+90*i + 10*self.np_random.randn()
            self.road.vehicles.append(
                vehicles_type(road,
                              position=road.network.get_lane(("a", "b", 1))
                              .position(x0, 1),
                              heading=road.network.get_lane(("a", "b", 1)).heading_at(100),
                              velocity=0,target_velocity = 0,
                              target_lane_index = ("a", "b", 1), lane_index = ("a", "b", 1),                             
                              enable_lane_change=False)
            )
            
        for i in range(np.random.randint(low=0,high=2*scene_complexity)):
            x0 = self.ego_x0+90+40*i + 10*self.np_random.randn()
            v = vehicles_type(road,
                              position=road.network.get_lane(("a", "b", 1))
                              .position( x0, 0),
                              heading=road.network.get_lane(("a", "b", 1)).heading_at(100),
                              velocity=max(0,10 + 2*self.np_random.randn()),
                              target_lane_index = ("a", "b", 1), lane_index = ("a", "b", 1),                             
                              enable_lane_change=False)
            front_vehicle, _ = self.road.neighbour_vehicles(v)
            d = v.lane_distance_to(front_vehicle) 
            if (d<5):
                continue
            elif(d<20):
                v.velocity = max(0,2.5 + 0.5*self.np_random.randn())
            self.road.vehicles.append(v)
        
        
        '''
        # stationary vehicles Left Lane
        for i in range(np.random.randint(low=0,high=2*scene_complexity)):
            x0 = self.ROAD_LENGTH-self.ego_x0-100-120*i + 10*self.np_random.randn()
            v = vehicles_type(road,
                              position=road.network.get_lane(("b", "a", 0))
                              .position(x0 , 1),
                              heading=road.network.get_lane(("b", "a", 0)).heading_at(x0),
                              velocity=0,target_velocity = 0,
                              target_lane_index = ("b", "a", 0), lane_index = ("b", "a", 0),
                              enable_lane_change=False)
            v.target_lane_index = ("b", "a", 0)
            v.lane_index = ("b", "a", 0)
            self.road.vehicles.append(v)'''

        
        for i in range(np.random.randint(low=0,high=2*scene_complexity)):
            x0 = self.ROAD_LENGTH-self.ego_x0-20-120*i + 10*self.np_random.randn()
            v = vehicles_type(road,
                              position=road.network.get_lane(("b", "a", 0))
                              .position(x0, 0.1),
                              heading=road.network.get_lane(("b", "a", 0)).heading_at(100),
                              velocity=max(0,20 + 5*self.np_random.randn()),
                              target_lane_index = ("b", "a", 0), lane_index = ("b", "a", 0),
                              enable_lane_change=True)
            v.target_lane_index = ("b", "a", 0)
            v.lane_index = ("b", "a", 0)
            front_vehicle, _ = self.road.neighbour_vehicles(v)
            d = v.lane_distance_to(front_vehicle)
            if(d<5):
                continue 
            elif(d<20):
                v.velocity = max(0,4 + self.np_random.randn())
            self.road.vehicles.append(v)

    def print_obs_space(self):
        print("obs space ")
        import pprint
        pp = pprint.PrettyPrinter(indent=4)
        obs_format = pp.pformat(np.round(np.reshape(self.previous_obs,(6, 5)),3))
        obs_format = obs_format.rstrip("\n")
        print(obs_format)
        print(self.previous_obs)
        print("actions")
        print("Optimal action ", AbstractEnv.ACTIONS[self.previous_action], "\n")

