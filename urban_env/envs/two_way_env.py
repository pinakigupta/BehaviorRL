######################################################################
#          Deep Reinforcement Learning for Autonomous Driving
#                  Created/Modified on: February 5, 2019
#                      Author: Munir Jojo-Verge, Pinaki Gupta
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
        "duration": 250,

    }

    def __init__(self, config=DEFAULT_CONFIG):
        super(TwoWayEnv, self).__init__()
        self.reset()
        self.goal_achieved = False
        self.ego_x0 = None

        

    def step(self, action):
        self.steps += 1
        self.previous_action = action
        obs, rew, done, info = super(TwoWayEnv, self).step(action)
        self.previous_obs = obs
        self.episode_travel = self.vehicle.position[0] - self.ego_x0
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
            if ('_predict_only', False) in self.config.items():
                if 'DIFFICULTY_LEVELS' in self.config:
                    low = 60*self.config['DIFFICULTY_LEVELS']
                    high = low + 20*self.config['DIFFICULTY_LEVELS']
                else:
                    low = 120
                    high = low + 40
                self.goal_achieved =  (self.vehicle.position[0] > 0.99 * self.ROAD_LENGTH ) and on_route
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
        elif self.goal_achieved:
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
        terminal = self.vehicle.crashed or self.goal_achieved or (not on_road) or (self.steps >= self.config["duration"])
        #print("self.steps ",self.steps," terminal ",terminal)
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
        ego_lane = road.network.get_lane(("a", "b", 1))
        ego_init_position = ego_lane.position(np.random.randint(low=660,high=720), 0)
        ego_vehicle = MDPVehicle(road, position=ego_init_position, velocity=np.random.randint(low=15,high=35))
        road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle
        self.ego_x0 = ego_vehicle.position[0]
        #print("ego_x",self.ego_x0)
        vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        scene_complexity = 3
        if 'DIFFICULTY_LEVELS' in self.config:
            scene_complexity = self.config['DIFFICULTY_LEVELS']

        if '_predict_only' in self.config:
            if self.config['_predict_only']:
                scene_complexity = 3
        
        # stationary vehicles
        stat_veh_x0 = []
        rand_stat_veh_count = np.random.randint(low=0,high=scene_complexity)
        for i in range(rand_stat_veh_count):
            x0 = self.ego_x0+90+90*i + 10*self.np_random.randn()
            stat_veh_x0.append(x0)
            self.road.vehicles.append(
                vehicles_type(road,
                              position=road.network.get_lane(("a", "b", 1))
                              .position(x0, 1),
                              heading=road.network.get_lane(("a", "b", 1)).heading_at(100),
                              velocity=0,target_velocity = 0,
                              target_lane_index = ("a", "b", 1), lane_index = ("a", "b", 1),                             
                              enable_lane_change=False)
            )
            
        rand_veh_count = np.random.randint(low=0,high=2*scene_complexity)
        for i in range(rand_veh_count):
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
        
        
        
        # stationary vehicles Left Lane
        #if (rand_stat_veh_count == 0):
            rand_stat_veh_count = np.random.randint(low=0,high=3*scene_complexity)
        #else:
        #    rand_stat_veh_count = 0
        for i in range(0):
            x0 = self.ROAD_LENGTH-self.ego_x0-100-120*i + 10*self.np_random.randn()
            x0_wrt_ego_lane = self.ROAD_LENGTH - x0
            min_offset = 1e6

            if stat_veh_x0:
                dist_from_ego_lane_parked_vehs = [x - x0_wrt_ego_lane for x in stat_veh_x0]
                min_offset = min([abs(y) for y in dist_from_ego_lane_parked_vehs])
            if (min_offset < 20):
                break
            else:
                v = vehicles_type(road,
                                  position=road.network.get_lane(("b", "a", 0))
                                  .position(x0 , 1),
                                  heading=road.network.get_lane(("b", "a", 0)).heading_at(x0),
                                  velocity=0,target_velocity = 0,
                                  target_lane_index = ("b", "a", 0), lane_index = ("b", "a", 0),
                                  enable_lane_change=False)
                v.target_lane_index = ("b", "a", 0)
                v.lane_index = ("b", "a", 0)
                self.road.vehicles.append(v)

        
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

    