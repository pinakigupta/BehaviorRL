######################################################################
#          Deep Reinforcement Learning for Autonomous Driving
#                  Created/Modified on: February 5, 2019
#                      Author: Munir Jojo-Verge
#######################################################################

from __future__ import division, print_function, absolute_import
import numpy as np

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
    
    OPPOSITE_LANE_CONSTRAINT = 1
    OPPOSITE_LANE_REWARD = -0.1    
    OVER_SPEED_REWARD = -0.8

    COLLISION_REWARD = -1
    """ The reward received when colliding with a vehicle."""

    RIGHT_LANE_CHANGE_REWARD = 0.0 #0.001
    """ The reward received when driving on the right-most lanes, linearly mapped to zero for other lanes."""
    LEFT_LANE_CHANGE_REWARD = 0.0
    """ The reward received when driving on the right-most lanes, linearly mapped to zero for other lanes."""
    
    HIGH_VELOCITY_REWARD = 0.0003
    """ The reward received when driving at full speed, linearly mapped to zero for lower speeds."""
    
    


    DEFAULT_CONFIG = {
        "observation": {
            "type": "Kinematics",
            "features": ['x', 'y', 'vx', 'vy'],
            "vehicles_count": 6
        },
        "other_vehicles_type": "urban_env.vehicle.behavior.IDMVehicle",
        "centering_position": [0.3, 0.5],
        "screen_width": 600 * 2,
        "screen_height": 300 * 2 
    }

    def __init__(self):
        super(TwoWayEnv, self).__init__()
        self.steps = 0
        self.reset()
        EnvViewer.SCREEN_HEIGHT = self.config['screen_height']
        EnvViewer.SCREEN_WIDTH = self.config['screen_width']     

    def step(self, action):
        self.steps += 1
        return super(TwoWayEnv, self).step(action)

    def _reward(self, action):
        """
            The vehicle is rewarded for driving with high velocity
        :param action: the action performed
        :return: the reward of the state-action transition
        """
        action_reward = {0: self.LEFT_LANE_CHANGE_REWARD,
                         1: 0,
                         2: self.RIGHT_LANE_CHANGE_REWARD,
                         3: 0,
                         4: 0}
        reward = self.COLLISION_REWARD * self.vehicle.crashed \
            + self.HIGH_VELOCITY_REWARD * self.vehicle.velocity_index / (self.vehicle.SPEED_COUNT - 1)
                 # + self.RIGHT_LANE_CHANGE_REWARD * self.vehicle.lane_index[2] / 1

        # # Altruistic penalty
        # for vehicle in self.road.vehicles:
        #     if vehicle.lane_index == ("b", "c", 2) and isinstance(vehicle, ControlledVehicle):
        #         reward += self.MERGING_VELOCITY_REWARD * \
        #                   (vehicle.target_velocity - vehicle.velocity) / vehicle.target_velocity

        return utils.remap(action_reward[action] + reward,
                           [self.COLLISION_REWARD, self.HIGH_VELOCITY_REWARD + self.LEFT_LANE_CHANGE_REWARD +  self.RIGHT_LANE_CHANGE_REWARD],
                           [0, 1])
        # neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)

        # high_velocity_reward = self.HIGH_VELOCITY_REWARD * self.vehicle.velocity_index / (self.vehicle.SPEED_COUNT - 1)

        # #over_speed_reward = self.OVER_SPEED_REWARD * (self.vehicle.velocity > self.vehicle.MAX_VELOCITY)

        # collision_reward = self.COLLISION_REWARD * int(self.vehicle.crashed) 

        # opposite_lane_reward = self.OPPOSITE_LANE_REWARD * np.abs(self.vehicle.lane_index[2]!=1)
        
        # reward = collision_reward + opposite_lane_reward + high_velocity_reward
        #     #  over_speed_reward + \
        #     #  
            

        # return reward

    def _is_terminal(self):
        """
            The episode is over if the ego vehicle crashed or the time is out.
        """
        return self.vehicle.crashed

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
        ego_vehicle = MDPVehicle(road, road.network.get_lane(("a", "b", 1)).position(30, 0), velocity=15)        
        road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        for i in range(5):
            self.road.vehicles.append(
                vehicles_type(road,
                              position=road.network.get_lane(("a", "b", 1))
                              .position(70+40*i + 10*self.np_random.randn(), 0),
                              heading=road.network.get_lane(("a", "b", 1)).heading_at(70+40*i),
                              velocity=10 + 2*self.np_random.randn(),
                              enable_lane_change=False)
            )
        for i in range(5):
            v = vehicles_type(road,
                              position=road.network.get_lane(("b", "a", 0))
                              .position(200+100*i + 10*self.np_random.randn(), 0),
                              heading=road.network.get_lane(("b", "a", 0)).heading_at(200+100*i),
                              velocity=17 + 5*self.np_random.randn(),
                              enable_lane_change=False)
            v.target_lane_index = ("b", "a", 0)
            self.road.vehicles.append(v)

