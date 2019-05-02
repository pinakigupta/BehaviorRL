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
from urban_env.vehicle.control import ControlledVehicle, MDPVehicle
from urban_env.vehicle.dynamics import Obstacle


class TwoWayEnv(AbstractEnv):
    """
        A risk management task: the agent is driving on a two-way lane with icoming traffic.
        It must balance making progress by overtaking and ensuring safety.

        These conflicting objectives are implemented by a reward signal and a constraint signal,
        in the CMDP/BMDP framework.
    """

    COLLISION_REWARD = 0
    LEFT_LANE_CONSTRAINT = 1
    LEFT_LANE_REWARD = 0.2
    HIGH_VELOCITY_REWARD = 0.8

    DEFAULT_CONFIG = {
        "observation": {
            "type": "Kinematics",
            "features": ['x', 'y', 'vx', 'vy'],
            "vehicles_count": 6
        },
        "other_vehicles_type": "urban_env.vehicle.behavior.IDMVehicle",
        "centering_position": [0.3, 0.5]
    }

    def __init__(self):
        super(TwoWayEnv, self).__init__()
        self.steps = 0
        self.reset()

    def step(self, action):
        self.steps += 1
        return super(TwoWayEnv, self).step(action)

    def _reward(self, action):
        """
            The vehicle is rewarded for driving with high velocity
        :param action: the action performed
        :return: the reward of the state-action transition
        """
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)

        reward = self.HIGH_VELOCITY_REWARD * self.vehicle.velocity_index / (self.vehicle.SPEED_COUNT - 1) \
            + self.LEFT_LANE_REWARD * (len(neighbours) - 1 - self.vehicle.target_lane_index[2]) / (len(neighbours) - 1)
        return reward

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
        ego_vehicle = MDPVehicle(road, road.network.get_lane(("a", "b", 1)).position(30, 0), velocity=30)
        road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        for i in range(3):
            self.road.vehicles.append(
                vehicles_type(road,
                              position=road.network.get_lane(("a", "b", 1))
                              .position(70+40*i + 10*self.np_random.randn(), 0),
                              heading=road.network.get_lane(("a", "b", 1)).heading_at(70+40*i),
                              velocity=24 + 2*self.np_random.randn(),
                              enable_lane_change=False)
            )
        for i in range(2):
            v = vehicles_type(road,
                              position=road.network.get_lane(("b", "a", 0))
                              .position(200+100*i + 10*self.np_random.randn(), 0),
                              heading=road.network.get_lane(("b", "a", 0)).heading_at(200+100*i),
                              velocity=20 + 5*self.np_random.randn(),
                              enable_lane_change=False)
            v.target_lane_index = ("b", "a", 0)
            self.road.vehicles.append(v)

