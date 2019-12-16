######################################################################
#          Deep Reinforcement Learning for Autonomous Driving
#                  Created/Modified on: February 5, 2019
#                      Author:   Pinaki Gupta
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


class MergeEnv(AbstractEnv):
    """
        A urban merge negotiation environment.

        The ego-vehicle is driving on a urban and approached a merge, with some vehicles incoming on the access ramp.
        It is rewarded for maintaining a high velocity and avoiding collisions, but also making room for merging
        vehicles.
    """
    EGO_MERGING = True

    COLLISION_REWARD = -1
    RIGHT_LANE_CHANGE_REWARD = 0.0
    LEFT_LANE_CHANGE_REWARD = 0.0 # When Ego is merging, We want to encourange the merging
    HIGH_VELOCITY_REWARD = 0.02
    #MERGING_VELOCITY_REWARD = -0.5
    

    DEFAULT_CONFIG = {
        "observation": {
            "type": "Kinematics"
        },
        "other_vehicles_type": "urban_env.vehicle.behavior.IDMVehicle",
        "centering_position": [0.3, 0.5],
        "screen_width": 600 * 2,
        "screen_height": 300 * 2 
    }

    def __init__(self):
        super(MergeEnv, self).__init__()
        self.reset()
        EnvViewer.SCREEN_HEIGHT = self.config['screen_height']
        EnvViewer.SCREEN_WIDTH = self.config['screen_width']     
        self.enable_auto_render = True

    def _reward(self, action):
        """
            The vehicle is rewarded for driving with high velocity on lanes to the right and avoiding collisions, but
            an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low velocity.
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

    def _is_terminal(self):
        """
            The episode is over when a collision occurs or when the access ramp has been passed.
        """
        return self.vehicle.crashed or self.vehicle.position[0] > 370

    def reset(self):
        self._make_road()
        self._make_vehicles()
        return super(MergeEnv, self).reset()

    def _make_road(self):
        """
            Make a road composed of a straight urban and a merging lane.
        :return: the road
        """
        net = RoadNetwork()

        # urban lanes
        ends = [150, 80, 80, 150]  # Before, converging, merge, after
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [0, StraightLane.DEFAULT_WIDTH]
        line_type = [[c, s], [n, c]]
        line_type_merge = [[c, s], [n, s]]
        for i in range(2):
            net.add_lane("a", "b", StraightLane([0, y[i]], [sum(ends[:2]), y[i]], line_types=line_type[i]))
            net.add_lane("b", "c", StraightLane([sum(ends[:2]), y[i]], [sum(ends[:3]), y[i]], line_types=line_type_merge[i]))
            net.add_lane("c", "d", StraightLane([sum(ends[:3]), y[i]], [sum(ends), y[i]], line_types=line_type[i]))

        # Merging lane
        amplitude = 3.25
        ljk = StraightLane([0, 6.5 + 4 + 4], [ends[0], 6.5 + 4 + 4], line_types=[c, c], forbidden=True)
        lkb = SineLane(ljk.position(ends[0], -amplitude), ljk.position(sum(ends[:2]), -amplitude),
                       amplitude, 2 * np.pi / (2*ends[1]), np.pi / 2, line_types=[c, c], forbidden=True)
        lbc = StraightLane(lkb.position(ends[1], 0), lkb.position(ends[1], 0) + [ends[2], 0],
                           line_types=[n, c], forbidden=True)
        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)
        net.add_lane("b", "c", lbc)
        road = Road(network=net, np_random=self.np_random)
        road.vehicles.append(Obstacle(road, lbc.position(ends[2], 0)))
        self.road = road

    def _make_vehicles(self):
        """
            Populate a road with several vehicles on the urban and on the merging lane, as well as an ego-vehicle.
        :return: the ego-vehicle
        """
        road = self.road
        ego_vehicle = MDPVehicle(road, road.network.get_lane(("a", "b", 1)).position(30, 0), velocity=30)
        road.vehicles.append(ego_vehicle)

        # Adding all agents on the main road
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(90, 0), velocity=29))
        road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 1)).position(70, 0), velocity=31))
        road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(5, 0), velocity=31.5))

        # Adding the merging vehicle 
        if self.EGO_MERGING:
            ego_vehicle = MDPVehicle(road, road.network.get_lane(("j", "k", 0)).position(110, 0), velocity=31.5)
        else:
            ego_vehicle = MDPVehicle(road, road.network.get_lane(("a", "b", 1)).position(30, 0), velocity=30)
            merging_v = other_vehicles_type(road, road.network.get_lane(("j", "k", 0)).position(110, 0), velocity=20)
            merging_v.target_velocity = 30
            road.vehicles.append(merging_v)

        road.vehicles.append(ego_vehicle)                
        self.vehicle = ego_vehicle
