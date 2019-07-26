######################################################################
#          Deep Reinforcement Learning for Autonomous Driving
#                  Created/Modified on: February 5, 2019
#                      Author: Munir Jojo-Verge, Pinaki Gupta
#######################################################################

from __future__ import division, print_function, absolute_import
import numpy as np
from gym import logger

from urban_env import utils
from urban_env.envs.abstract import AbstractEnv
from urban_env.road.road import Road, RoadNetwork
from urban_env.vehicle.control import MDPVehicle
from urban_env.envs.graphics import EnvViewer
from handle_model_files import is_predict_only


class MultilaneEnv(AbstractEnv):
    """
        A urban driving environment.

        The vehicle is driving on a straight urban with several lanes, and is rewarded for reaching a high velocity,
        staying on the rightmost lanes and avoiding collisions.
    """

    COLLISION_REWARD = -200
    """ The reward received when colliding with a vehicle."""
    RIGHT_LANE_REWARD = 0.1
    """ The reward received when driving on the right-most lanes, linearly mapped to zero for other lanes."""
    VELOCITY_REWARD = 5
    """ The reward received when driving at full speed, linearly mapped to zero for lower speeds."""
    LANE_CHANGE_REWARD = -1
    """ The reward received at each lane change action."""
    GOAL_REWARD = 2000

    ROAD_LENGTH = 500

    DEFAULT_CONFIG = {
        "observation": {
            "type": "Kinematics",
            "features": ['x', 'y', 'vx', 'vy', 'psi', 'lane_psi'],
            "vehicles_count": 6
        },
        "initial_spacing": 2,
        "other_vehicles_type": "urban_env.vehicle.behavior.IDMVehicle",
        "centering_position": [0.3, 0.5],
        "collision_reward": COLLISION_REWARD,
        "screen_width": 1800,
        "screen_height": 300,
        "duration": 250,
        "_predict_only": is_predict_only(),
    }

    DIFFICULTY_LEVELS = {
        "EASY": {
            "lanes_count": 2,
            "vehicles_count": 5,
            "duration": 20
        },
        "MEDIUM": {
            "lanes_count": 3,
            "vehicles_count": 10,
            "duration": 30
        },
        "HARD": {
            "lanes_count": 4,
            "vehicles_count": 50,
            "duration": 40
        },
    }

    def __init__(self, config=DEFAULT_CONFIG):
        super(MultilaneEnv, self).__init__(config)
        self.config.update(self.DIFFICULTY_LEVELS["HARD"])
        if self.config["_predict_only"]:
            self.ROAD_LENGTH = 1000
        self.steps = 0
        self.ego_x0 = None
        EnvViewer.SCREEN_HEIGHT = self.config['screen_height']
        EnvViewer.SCREEN_WIDTH = self.config['screen_width'] 
        self.reset()


    def _on_route(self, veh=None):
        return True
    
    def _on_road(self, veh=None):
        if veh is None:
            veh = self.vehicle
        return (veh.position[0] < self.ROAD_LENGTH) and (veh.position[0] > 0)

    def _goal_achieved(self, veh=None):
        if veh is None:
            veh = self.vehicle
        return (veh.position[0] > 0.99 * self.ROAD_LENGTH) and \
                self._on_route(veh)


    def reset(self):
        self._create_road()
        self._create_vehicles()
        self.steps = 0
        return super(MultilaneEnv, self).reset()

    def step(self, action):
        self.steps += 1
        obs, rew, done, info = super(MultilaneEnv, self).step(action)
        self.goal = (self.ROAD_LENGTH - self.vehicle.position[0]) / (7.0 * MDPVehicle.SPEED_MAX) # Normalize
        self.goal = min(1.0, max(-1.0, self.goal)) # Clip
        obs[0] = self.goal # Just a temporary implementation wo explicitly mentioning the goal
        self.episode_travel = self.vehicle.position[0] - self.ego_x0 
        return (obs, rew, done, info)

    def _create_road(self):
        """
            Create a road composed of straight adjacent lanes.
        """
        self.road = Road(network=RoadNetwork.straight_road_network(lanes=self.config["lanes_count"], length=self.ROAD_LENGTH),
                         np_random=self.np_random)

    def _create_vehicles(self):
        """
            Create some new random vehicles of a given type, and add them on the road.
        """
        self.vehicle = MDPVehicle.create_random(self.road, np.random.randint(low=15,high=35), spacing=self.config["initial_spacing"])
        self.road.vehicles.append(self.vehicle)
        self.ego_x0 = self.vehicle.position[0]

        vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        for _ in range(self.config["vehicles_count"]):
            self.road.vehicles.append(vehicles_type.create_random(self.road))

    def _reward(self, action):
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        action_lookup = dict(map(reversed, AbstractEnv.ACTIONS.items()))
        action_reward = {action_lookup['LANE_LEFT']: self.LANE_CHANGE_REWARD, 
                         action_lookup['IDLE']: 0, 
                         action_lookup['LANE_RIGHT']: self.LANE_CHANGE_REWARD, 
                         action_lookup['FASTER']: 0, 
                         action_lookup['SLOWER']: 0,
                         action_lookup['LANE_LEFT_AGGRESSIVE']: self.LANE_CHANGE_REWARD,
                         action_lookup['LANE_RIGHT_AGGRESSIVE']: self.LANE_CHANGE_REWARD
                         }
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)

        collision_reward = self.COLLISION_REWARD * self.vehicle.crashed
        velocity_reward = self.VELOCITY_REWARD * (self.vehicle.velocity_index -1) / (self.vehicle.SPEED_COUNT - 1)
        if (velocity_reward > 0):
            velocity_reward *= self._on_route()
        goal_reward = self.GOAL_REWARD

        if self.vehicle.crashed:
            reward = collision_reward + min(0.0, velocity_reward)
        elif self._goal_achieved():
            reward = goal_reward + velocity_reward
        else:
            reward = velocity_reward
        return reward


    def _is_terminal(self):
        """
            The episode is over if the ego vehicle crashed or the time is out.
        """
        return  self.vehicle.crashed or \
                self._goal_achieved() or \
                (not self._on_road()) or \
                self.steps >= self.config["duration"]

    def _constraint(self, action):
        """
            The constraint signal is the occurrence of collision
        """
        return float(self.vehicle.crashed)
