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
from urban_env.vehicle.dynamics import Vehicle, Obstacle


class MultilaneEnv(AbstractEnv):
    """
        A urban driving environment.

        The vehicle is driving on a straight urban with several lanes, and is rewarded for reaching a high velocity,
        staying on the rightmost lanes and avoiding collisions.
    """

    RIGHT_LANE_REWARD = 0.1
    """ The reward received when driving on the right-most lanes, linearly mapped to zero for other lanes."""
    LANE_CHANGE_REWARD = -1
    AGGRESSIVE_LANE_CHANGE_REWARD = -3

    ROAD_LENGTH = 500
    ROAD_SPEED = 35


    DEFAULT_CONFIG = {**AbstractEnv.DEFAULT_CONFIG, 
        **{
            "observation": {
                "type": "Kinematics",
                "features": ['x', 'y', 'vx', 'vy', 'psi'],
                "relative_features": ['x'],
                "vehicles_count": 6
            },
            "other_vehicles_type": "urban_env.vehicle.control.IDMDPVehicle",
            "duration": 250,
            "_predict_only": is_predict_only(),
            "screen_width": 1600,
            "screen_height": 400,
            "DIFFICULTY_LEVELS": 2,
            "COLLISION_REWARD": -200,
            "INVALID_ACTION_REWARD": 0,
            "VELOCITY_REWARD": 5,
            "GOAL_REWARD": 2000,
            "OBS_STACK_SIZE": 1,
            "GOAL_LENGTH": 1000,
            "initial_spacing": 2,
            "centering_position": [0.3, 0.5],
            }
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
        self.config.update(self.DIFFICULTY_LEVELS["MEDIUM"])
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
        self.steps = 0
        self._create_road()
        self._create_vehicles()
        return super(MultilaneEnv, self).reset()

    def step(self, action):
        self.steps += 1
        self.previous_action = action
        obs, rew, done, info = super(MultilaneEnv, self).step(action)
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

        self.vehicle = MDPVehicle.create_random(road=self.road,
                                                velocity=np.random.randint(low=15,high=35),
                                                spacing=self.config["initial_spacing"],
                                                config=self.config)


        self.vehicle.is_ego_vehicle = True
        self.road.vehicles.append(self.vehicle)
        self.ego_x0 = self.vehicle.position[0]

        vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        ahead_vehicles = self.config["vehicles_count"] // 2
        behind_vehicles = self.config["vehicles_count"] - ahead_vehicles
        for _ in range(ahead_vehicles):
            self.road.vehicles.append(vehicles_type.create_random(road=self.road,
                                                                  ahead=True,
                                                                  config=self.config)

                                     )

        for _ in range(behind_vehicles):
            self.road.vehicles.append(vehicles_type.create_random(road=self.road,
                                                                  ahead=False,
                                                                  config=self.config)
                                     )

        for _ in range(10):
            self.road.vehicles.append(Obstacle.create_random(road=self.road,
                                                             ahead=False,
                                                             config=self.config
                                                             )
                                     )

        # Add the virtual obstacles
        lane = self.road.network.lanes_list()[0]
        x0 = lane.length/2
        position = lane.position(x0, -3.5)
        lane_index = self.road.network.get_closest_lane_index(
                                                            position=position,
                                                            heading=0  
                                                             )  
        virtual_obstacle_left =       Obstacle(self.road,
                                               position=position,
                                               heading=lane.heading_at(x0),
                                               velocity=0,
                                               target_velocity=0,
                                               lane_index=lane_index,
                                               target_lane_index=lane_index,                     
                                               enable_lane_change=False,
                                               config=self.config)
        virtual_obstacle_left.LENGTH = lane.length
        virtual_obstacle_left.virtual = True
        self.road.vehicles.append(virtual_obstacle_left)
        self.road.virtual_vehicles.append(virtual_obstacle_left)


        lane = self.road.network.lanes_list()[-1]
        x0 = lane.length/2
        position = lane.position(x0, 3.5)
        virtual_obstacle_right =      Obstacle(self.road,
                                               position=position,
                                               heading=lane.heading_at(x0),
                                               velocity=0,
                                               target_velocity=0,
                                               lane_index=lane_index,
                                               target_lane_index=lane_index,                     
                                               enable_lane_change=False,
                                               config=self.config)
        virtual_obstacle_right.LENGTH = lane.length
        virtual_obstacle_right.virtual = True
        self.road.vehicles.append(virtual_obstacle_right)
        self.road.virtual_vehicles.append(virtual_obstacle_right)


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
                         action_lookup['LANE_LEFT_AGGRESSIVE']: self.AGGRESSIVE_LANE_CHANGE_REWARD,
                         action_lookup['LANE_RIGHT_AGGRESSIVE']: self.AGGRESSIVE_LANE_CHANGE_REWARD
                         }
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)

        collision_reward = self.config["COLLISION_REWARD"] * self.vehicle.crashed
        velocity_reward = self.config["VELOCITY_REWARD"] * (self.vehicle.velocity_index -1) / (self.vehicle.SPEED_COUNT - 1)
        if (velocity_reward > 0):
            velocity_reward *= self._on_route()
        goal_reward = self.config["GOAL_REWARD"]

        if self.vehicle.crashed:
            reward = collision_reward + min(0.0, velocity_reward + action_reward[action])
        elif self._goal_achieved():
            reward = goal_reward + velocity_reward + action_reward[action]
        else:
            reward = velocity_reward + action_reward[action]
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

    def print_obs_space(self):
        print("obs space, step ", self.steps)
        #sys.stdout.flush()
        pp = pprint.PrettyPrinter(indent=4)
        numoffeatures = len(self.config["observation"]["features"])
        numfofobs = len(self.obs)
        numofvehicles = numfofobs//numoffeatures
        close_vehicle_ids =  [int(self.vehicle.Id())]
        modified_obs = self.obs
        for v in self.close_vehicles:
            close_vehicle_ids.append(int(v.Id()))
        close_vehicle_ids.extend([-1]*(numofvehicles-len(close_vehicle_ids)))
        Idx = 0
        obs_Idx = 0
        while True:
            temp = modified_obs
            del(modified_obs)
            modified_obs = np.insert(temp, obs_Idx, close_vehicle_ids[Idx])
            del(temp)
            Idx += 1
            obs_Idx += numoffeatures+1
            if Idx >= len(close_vehicle_ids):
                break

        np.set_printoptions(precision=3, suppress=True)
        obs_format = pp.pformat(np.round(np.reshape(modified_obs, (numofvehicles, numoffeatures+1 )), 3))
        obs_format = obs_format.rstrip("\n")
        print(obs_format)
