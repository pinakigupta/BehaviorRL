######################################################################
#          Deep Reinforcement Learning for Autonomous Driving
#                  Created/Modified on: February 5, 2019
#                      Author: Munir Jojo-Verge, Pinaki Gupta
#######################################################################

from __future__ import division, print_function, absolute_import
import copy
import gym
import pandas
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from collections import deque

from urban_env import utils
from urban_env.envs.finite_mdp import  compute_ttc_grid
from urban_env.envs.graphics import EnvViewer
from urban_env.road.lane import AbstractLane
from urban_env.vehicle.behavior import IDMVehicle
from urban_env.vehicle.control import MDPVehicle
from urban_env.vehicle.dynamics import Obstacle
from urban_env.envdict import RED, GREEN, BLUE, YELLOW, BLACK, PURPLE, DEFAULT_COLOR, EGO_COLOR, WHITE

class ObservationType(object):
    def space(self):
        raise NotImplementedError()

    def observe(self):
        raise NotImplementedError()


class TimeToCollisionObservation(ObservationType):
    def __init__(self, env, ref_vehicle, horizon=10, **kwargs):
        self.env = env
        self.horizon = horizon
        self.vehicle = ref_vehicle

    def space(self):
        try:
            return spaces.Box(shape=self.observe().shape, low=0, high=1, dtype=np.float32)
        except AttributeError:
            return None

    def observe(self):
        grid = compute_ttc_grid(self.env, time_quantization=1/self.env.POLICY_FREQUENCY, horizon=self.horizon)
        padding = np.ones(np.shape(grid))
        padded_grid = np.concatenate([padding, grid, padding], axis=1)
        obs_lanes = 3
        l0 = grid.shape[1] + self.vehicle.lane_index[2] - obs_lanes // 2
        lf = grid.shape[1] + self.vehicle.lane_index[2] + obs_lanes // 2
        clamped_grid = padded_grid[:, l0:lf+1, :]
        repeats = np.ones(clamped_grid.shape[0])
        repeats[np.array([0, -1])] += clamped_grid.shape[0]
        padded_grid = np.repeat(clamped_grid, repeats.astype(int), axis=0)
        obs_velocities = 3
        v0 = grid.shape[0] + self.vehicle.velocity_index - obs_velocities // 2
        vf = grid.shape[0] + self.vehicle.velocity_index + obs_velocities // 2
        clamped_grid = padded_grid[v0:vf + 1, :, :]
        return clamped_grid


class KinematicObservation(ObservationType):
    """
        Observe the kinematics of nearby vehicles.
    """
    FEATURES = ['presence', 'x', 'y', 'vx', 'vy', 'psi', 'lane_psi', 'length']
    #STACK_SIZE = 2

    def __init__(self, env, ref_vehicle, features=FEATURES, vehicles_count=9, **kwargs):
        """
        :param env: The environment to observe
        :param features: Names of features used in the observation
        :param vehicles_count: Number of observed vehicles
        """
        self.env = env
        self.features = features
        self.vehicles_count = vehicles_count
        self.virtual_vehicles_count = 1
        self.close_vehicles = None
        self.observations = None
        self.vehicle = ref_vehicle


    def space(self):
        one_obs_space = spaces.Box(shape=(len(self.features) * (self.vehicles_count + self.virtual_vehicles_count),), low=-1, high=1, dtype=np.float32)
        if(self.env.OBS_STACK_SIZE == 1):
            return one_obs_space
        return spaces.Tuple(tuple([one_obs_space]*self.env.OBS_STACK_SIZE))

    def normalize(self, df):
        """
            Normalize the observation values.

            For now, assume that the road is straight along the x axis.
        :param Dataframe df: observation data
        """
        side_lanes = self.env.road.network.all_side_lanes(self.vehicle.lane_index)
        self.x_position_range = 7.0 * MDPVehicle.SPEED_MAX
        self.y_position_range = AbstractLane.DEFAULT_WIDTH * len(side_lanes)
        self.velocity_range = 1.5*MDPVehicle.SPEED_MAX
        df['x'] = utils.remap(df['x']  , [- self.x_position_range,  self.x_position_range], [-1, 1])
        df['y'] = utils.remap(df['y'], [-self.y_position_range, self.y_position_range], [-1, 1])
        df['vx'] = utils.remap(df['vx'] , [-self.velocity_range, self.velocity_range], [-1, 1])
        df['vy'] = utils.remap(df['vy'], [-self.velocity_range, self.velocity_range], [-1, 1])

        if 'psi' in df:
            df['psi'] = df['psi']/(2*np.pi)
        if 'lane_psi' in df:
            df['lane_psi'] = df['lane_psi']/(2*np.pi)
        if 'length' in df:
            df['length'] = df['length']/400
        return df

    def observe(self):
        # Add ego-vehicle
        df = pandas.DataFrame.from_records([self.vehicle.to_dict(self.vehicle)])[self.features]
        '''for col in df.columns:
            df[col].values[:] = 0

        df = df.append(pandas.DataFrame.from_records([self.vehicle.to_dict(self.vehicle)])[self.features])'''


                
        # Add nearby traffic
        self.close_vehicles = self.env.road.closest_vehicles_to(self.vehicle,
                                                           self.vehicles_count - 1,
                                                           7.0 * MDPVehicle.SPEED_MAX)

        
        if self.close_vehicles:
            df = df.append(pandas.DataFrame.from_records(
                [v.to_dict(self.vehicle)
                 for v in self.close_vehicles[-self.vehicles_count + 1:]])[self.features],
                           ignore_index=True)



        # Normalize
        #df = df.iloc[1:]
        df = self.normalize(df)
        # Fill missing rows
        if df.shape[0] < self.vehicles_count+1:
            rows = -np.ones((self.vehicles_count+1 - df.shape[0], len(self.features)))
            df = df.append(pandas.DataFrame(data=rows, columns=self.features), ignore_index=True)

        for v in self.env.road.vehicles:
            v.color = GREEN if v in self.close_vehicles else None
        # Reorder
        df = df[self.features]
        # Clip
        obs = np.clip(df.values, -1, 1)
        # Flatten
        obs = np.ravel(obs)
        goal = (self.env.ROAD_LENGTH - self.vehicle.position[0]) / (7.0 * MDPVehicle.SPEED_MAX) # Normalize
        goal = min(1.0, max(-1.0, goal)) # Clip
        obs[0] = goal # Just a temporary implementation wo explicitly mentioning the goal
        '''obs_idx = 1
        for virtual_v in self.env.road.virtual_vehicles:
            obs[obs_idx] = virtual_v.position[1]/self.y_position_range
            obs_idx += 1'''
        
        if(self.env.OBS_STACK_SIZE == 1):
            return obs
        if self.observations is None:
            self.observations = deque([obs]*self.env.OBS_STACK_SIZE, maxlen=self.env.OBS_STACK_SIZE)
            return tuple(self.observations)
        else:
            #self.observations.pop(len(self.observations)-1)
            self.observations.append(obs)
            return tuple(self.observations)
        return None


class KinematicsGoalObservation(KinematicObservation):
    def __init__(self, env, ref_vehicle, scale, **kwargs):
        self.scale = scale
        self.vehicle = ref_vehicle
        super(KinematicsGoalObservation, self).__init__(env, ref_vehicle, **kwargs)

    def space(self):
        try:
            obs = self.observe()
            return spaces.Dict(dict(
                desired_goal=spaces.Box(-np.inf, np.inf, shape=obs["desired_goal"].shape, dtype=np.float32),
                achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype=np.float32),
                observation=spaces.Box(-np.inf, np.inf, shape=obs["observation"].shape, dtype=np.float32),
            ))
        except AttributeError:
            return None

    def observe(self):
        obs = np.ravel(pandas.DataFrame.from_records([self.vehicle.to_dict()])[self.features])
        goal = np.ravel(pandas.DataFrame.from_records([self.env.goal.to_dict()])[self.features])
        obs = {
            "observation": obs / self.scale,
            "achieved_goal": obs / self.scale,
            "desired_goal": goal / self.scale
        }
        return obs


def observation_factory(env, ref_vehicle, config):
    if config["type"] == "TimeToCollision":
        return TimeToCollisionObservation(env, ref_vehicle, **config)
    elif config["type"] == "Kinematics":
        return KinematicObservation(env, ref_vehicle, **config)
    elif config["type"] == "KinematicsGoal":
        return KinematicsGoalObservation(env, ref_vehicle, **config)
    else:
        raise ValueError("Unkown observation type")