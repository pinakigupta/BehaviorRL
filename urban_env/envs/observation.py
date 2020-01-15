######################################################################
#          Deep Reinforcement Learning for Autonomous Driving
#                  Created/Modified on: February 5, 2019
#                      Author:   Pinaki Gupta
#######################################################################

from __future__ import division, print_function, absolute_import
import copy
import gym
import pandas
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from collections import deque
from functools import partial
import multiprocessing
import time

from urban_env import utils
from urban_env.envs.finite_mdp import  compute_ttc_grid
from urban_env.envs.graphics import EnvViewer
from urban_env.road.lane import AbstractLane
from urban_env.vehicle.behavior import IDMVehicle
from urban_env.vehicle.control import MDPVehicle
from urban_env.vehicle.dynamics import Obstacle
from urban_env.envdict import RED, GREEN, BLUE, YELLOW, BLACK, PURPLE, DEFAULT_COLOR, EGO_COLOR, WHITE
from urban_env.utils import print_execution_time

from handle_model_files import is_predict_only

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
    FEATURES = [ 'x', 'y', 'vx', 'vy', 'psi', 'lane_psi', 'length']
    #STACK_SIZE = 2

    def __init__(self, 
                 env, 
                 ref_vehicle, 
                 features=FEATURES, 
                 relative_features=FEATURES, 
                 constraint_features=FEATURES, 
                 pedestrian_features=FEATURES,
                 obs_count=9, 
                 obs_size=10, 
                 **kwargs
                 ):
        """
        :param env: The environment to observe
        :param features: Names of features used in the observation
        :param obs_count: Number of observed vehicles
        """
        self.env = env
        self.features = features
        self.relative_features = relative_features
        self.constraint_features = constraint_features
        self.pedestrian_features = pedestrian_features
        self.obs_count = obs_count
        self.obs_size = obs_size
        self.virtual_obs_count = 1
        self.close_vehicles = None
        self.observations = None
        self.vehicle = ref_vehicle
        self.route_lane = None 


    def space(self):
        one_obs_space = spaces.Box(shape=(len(self.features) * (self.obs_size + self.obs_size),), low=-1, high=1, dtype=np.float32)
        if(self.env.config["OBS_STACK_SIZE"] == 1):
            return one_obs_space
        return spaces.Tuple(tuple([one_obs_space]*self.env.config["OBS_STACK_SIZE"]))

    def normalize(self, df):
        """
            Normalize the observation values.

            For now, assume that the road is straight along the x axis.
        :param Dataframe df: observation data
        """

        self.x_position_range = self.env.config["x_position_range"]
        self.y_position_range = self.env.config["y_position_range"]
        self.velocity_range = self.env.config["velocity_range"]


        if 'x' in df:
            df['x'] = utils.remap(df['x'], [- self.x_position_range, self.x_position_range], [-1, 1])
        if 'y' in df:
            df['y'] = utils.remap(df['y'], [-self.y_position_range, self.y_position_range], [-1, 1])
        if 'vx' in df:
            df['vx'] = utils.remap(df['vx'], [-self.velocity_range, self.velocity_range], [-1, 1])
        if 'vy' in df:
            df['vy'] = utils.remap(df['vy'], [-self.velocity_range, self.velocity_range], [-1, 1])

        

        if 'psi' in df:
            if hasattr(self.vehicle, 'route_lane_index'):
                df['psi'] = (df['psi'] - road.network.get_lane(self.vehicle.route_lane_index).heading_at(self.vehicle.position[0]))/(2*np.pi)
                for i in  range(len(df['psi'])):
                    if df['psi'][i] == -0.5:
                        df['psi'][i] = 0.5 # since the agent mostly got trained within 0 - 0.5 
            else:
                df['psi'] /= (2*np.pi)

        if 'lane_psi' in df:
            df['lane_psi'] = df['lane_psi']/(2*np.pi)
        if 'length' in df:
            df['length'] = df['length']/100
        if 'width' in df:
            df['width'] = df['width']/10            
        return df



    def observe(self):

        
        # Add ego-vehicle
        df = pandas.DataFrame.from_records([self.vehicle.to_dict(self.relative_features, self.vehicle)])[self.features]
        '''for col in df.columns:
            df[col].values[:] = 0

        df = df.append(pandas.DataFrame.from_records([self.vehicle.to_dict(self.relative_features, self.vehicle)])[self.features])'''


                
        # Add nearby traffic
        self.close_vehicles = self.env.road.closest_vehicles_to(self.vehicle,
                                                                self.obs_count - 1,
                                                                self.env.config["PERCEPTION_DISTANCE"])

        
        if self.close_vehicles:
            df = df.append(pandas.DataFrame.from_records(
                [v.to_dict(self.relative_features, self.vehicle)
                 for v in self.close_vehicles[-self.obs_count + 1:]])[self.features],
                           ignore_index=True)



        # Normalize
        #df = df.iloc[1:]
        #df = self.normalize(df)
        # Fill missing rows
        if df.shape[0] < self.obs_size+1:
            rows = -np.ones((self.obs_size+1 - df.shape[0], len(self.features)))
            df = df.append(pandas.DataFrame(data=rows, columns=self.features), ignore_index=True)

        if self.vehicle.is_ego():
            for v in self.close_vehicles:
                if (v.color==DEFAULT_COLOR) or (v.color is None):
                    v.color = GREEN
        # Reorder
        df = df[self.features]
        # Clip
        obs = np.clip(df.values, -1, 1)
        # Flatten
        obs = np.ravel(obs)

        if hasattr(self.vehicle, 'route_lane_index'):
            if self.route_lane is None:
                self.route_lane = self.env.road.network.get_lane(self.vehicle.route_lane_index)
            lane_coords = self.route_lane.local_coordinates(self.vehicle.position)
            
            target_position = self.route_lane.length if is_predict_only() else self.env.config["GOAL_LENGTH"]
            goal = (target_position - lane_coords[0]) / self.env.config["PERCEPTION_DISTANCE"] # Normalize
            goal = min(1.0, max(-1.0, goal)) # Clip
            obs[0] = goal # Just a temporary implementation wo explicitly mentioning the goal
        
        if(self.env.config["OBS_STACK_SIZE"] == 1):
            return obs
        if self.observations is None:
            self.observations = deque([obs]*self.env.config["OBS_STACK_SIZE"], maxlen=self.env.config["OBS_STACK_SIZE"])
            return tuple(self.observations)
        else:
            self.observations.append(obs)
            return tuple(self.observations)
        return None


class KinematicsGoalObservation(KinematicObservation):
    def __init__(self,
                 env,
                 ref_vehicle,
                 scale,
                 goals_count=1,
                 goals_size=1,
                 pedestrians_count=0,
                 pedestrians_size=0,
                 constraints_count=0,
                 **kwargs):
        self.scale = scale
        self.vehicle = ref_vehicle
        self.goals_count = goals_count
        self.goals_size = goals_size
        self.pedestrians_count = pedestrians_count
        self.pedestrians_size = pedestrians_size
        self.constraints_count = constraints_count
        super(KinematicsGoalObservation, self).__init__(env, ref_vehicle, **kwargs)
        self._set_closest_goals()
        self._set_closest_pedestrians()

    def space(self):
        try:
            obs = self.observe()
            return spaces.Dict(dict(
                observation=spaces.Box(-np.inf, np.inf, shape=obs["observation"].shape, dtype=np.float32),
                pedestrians=spaces.Box(-np.inf, np.inf, shape=obs["pedestrians"].shape, dtype=np.float32),
                achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype=np.float32),
                constraint=spaces.Box(-np.inf, np.inf, shape=obs["constraint"].shape, dtype=np.float32),
                desired_goal=spaces.Box(-np.inf, np.inf, shape=obs["desired_goal"].shape, dtype=np.float32),
                impatience=spaces.Box(-np.inf, np.inf, shape=obs["impatience"].shape, dtype=np.float32),
                placeholder=spaces.Box(-np.inf, np.inf, shape=obs["placeholder"].shape, dtype=np.float32),
            ))
        except AttributeError:
            return None

   

    def observation_worker(self, obsname, obs_dict):
        current_wall_time = time.time()

        if obsname is "observation":
            obs_dict.update({"observation": self.observe_vehicles()})
        elif obsname is "pedestrians":
            obs_dict.update({"pedestrians": self.observe_pedestrians()})
        elif obsname is "achieved_goal":
            obs_dict.update({"achieved_goal": self.observe_self()})
        elif obsname is "constraint":
            obs_dict.update({"constraint": self.observe_constraints()})
        elif obsname is "desired_goal":
            obs_dict.update({"desired_goal": self.observe_goals()})
        elif obsname is "impatience":
            obs_dict.update({"impatience": np.array([])})
        elif obsname is "placeholder":
            obs_dict.update({"placeholder": np.array([])})

    def observe(self):
        current_wall_time = time.time()
        manager = multiprocessing.Manager()
        #obs_dict = manager.dict()
        obs_dict = {}
        jobs = []
        '''obs_dict = {
                        "observation": self.observe_vehicles(),
                        "pedestrians": self.observe_pedestrians(),
                        "achieved_goal": self.observe_self(),
                        "constraint": self.observe_constraints(),
                        "desired_goal": self.observe_goals(),
                        "impatience": np.array([]), #min(1.0, max(0.0, self.steps/self.config["duration"]))
                        "placeholder": np.array([]),
                    }'''
        obsnames = ["observation", "pedestrians", "achieved_goal", "constraint", "desired_goal", "impatience", "placeholder"]
        for obsname in obsnames:
            self.observation_worker(obsname, obs_dict)
            #current_wall_time = print_execution_time(current_wall_time, "After " + obsname)
            '''p = multiprocessing.Process(target=self.observation_worker, args=(obsname, obs_dict))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()'''

        self.return_dict = {}
        for key, value in obs_dict.items():
            self.return_dict[key] = value

        return self.return_dict
    
    def eval_func(self, v):
        
        
        obs = np.ravel(pandas.DataFrame.from_records([self.vehicle.to_dict(self.relative_features, self.vehicle)])[self.features])
        goal = np.ravel(pandas.DataFrame.from_records([v.to_dict(self.relative_features, self.vehicle)])[self.features])
        return self.env.distance_2_goal_reward(obs, goal)

    def _set_closest_goals(self):            
        self.closest_goals = self.env.road.closest_objects_to(
                                                                vehicle=self.vehicle,
                                                                count=self.goals_count,
                                                                perception_distance=self.env.config["PERCEPTION_DISTANCE"],
                                                                objects=self.env.road.goals
                                                            ) 
        if self.vehicle.is_ego():
            for v in self.env.road.goals:
                if v in self.closest_goals:
                    v.color = WHITE
                    v.hidden = False
                else:
                    v.hidden = True      

    def _set_closest_pedestrians(self):            
        self.closest_pedestrians = self.env.road.closest_objects_to(
                                                                    vehicle=self.vehicle,
                                                                    count=self.pedestrians_count,
                                                                    perception_distance=self.env.config["PERCEPTION_DISTANCE"],
                                                                    objects=self.env.road.pedestrians
                                                                   )        

    def observe_self(self):
        current_wall_time = time.time()
        ego_obs = [self.vehicle.to_dict(self.relative_features, self.vehicle)]
        ego_obs = pandas.DataFrame.from_records(ego_obs)[self.features]
        #ego_obs = self.normalize(ego_obs)
        ego_obs = np.ravel(ego_obs)
        return ego_obs

    def observe_vehicles(self):
        return super(KinematicsGoalObservation, self).observe()

    def observe_goals(self):
        self._set_closest_goals()
        closest_obs_count = 0
        if self.closest_goals:
            goal = pandas.DataFrame.from_records([v.to_dict(self.relative_features, self.vehicle) for v in self.closest_goals])[self.features]
            #goal = self.normalize(goal)
            closest_obs_count = goal.shape[0]
        if closest_obs_count == 0:
            rows = -np.ones((self.goals_size, len(self.features)))
            goal = pandas.DataFrame(data=rows, columns=self.features)            
        # Fill missing rows
        elif closest_obs_count < self.goals_size:
            rows = -np.ones((self.goals_size -closest_obs_count, len(self.features)))
            goal = goal.append(pandas.DataFrame(data=rows, columns=self.features), ignore_index=True)
        goals = np.ravel(goal) #flatten
        return goals

    def observe_pedestrians(self):
        self._set_closest_pedestrians()
        closest_obs_count = 0
        if self.closest_pedestrians:
            peds = pandas.DataFrame.from_records([v.to_dict(self.relative_features, self.vehicle) for v in self.closest_pedestrains])\
                                                    [self.pedestrian_features]
            #peds = self.normalize(peds)
            closest_obs_count = peds.shape[0]
        # Fill missing rows
        if closest_obs_count == 0:
            rows = -np.ones((self.pedestrians_size, len(self.pedestrian_features)))
            peds = pandas.DataFrame(data=rows, columns=self.pedestrian_features)
        elif closest_obs_count < self.pedestrians_size:
            rows = -np.ones((self.pedestrians_size - closest_obs_count, len(self.pedestrian_features)))
            peds = peds.append(pandas.DataFrame(data=rows, columns=self.pedestrian_features), ignore_index=True)
        pedestrians = np.ravel(peds) #flatten
        return pedestrians

    def observe_constraints(self):
        constraint = pandas.DataFrame.from_records(
                    [v.to_dict(self.relative_features, self.vehicle) 
                                for v in self.env.road.virtual_vehicles
                                    if v not in self.env.road.goals])[self.constraint_features]
        if constraint.shape[0] < self.constraints_count:
            rows = -np.ones((self.constraints_count - constraint.shape[0], len(self.constraint_features)))
            constraint = constraint.append(pandas.DataFrame(data=rows, columns=self.constraint_features), ignore_index=True)
        current_wall_time = time.time()
        #constraint = self.normalize(constraint)
        constraint = np.ravel(constraint)
        return constraint

    def closest_vehicles(self):
        closest_to_ref = [self.vehicle]
        if self.close_vehicles:
            closest_to_ref = closest_to_ref + self.close_vehicles
        close_vehicles_dict = {
                                "observation": closest_to_ref,
                                "achieved_goal": [self.vehicle],
                                "constraint": [v for v in self.env.road.virtual_vehicles if v not in self.env.road.goals],
                                "desired_goal": self.closest_goals,
                                "pedestrians": self.closest_pedestrians,          
                              }
        return close_vehicles_dict


def observation_factory(env, ref_vehicle, config):
    if config["type"] == "TimeToCollision":
        return TimeToCollisionObservation(env, ref_vehicle, **config)
    elif config["type"] == "Kinematics":
        return KinematicObservation(env, ref_vehicle, **config)
    elif config["type"] == "KinematicsGoal":
        return KinematicsGoalObservation(env, ref_vehicle, **config)
    else:
        raise ValueError("Unkown observation type")