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
from urban_env.envs.observation import observation_factory
from urban_env.envs.finite_mdp import finite_mdp, compute_ttc_grid
from urban_env.envs.graphics import EnvViewer
from urban_env.road.lane import AbstractLane
from urban_env.vehicle.behavior import IDMVehicle
from urban_env.vehicle.control import MDPVehicle
from urban_env.vehicle.dynamics import Obstacle
from urban_env.envdict import ACTIONS_DICT
from handle_model_files import is_predict_only


class AbstractEnv(gym.Env):
    """
        A generic environment for various tasks involving a vehicle driving on a road.

        The environment contains a road populated with vehicles, and a controlled ego-vehicle that can change lane and
        velocity. The action space is fixed, but the observation space and reward function must be defined in the
        environment implementations.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    ACTIONS = ACTIONS_DICT
    
    """ Which Actions are Allowed for the current Agent """
    ACTION_MASKS = [True,True,True,True,True,True,True] 

    """
        A mapping of action indexes to action labels
    """
    ACTIONS_INDEXES = {v: k for k, v in ACTIONS.items()}
    """
        A mapping of action labels to action indexes
    """

    DEFAULT_CONFIG = {
        "observation": {
            "type": "TimeToCollision"
        },
        "DIFFICULTY_LEVELS": 2,
        "SIMULATION_FREQUENCY": 15, # The frequency at which the system dynamics are simulated [Hz]
        "POLICY_FREQUENCY": 1 , #The frequency at which the agent can take actions [Hz]
        "PERCEPTION_DISTANCE": 7.0 * MDPVehicle.SPEED_MAX, # The maximum distance of any vehicle present in the observation [m]
        "MODEL":                {
                                #    "use_lstm": True,
                                    "fcnet_hiddens": [256],
                                 },     
                     }

    BUFFER_LENGTH = 50

    _max_episode_steps = None
    #_predict_only = False

    def __init__(self, config=None):
        # Configuration
        self.config = {**self.DEFAULT_CONFIG, **config}

        # Seeding
        self.np_random = None
        self.seed()

        # Scene
        self.road = None
        self.vehicle = None
        self.close_vehicles = None

        # Spaces
        self.observation = None
        #self.define_spaces()

        # Running
        self.time = 0
        self.done = False

        # Rendering
        self.viewer = None
        self.automatic_rendering_callback = None
        self.should_update_rendering = True
        self.rendering_mode = 'human'
        self.enable_auto_render = False

        # Action , obs and reward 
        self.action = None
        self.actions = None
        self.reward = None
        self.obs = None
        self.episode_reward = 0
        self.episode_count = 0
        self.episode_reward_buffer = deque(maxlen=self.BUFFER_LENGTH)

        if 'DIFFICULTY_LEVELS' in self.config:
            self.scene_complexity = self.config['DIFFICULTY_LEVELS']


        

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def configure(self, config):
        self.config.update(config)

    def define_spaces(self):
        self.action_space = spaces.Discrete(len(self.ACTIONS))

        if "observation" not in self.config:
            raise ValueError("The observation configuration must be defined")
        self.observation = observation_factory(self, self.vehicle, self.config["observation"])
        self.observation_space = self.observation.space()
        self.observations = {v: observation_factory(self, v, self.config["observation"]) for v in self.road.vehicles}

    def is_discrete_action(self):
        is_discrete = (type(self.action_space) == gym.spaces.discrete.Discrete)
        return is_discrete

    def _reward(self, action):
        """
            Return the reward associated with performing a given action and ending up in the current state.

        :param action: the last action performed
        :return: the reward
        """
        raise NotImplementedError()

    def _info(self):
        return {}


    def _is_terminal(self):
        """
            Check whether the current state is a terminal state
        :return:is the state terminal
        """
        raise NotImplementedError()

    def _constraint(self, action):
        """
            A constraint metric, for budgeted MDP.

            If a constraint is defined, it must be used with an alternate reward that doesn't contain it
            as a penalty.
        :param action: the last action performed
        :return: the constraint signal, the alternate (constraint-free) reward
        """
        return None, self._reward(action)

    def reset(self):
        """
            Reset the environment to it's initial configuration
        :return: the observation of the reset state
        """
        self.episode_reward = 0
        self.define_spaces()
        obs = self.observation.observe()
        #print("resetting env", "current_curriculam = ", self.get_curriculam())
        return obs

    def step(self, action):
        """
            Perform an action and step the environment dynamics.

            The action is executed by the ego-vehicle, and all other vehicles on the road performs their default
            behaviour for several simulation timesteps until the next decision making step.
        :param int action: the action performed by the ego-vehicle
        :return: a tuple (observation, reward, terminal, info)
        """
        if self.road is None or self.vehicle is None:
            raise NotImplementedError("The road and vehicle must be initialized in the environment implementation")

        if is_predict_only() and not self.DEFAULT_CONFIG["_predict_only"]:
            self.DEFAULT_CONFIG["_predict_only"] = True
            self.reset()

        self._simulate(action)
        

        obs = self.observation.observe()
        self.obs = obs
        reward = self._reward(action)
        self.action = action
        self.reward = reward
        self.episode_reward += self.reward
        terminal = self._is_terminal()

        if terminal:
            self.episode_reward_buffer.append(self.episode_reward)
            self.episode_count +=1
            #self._set_curriculam(curriculam_reward_threshold=1500)
            self._set_curriculam(curriculam_reward_threshold=0.5)

        self.close_vehicles = self.observation.close_vehicles
        constraint = self._constraint(action)
        info = {'constraint': constraint, "c_": constraint}
        info = {**info, **self._info() }
        #print("self.steps ", self.steps, " obs ", obs)

        return obs, reward, terminal, info


    def _simulate(self, action=None):
        """
            Perform several steps of simulation with constant action
        """
        SCALE = 1.0
        if '_predict_only' in self.config:
            if not self.config['_predict_only']:
                SCALE = 1.4

        for k in range(int(self.config["SIMULATION_FREQUENCY"] // self.config["POLICY_FREQUENCY"])):
            if action is not None : # and self.time % int(self.SIMULATION_FREQUENCY // self.POLICY_FREQUENCY) == 0:
                # Forward action to the vehicle                    
                if type(action) is not dict: 
                    self.vehicle.act(self.ACTIONS[action])
                else:
                    self.vehicle.act(action)

            self.road.act(self.observations)
            self.road.step(1 / self.config["SIMULATION_FREQUENCY"], SCALE)
            self.time += 1

            # Automatically render intermediate simulation steps if a viewer has been launched
            self._automatic_rendering()

            # Stop at terminal states
            #if self.done or self._is_terminal():
                #print("self.done", self.done, " _is_terminal ", self._is_terminal())
                #break
        self.enable_auto_render = False

        

    def render(self, mode='human'):
        """
            Render the environment.

            Create a viewer if none exists, and use it to render an image.
        :param mode: the rendering mode
        """
        self.rendering_mode = mode

        if self.viewer is None:
            self.viewer = EnvViewer(self)

        self.enable_auto_render = True

        # If the frame has already been rendered, do nothing
        if self.should_update_rendering:
            self.viewer.display()

        if mode == 'rgb_array':
            image = self.viewer.get_image()
            self.viewer.handle_events()
            return image
        elif mode == 'human':
            self.viewer.handle_events()
        self.should_update_rendering = False

    def close(self):
        """
            Close the environment.

            Will close the environment viewer if it exists.
        """
        self.done = True
        if self.viewer is not None:
            self.viewer.close()
        self.viewer = None
        

    def get_available_actions(self):
        """
            Get the list of currently available actions.

            Lane changes are not available on the boundary of the road, and velocity changes are not available at
            maximal or minimal velocity.

        :return: the list of available actions
        """
        actions = [self.ACTIONS_INDEXES['IDLE']]
        for l_index in self.road.network.side_lanes(self.vehicle.lane_index):
            if l_index[2] < self.vehicle.lane_index[2] \
                    and self.road.network.get_lane(l_index).is_reachable_from(self.vehicle.position):
                actions.append(self.ACTIONS_INDEXES['LANE_LEFT'])
                actions.append(self.ACTIONS_INDEXES['LANE_LEFT_AGGRESSIVE'])
            if l_index[2] > self.vehicle.lane_index[2] \
                    and self.road.network.get_lane(l_index).is_reachable_from(self.vehicle.position):
                actions.append(self.ACTIONS_INDEXES['LANE_RIGHT'])
                actions.append(self.ACTIONS_INDEXES['LANE_RIGHT_AGGRESSIVE'])
        if self.vehicle.velocity_index < self.vehicle.SPEED_COUNT - 1:
            actions.append(self.ACTIONS_INDEXES['FASTER'])
        if self.vehicle.velocity_index > 0:
            actions.append(self.ACTIONS_INDEXES['SLOWER'])
        return actions

    def _automatic_rendering(self):
        """
            Automatically render the intermediate frames while an action is still ongoing.
            This allows to render the whole video and not only single steps corresponding to agent decision-making.

            If a callback has been set, use it to perform the rendering. This is useful for the environment wrappers
            such as video-recording monitor that need to access these intermediate renderings.
        """
        if self.viewer is not None and self.enable_auto_render:
            self.should_update_rendering = True

            if self.automatic_rendering_callback:
                self.automatic_rendering_callback()
            else:
                self.render(self.rendering_mode)

    def simplify(self):
        """
            Return a simplified copy of the environment where distant vehicles have been removed from the road.

            This is meant to lower the policy computational load while preserving the optimal actions set.

        :return: a simplified environment state
        """
        state_copy = copy.deepcopy(self)
        state_copy.road.vehicles = [state_copy.vehicle] + state_copy.road.close_vehicles_to(
            state_copy.vehicle, [-self.config["PERCEPTION_DISTANCE"] / 2, self.config["PERCEPTION_DISTANCE"]])

        return state_copy

    def change_vehicles(self, vehicle_class_path):
        """
            Change the type of all vehicles on the road
        :param vehicle_class_path: The path of the class of behavior for other vehicles
                             Example: "urban_env.vehicle.behavior.IDMVehicle"
        :return: a new environment with modified behavior model for other vehicles
        """
        vehicle_class = utils.class_from_path(vehicle_class_path)

        env_copy = copy.deepcopy(self)
        vehicles = env_copy.road.vehicles
        for i, v in enumerate(vehicles):
            if v is not env_copy.vehicle and not isinstance(v, Obstacle):
                vehicles[i] = vehicle_class.create_from(v)
        return env_copy

    def set_preferred_lane(self, preferred_lane=None):
        env_copy = copy.deepcopy(self)
        if preferred_lane:
            for v in env_copy.road.vehicles:
                if isinstance(v, IDMVehicle):
                    raise NotImplementedError()
                    # Vehicle with lane preference are also less cautious
                    v.LANE_CHANGE_MAX_BRAKING_IMPOSED = 1000
        return env_copy

    def set_route_at_intersection(self, _to):
        env_copy = copy.deepcopy(self)
        for v in env_copy.road.vehicles:
            if isinstance(v, IDMVehicle):
                v.set_route_at_intersection(_to)
        return env_copy

    def randomize_behaviour(self):
        env_copy = copy.deepcopy(self)
        for v in env_copy.road.vehicles:
            if isinstance(v, IDMVehicle):
                v.randomize_behavior()
        return env_copy

    def to_finite_mdp(self):
        return finite_mdp(self, time_quantization=1/self.POLICY_FREQUENCY)

    def __deepcopy__(self, memo):
        """
            Perform a deep copy but without copying the environment viewer.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k not in ['viewer', 'automatic_rendering_callback']:
                setattr(result, k, copy.deepcopy(v, memo))
            else:
                setattr(result, k, None)
        return result

    def set_config(self, key, value):
        self.config[key] = value

    def set_actions(self, actions):
        self.actions = actions

    def set_curriculam(self, value):
        self.set_config("DIFFICULTY_LEVELS", value)

    def get_curriculam(self):
        return self.config["DIFFICULTY_LEVELS"]
    
    def _set_curriculam(self, curriculam_reward_threshold):
        from color import color
        import sys

        sys.stdout.flush()
        
        def length_(buffer):
            buffer_length = 0
            for elem in buffer:
                buffer_length += 1
            return buffer_length

        def mean_(buffer):
            buffer_length = 0
            buffer_sum = 0
            for elem in buffer:
                buffer_length += 1
                buffer_sum += elem
            return (buffer_sum/buffer_length)        

        episode_buffer_length = length_(self.episode_reward_buffer)

        if(episode_buffer_length==self.BUFFER_LENGTH):
            mean_episode_reward = mean_(self.episode_reward_buffer)
            current_curriculam = self.get_curriculam()
            if mean_episode_reward > curriculam_reward_threshold:
                self.episode_reward_buffer.clear()
                new_curriculam = current_curriculam+1
                self.set_curriculam(new_curriculam)
                print(color.BOLD + 'updating curriculam to ' + str(new_curriculam) + 
                " as mean episode_reward is " + str(mean_episode_reward) + color.END)
                self.reset()
            '''else:
                print(self.episode_count,':', mean_episode_reward, end='\t')
                #print("mean_episode_reward ", mean_episode_reward, 
                #"current_curriculam ", current_curriculam, " episode_buffer_length ", episode_buffer_length)
        else :
             print(self.episode_count,'::',episode_buffer_length, end='\t')'''
