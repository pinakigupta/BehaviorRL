from __future__ import division, print_function, absolute_import
import numpy as np
from gym import logger
import gym
import numpy as np

from urban_env import utils
from urban_env.envs.abstract import AbstractEnv
from urban_env.envs.two_way_env import TwoWayEnv
from urban_env.envs.graphics import EnvViewer


import random



class MultiTaskEnv(AbstractEnv):
    ENV_LIST = ['multilane-v0', 'two-way-v0']
    def __init__(self, env_config=None):
        # pick actual env based on worker and env indexes
        self.reset()
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
    def reset(self):
        self.env = gym.make(MultiTaskEnv.random_env())
        return self.env.reset()
    def step(self, action):
        step_return = self.env.step(action)
    #    self.print_obs_space()
        return step_return
    def render(self, mode='human'):
        return self.env.render(mode)
    def close(self):
        return self.env.close()
    def _is_terminal(self):
        return self.env._is_terminal()
    def _reward(self, action):
        return self.env._reward(action)
    def print_obs_space(self):
        return self.env.print_obs_space()
    @classmethod
    def random_env(cls):
        return np.random.choice(cls.ENV_LIST)