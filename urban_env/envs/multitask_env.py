from __future__ import division, print_function, absolute_import
import numpy as np
from gym import logger
import gym

from urban_env import utils
from urban_env.envs.abstract import AbstractEnv
from urban_env.envs.two_way_env import TwoWayEnv
from urban_env.envs.graphics import EnvViewer

import random


class MultiTaskEnv(AbstractEnv):
    def __init__(self, env_config=None):
        # pick actual env based on worker and env indexes
        self.env = gym.make(random.choice(['multilane-v0', 'two-way-v0']))
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
    def reset(self):
        return self.env.reset()
    def step(self, action):
        return self.env.step(action)
    def render(self, mode='human'):
        return self.env.render(mode)
    def close(self):
        return self.env.close() 