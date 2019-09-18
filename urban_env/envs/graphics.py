######################################################################
#          Deep Reinforcement Learning for Autonomous Driving
#                  Created/Modified on: February 5, 2019
#                      Author: Munir Jojo-Verge, Pinaki Gupta
#######################################################################

from __future__ import division, print_function, absolute_import
from queue import *
import os

import numpy as np
import pygame

from urban_env.road.graphics import WorldSurface, RoadGraphics
from urban_env.vehicle.graphics import VehicleGraphics
from urban_env.envdict import ACTIONS_DICT

class EnvViewer(object):
    """
        A viewer to render a urban driving environment.
    """
    SCREEN_WIDTH = 1750
    SCREEN_HEIGHT = 150
    SAVE_IMAGES = False

    def __init__(self, env):
        self.env = env

        pygame.init()
        panel_size = (self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
        self.screen = pygame.display.set_mode([self.SCREEN_WIDTH, self.SCREEN_HEIGHT])
        self.sim_surface = WorldSurface(panel_size, 0, pygame.Surface(panel_size))
        self.sim_surface.centering_position = env.config.get("centering_position", self.sim_surface.INITIAL_CENTERING)
        self.clock = pygame.time.Clock()

        self.enabled = True
        if "SDL_VIDEODRIVER" in os.environ and os.environ["SDL_VIDEODRIVER"] == "dummy":
            self.enabled = False

        self.agent_display = None
        self.agent_surface = None
        self.vehicle_trajectory = None
        self.vehicle_trajectories = []
        self.frame = 0

    def set_agent_display(self, agent_display):
        """
            Set a display callback provided by an agent, so that they can render their behaviour on a dedicated
            agent surface, or even on the simulation surface.
        :param agent_display: a callback provided by the agent to display on surfaces
        """
        if self.agent_display is None:
            if self.SCREEN_WIDTH > self.SCREEN_HEIGHT:
                self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, 2 * self.SCREEN_HEIGHT))
            else:
                self.screen = pygame.display.set_mode((2 * self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            self.agent_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.agent_display = agent_display

    def set_agent_action_sequence(self, actions):
        """
            Set the sequence of actions chosen by the agent, so that it can be displayed
        :param actions: list of action, following the env's action space specification
        """
        from multiprocessing import Process, Pool, Manager
        if hasattr(self.env.action_space, 'n'):
            actions = [self.env.ACTIONS[a] for a in actions]
        self.vehicle_trajectories.clear()
        self.vehicle_trajectories = \
                [self.env.vehicle.predict_trajectory(actions=actions,
                                                     action_duration=1/self.env.POLICY_FREQUENCY,
                                                     trajectory_timestep=1/3/self.env.POLICY_FREQUENCY,
                                                     dt=1/self.env.SIMULATION_FREQUENCY)]

        '''for v in self.env.road.closest_vehicles_to(self.env.vehicle, 4):
            if v not in self.env.road.virtual_vehicles:
                self.vehicle_trajectories.append\
                    (Process(target=v.predict_trajectory,
                             args=(actions,
                                   1/self.env.POLICY_FREQUENCY,
                                   1/1/self.env.POLICY_FREQUENCY,
                                   1/self.env.SIMULATION_FREQUENCY,
                                   2)
                            )
                    )'''
        
        out_q =  Manager().list()
        p = [Process(target=v.predict_trajectory,
                      args=(actions,
                            1/self.env.POLICY_FREQUENCY,
                            1/1/self.env.POLICY_FREQUENCY,
                            1/self.env.SIMULATION_FREQUENCY,
                            out_q,
                            2)
                            ) for v in self.env.road.closest_vehicles_to(self.env.vehicle, 4) \
                                if v not in self.env.road.virtual_vehicles]

        if p:
            for process in p:
                process.start()

            #self.vehicle_trajectories = out_q.get()

            for process in p:
                process.join()
        
            for trajectory in out_q:
                self.vehicle_trajectories.append(trajectory)

            


    def handle_events(self):
        """
            Handle pygame events by forwarding them to the display and environment vehicle.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.env.close()
            self.sim_surface.handle_event(event)
            if self.env.vehicle:
                VehicleGraphics.handle_event(self.env.vehicle, event)

    def display(self):
        """
            Display the road and vehicles on a pygame window.
        """
        if not self.enabled:
            return

        if self.env.actions is not None:
            if self.env.actions:
                self.set_agent_action_sequence(self.env.actions)

        self.sim_surface.move_display_window_to(self.window_position())
        RoadGraphics.display(self.env.road, self.sim_surface)
        if self.vehicle_trajectories:
            for vehicle_trajectory in self.vehicle_trajectories:
                VehicleGraphics.display_trajectory(
                    vehicle_trajectory,
                    self.sim_surface)
        RoadGraphics.display_traffic(self.env.road, self.sim_surface)

        if self.agent_display:
            self.agent_display(self.agent_surface, self.sim_surface)
            if self.SCREEN_WIDTH > self.SCREEN_HEIGHT:
                self.screen.blit(self.agent_surface, (0, self.SCREEN_HEIGHT))
            else:
                self.screen.blit(self.agent_surface, (self.SCREEN_WIDTH, 0))

        self.screen.blit(self.sim_surface, (0, 0))
        self.clock.tick(self.env.SIMULATION_FREQUENCY)
        pygame.display.flip()

        if self.SAVE_IMAGES:
            pygame.image.save(self.screen, "urban-env_{}.png".format(self.frame))
            self.frame += 1

        caption = "Urban-AD ( "
        caption += "action = " + str(ACTIONS_DICT[self.env.previous_action])
        caption += " accel = " + str(round(self.env.vehicle.control_action['acceleration']))
        caption += " steering = " + str(round(self.env.vehicle.control_action['steering']))
        caption += " steps = " + str(self.env.steps)
        if self.env.episode_travel:
            caption += ', ep travel  = {:.2f}'.format(self.env.episode_travel)
        caption += ', reward  = {:.2f}'.format(self.env.reward)  
        caption += ', ep reward  = {:.2f}'.format(self.env.episode_reward)
        caption += " )"
        pygame.display.set_caption(caption)

    def get_image(self):
        """
        :return: the rendered image as a rbg array
        """
        data = pygame.surfarray.array3d(self.screen)
        return np.moveaxis(data, 0, 1)

    def window_position(self):
        """
        :return: the world position of the center of the displayed window.
        """
        if self.env.vehicle:
            return self.env.vehicle.position
        else:
            return np.array([0, 0])

    def close(self):
        """
            Close the pygame window.
        """
        pygame.quit()

