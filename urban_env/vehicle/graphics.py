######################################################################
#          Deep Reinforcement Learning for Autonomous Driving
#                  Created/Modified on: February 5, 2019
#                      Author: Munir Jojo-Verge, Pinaki Gupta
#######################################################################

from __future__ import division, print_function
import numpy as np
import pygame

from urban_env.vehicle.dynamics import Vehicle, Obstacle
from urban_env.vehicle.control import ControlledVehicle, MDPVehicle, IDMDPVehicle
from urban_env.vehicle.behavior import IDMVehicle, LinearVehicle
from urban_env.envdict import RED, GREEN, BLUE, YELLOW, BLACK, PURPLE, DEFAULT_COLOR, EGO_COLOR, WHITE, GREY

class VehicleGraphics(object):


    @classmethod
    def display(cls, vehicle, surface, mdp_vehicle=None, transparent=False, virtual=False):
        """
            Display a vehicle on a pygame surface.

            The vehicle is represented as a colored rotated rectangle.

        :param vehicle: the vehicle to be drawn
        :param surface: the surface to draw the vehicle on
        :param transparent: whether the vehicle should be drawn slightly transparent
        """
        v = vehicle        

        #######################################################
        # if isinstance(v, Obstacle):
        #     length = v.LENGTH
        #     width  = v.WIDTH
        #     s = pygame.Surface((surface.pix(length), surface.pix(length)), pygame.SRCALPHA)  # per-pixel alpha
        #     rect = (0, surface.pix(length) / 2 - surface.pix(width) / 2, surface.pix(length), surface.pix(width))        
        #     pygame.draw.rect(s, cls.get_color(v, transparent), rect, 0)        
        #     pygame.draw.rect(s, BLACK, rect, 1)                  
            
        #     x = v.position[0] - length / 2
        #     y = v.position[1] - width / 2            

        # elif isinstance(v, MDPVehicle):
        #     length = v.LENGTH * 1.5
        #     width  = v.WIDTH  * 1.5
        #     s = pygame.image.load('./urban_env/vehicle/img/ego_3.png')
        #     s = pygame.transform.scale(s, (surface.pix(length),surface.pix(width)))
        #     #cls.display_Ego_params(v, surface, transparent)
        #     x = v.position[0] - length / 2
        #     y = v.position[1] - width / 2
        # else:
        #     length = v.LENGTH * 1.5
        #     width  = v.WIDTH  * 1.5transparent
        #     s = pygame.image.load('./urban_env/vehicle/img/car_3.png')
        #     s = pygame.transform.scale(s, (surface.pix(length),surface.pix(width)))
        #     x = v.position[0] + length / 2
        #     y = v.position[1] - width
        #######################################################
        veh_length = min(v.LENGTH, 200)
        veh_width =  v.WIDTH
        s = pygame.Surface((surface.pix(veh_length), surface.pix(veh_length)), pygame.SRCALPHA)  # per-pixel alpha
        rect = (0, surface.pix(veh_length) / 2 - surface.pix(veh_width) / 2, surface.pix(veh_length), surface.pix(veh_width))
        if not v.virtual:       
            pygame.draw.rect(s, cls.get_color(v, transparent), rect, 0)
        pygame.draw.rect(s, BLACK, rect, 1)

        s = pygame.Surface.convert_alpha(s)
        h = v.heading if abs(v.heading) > 2 * np.pi / 180 else 0
        sr = pygame.transform.rotate(s, -h * 180 / np.pi)
        #if not v.virtual:
        surface.blit(sr, (surface.pos2pix(v.position[0] - veh_length / 2, v.position[1] - veh_length / 2)))
        #else:
        #    surface.blit(sr, (surface.pos2pix(mdp_vehicle.position[0] - veh_length / 2, v.position[1] - veh_length / 2)))

        font_type = 'freesansbold.ttf'
        size = 12
        color = WHITE
        font = pygame.font.Font(font_type, size) 

        if not(v.virtual or virtual) or True:
            text = ' '*10+v.Id()
            text = font.render(text, False, color) 
            textRect = text.get_rect()
            textRect.center = (surface.pos2pix(v.position[0], v.position[1]))
            text = pygame.transform.rotate(text, -h * 180 / np.pi)
            surface.blit(text, textRect)
        #print("Unable to render text", text)
                

    @classmethod
    def display_Ego_params(cls, vehicle, surface, transparent=False):
        """
            Display Ego vehicle's parameter on a pygame surface.            

        :param vehicle: the vehicle to be drawn
        :param surface: the surface to draw the vehicle on
        :param transparent: whether the vehicle should be drawn slightly transparent
        """
        font_type = 'Ubuntu'
        size = 16
        color = YELLOW
        next_line_step = -20
        ini_line = surface.get_height() + next_line_step
        
        ### Velocity
        line = 0
        text = 'Ego Velocity (m/s): {:.3f}'.format(vehicle.velocity)
        font = pygame.font.SysFont(font_type, size, bold=True)
        text = font.render(text, True, color)
        surface.blit(text, (0, ini_line + line*next_line_step))

        ### Target Velocity
        line += 1
        text = 'Target Velocity (m/s): {:.3f}'.format(vehicle.target_velocity)
        font = pygame.font.SysFont(font_type, size, bold=True)
        text = font.render(text, True, color)
        surface.blit(text, (0, ini_line + line*next_line_step))

        ### Distance to Goal
        line += 1
        text = 'Distance to Goal (m): {:.3f}'.format(vehicle.distance_to_goal)
        font = pygame.font.SysFont(font_type, size, bold=True)
        text = font.render(text, True, color)
        surface.blit(text, (0, ini_line + line*next_line_step))        

        ### On Road        
        line += 1
        text = 'On Road: {}'.format(vehicle.is_on_the_road)
        font = pygame.font.SysFont(font_type, size, bold=True)
        tmp_color = color
        if not vehicle.is_on_the_road:            
            color = RED
        text = font.render(text, True, color)
        color = tmp_color

        surface.blit(text, (0, ini_line + line*next_line_step))

        ### High Velocity Reward        
        line += 1
        text = 'High Velocity Reward: {:.3f}'.format(vehicle.high_vel_reward)
        font = pygame.font.SysFont(font_type, size, bold=True) 
        tmp_color = color
        if vehicle.high_vel_reward < 0:            
            color = RED
        text = font.render(text, True, color)
        color = tmp_color

        surface.blit(text, (0, ini_line + line*next_line_step))

        ### Reverse Reward        
        line += 1
        text = 'Reverse Reward: {:.3f}'.format(vehicle.reverse_reward)
        font = pygame.font.SysFont(font_type, size, bold=True)
        if vehicle.reverse_reward < 0:            
            color = RED
        text = font.render(text, True, color)
        color = tmp_color        
        surface.blit(text, (0, ini_line + line*next_line_step))

        ### Lane Change Reward        
        # line += 1
        # text = 'Lane Change Reward: {:.3f}'.format(vehicle.lane_change_reward)
        # font = pygame.font.SysFont(font_type, size, bold=True)
        # if vehicle.lane_change_reward < 0:            
        #     color = RED
        # text = font.render(text, True, color)
        # color = tmp_color

        # surface.blit(text, (0, ini_line + line*next_line_step))

        ### Off-Road Reward        
        line += 1
        text = 'Off-Road Reward: {:.3f}'.format(vehicle.off_road_reward)
        font = pygame.font.SysFont(font_type, size, bold=True)
        if vehicle.off_road_reward < 0:            
            color = RED
        text = font.render(text, True, color)
        color = tmp_color

        surface.blit(text, (0, ini_line + line*next_line_step))

        ### Distance To Goal Reward        
        line += 1
        text = 'Distance To Goal Reward: {:.3f}'.format(vehicle.distance_goal_reward)
        font = pygame.font.SysFont(font_type, size, bold=True)
        if vehicle.distance_goal_reward < 0:            
            color = RED
        text = font.render(text, True, color)
        color = tmp_color

        surface.blit(text, (0, ini_line + line*next_line_step))

        ### reward_wrt_initial_dist_to_goal
        line += 1
        text = 'reward_wrt_initial_dist_to_goal: {:.3f}'.format(vehicle.reward_wrt_initial_dist_to_goal)
        font = pygame.font.SysFont(font_type, size, bold=True)
        text = font.render(text, True, color)
        surface.blit(text, (0, ini_line + line*next_line_step))

        ### heading_towards_goal_reward
        line += 1
        text = 'heading_towards_goal_reward: {:.3f}'.format(vehicle.heading_towards_goal_reward)
        font = pygame.font.SysFont(font_type, size, bold=True)
        text = font.render(text, True, color)
        surface.blit(text, (0, ini_line + line*next_line_step))       

        ### Total Reward (per action)        
        line += 1
        text = 'Total Reward: {:.3f}'.format(vehicle.reward)
        font = pygame.font.SysFont(font_type, size, bold=True)
        if vehicle.reward < 0:            
            color = RED
        text = font.render(text, True, color)
        color = tmp_color

        surface.blit(text, (0, ini_line + line*next_line_step))        

        ### Ego Position        
        line += 1
        text = 'Ego Pos (x,y): {:.3f}, {:.3f}'.format(vehicle.curr_position[0], vehicle.curr_position[1])
        font = pygame.font.SysFont(font_type, size, bold=True)
        text = font.render(text, True, color)
        color = tmp_color

        surface.blit(text, (0, ini_line + line*next_line_step))        
        
        ### Ego Actions        
        line += 1
        text = 'Ego Actions (St,Acc): {:.3f}, {:.3f}'.format(vehicle.action['steering'], vehicle.action['acceleration'])
        font = pygame.font.SysFont(font_type, size, bold=True)
        text = font.render(text, True, color)
        color = tmp_color

        surface.blit(text, (0, ini_line + line*next_line_step))        

        # Display EGO Goal                    
        pix_pos= surface.pos2pix(vehicle.goal_state[0], vehicle.goal_state[1])
        radius = abs(pix_pos[0] - surface.pos2pix(vehicle.goal_state[0]+3, vehicle.goal_state[1])[0])
        pygame.draw.circle(surface, GREEN, pix_pos, radius)

        

    @classmethod
    def display_trajectory(cls, states, surface):
        """
            Display the whole trajectory of a vehicle on a pygame surface.

        :param states: the list of vehicle states within the trajectory to be displayed
        :param surface: the surface to draw the vehicle future states on
        """
        transparent=2
        for vehicle in states:
            cls.display(vehicle, surface, transparent=transparent, virtual=True)
            transparent*=0.2

    @classmethod
    def get_color(cls, vehicle, transparent=False):
        color = DEFAULT_COLOR
        if vehicle.crashed:
            color = RED

        if vehicle.is_ego():
            color = EGO_COLOR
        elif isinstance(vehicle, IDMDPVehicle):
            color = YELLOW
        elif vehicle.color is not None:
            color = vehicle.color
        elif isinstance(vehicle, IDMVehicle):
            color = BLUE
        elif isinstance(vehicle, Obstacle):
            color = GREY

        if transparent:
            color = (color[0], color[1], color[2], transparent*50)

        return color

    @classmethod
    def handle_event(cls, vehicle, event):
        """
            Handle a pygame event depending on the vehicle type

        :param vehicle: the vehicle receiving the event
        :param event: the pygame event
        """
        if isinstance(vehicle, ControlledVehicle):
            cls.control_event(vehicle, event)
        elif isinstance(vehicle, Vehicle):
            cls.dynamics_event(vehicle, event)

    @classmethod
    def control_event(cls, vehicle, event):
        """
            Map the pygame keyboard events to control decisions

        :param vehicle: the vehicle receiving the event
        :param event: the pygame event
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT:
                vehicle.act("FASTER")
            if event.key == pygame.K_LEFT:
                vehicle.act("SLOWER")
            if event.key == pygame.K_DOWN:
                vehicle.act("LANE_RIGHT")
            if event.key == pygame.K_UP:
                vehicle.act("LANE_LEFT")

    @classmethod
    def dynamics_event(cls, vehicle, event):
        """
            Map the pygame keyboard events to dynamics actuation

        :param vehicle: the vehicle receiving the event
        :param event: the pygame event
        """
        action = vehicle.action.copy()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT:
                action['steering'] = 45 * np.pi / 180
            if event.key == pygame.K_LEFT:
                action['steering'] = -45 * np.pi / 180
            if event.key == pygame.K_DOWN:
                action['acceleration'] = -6
            if event.key == pygame.K_UP:
                action['acceleration'] = 5
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_RIGHT:
                action['steering'] = 0
            if event.key == pygame.K_LEFT:
                action['steering'] = 0
            if event.key == pygame.K_DOWN:
                action['acceleration'] = 0
            if event.key == pygame.K_UP:
                action['acceleration'] = 0
        if action != vehicle.action:
            vehicle.act(action)
