######################################################################
#          Deep Reinforcement Learning for Autonomous Driving
#                  Created/Modified on: February 5, 2019
#                      Author:   Pinaki Gupta
#######################################################################

from __future__ import division, print_function
import abc
import numpy as np
import pandas as pd
import copy
from time import time
from collections import deque
import ray

import gym

from urban_env import utils
from urban_env.logger import Loggable


def unwrap_vehicle_step(v, dt):
    return v.step(dt)


class Vehicle(Loggable):
    """
        A moving vehicle on a road, and its dynamics.

        The vehicle is represented by a dynamical system: a modified bicycle model.
        It's state is propagated depending on its steering and acceleration actions.
    """


    DEFAULT_CONFIG = {**{
                      "DEFAULT_LENGTH": 5.0 , # Vehicle length [m]
                      "DEFAULT_WIDTH": 2.0 , #  Vehicle width [m] 
                      "COLLISIONS_ENABLED": True, # Enable collision detection between vehicles
                      "DEFAULT_VELOCITIES": [23, 25], #Range for random initial velocities [m/s]
                      "MAX_VELOCITY": 40, #Maximum reachable velocity [m/s]
                      },
                      **{
                            'front_edge_to_center': 3.898,
                            'back_edge_to_center': 0.853,
                            'left_edge_to_center': 0.9655,
                            'right_edge_to_center': 0.9655,

                            'length': 4.751,
                            'width': 1.931,
                            'height': 1.655,
                            'min_turn_radius': 4.63,
                            'max_acceleration': 2.94,
                            'max_deceleration': -6.0,
                            'max_steer_angle': 0.58904875,
                            'max_steer_angle_rate': 8.3733,
                            'min_steer_angle_rate': 0,
                            'steer_ratio': 16.00,
                            'wheel_base': 2.95,
                            'wheel_rolling_radius': 0.335,
                            'max_abs_speed_when_stopped': 0.1
                        }
                    }


    def __init__(self, 
                 road, 
                 position, 
                 lane_index=None, 
                 heading=0, 
                 velocity=0, 
                 length=5.0,
                 width=2.0, 
                 virtual=False, 
                 color=None, 
                 config=DEFAULT_CONFIG, 
                 **kwargs
                 ):
        self.config = {**self.DEFAULT_CONFIG, **config}
        self.LENGTH = length
        self.WIDTH = width
        self.road = road
        self.position = np.array(position).astype('float')
        self.heading = heading
        self.velocity = velocity
        self.acceleration = 0.0
        self.jerk = 0.0
        self.color = color
        if lane_index is None:
            self.lane_index = self.road.network.get_closest_lane_index(self.position, self.heading) if self.road else np.nan
        else:
            self.lane_index = lane_index
        self.lane = self.road.network.get_lane(self.lane_index) if self.road else None
        self.action = {'steering': 0, 'acceleration': 0}
        self.reverse = False
        self.PRNDL = "D"
        self.braking = None
        self.throttle = None
        self.action_validity = True
        self.crashed = False
        self.log = []
        self.virtual = virtual
        self.is_projection = False
        self.is_ego_vehicle = False
        self.control_action = None
        self.hidden = False
        self.LGAgent = None
        self.projection = []
        self.projection_length = None
        self.steps = 0

    
    def is_ego(self):
        return self.is_ego_vehicle

    @classmethod
    def make_on_lane(cls, road, lane_index, longitudinal, velocity=0):
        """
            Create a vehicle on a given lane at a longitudinal position.

        :param road: the road where the vehicle is driving
        :param lane_index: index of the lane where the vehicle is located
        :param longitudinal: longitudinal position along the lane
        :param velocity: initial velocity in [m/s]
        :return: A vehicle with at the specified position
        """
        lane = road.network.get_lane(lane_index)
        return cls(
                   road=road,
                   position=lane.position(longitudinal, 0),
                   heading=lane.heading_at(longitudinal),
                   velocity=velocity
                   )

    @classmethod
    def create_random(cls, road, velocity=None, spacing=1, ahead=True, config=None, **kwargs):
        """
            Create a random vehicle on the road.

            The lane and /or velocity are chosen randomly, while longitudinal position is chosen behind the last
            vehicle in the road with density based on the number of lanes.

        :param road: the road where the vehicle is driving
        :param velocity: initial velocity in [m/s]. If None, will be chosen randomly
        :param spacing: ratio of spacing to the front vehicle, 1 being the default
        :return: A vehicle with random position and/or velocity
        """
        if velocity is None:
            velocity = road.np_random.uniform(
                Vehicle.DEFAULT_CONFIG["DEFAULT_VELOCITIES"][0], Vehicle.DEFAULT_CONFIG["DEFAULT_VELOCITIES"][1])
        default_spacing = 1.5*velocity
        _from = road.np_random.choice(list(road.network.graph.keys()))
        _to = road.np_random.choice(list(road.network.graph[_from].keys()))
        _id = road.np_random.choice(len(road.network.graph[_from][_to]))
        offset = spacing * default_spacing * np.exp(-5 / 30 * len(road.network.graph[_from][_to]))
        if ahead:
            x0 = np.max([v.position[0] for v in road.vehicles]
                    ) if len(road.vehicles) else 3*offset
            x0+=offset * road.np_random.uniform(0.9, 1.1)
        else:
            x0 = np.min([v.position[0] for v in road.vehicles]
                    ) if len(road.vehicles) else 3*offset
            x0-=offset * road.np_random.uniform(0.9, 1.1)
        
        #x0=x0+delta_x0
        v = cls(
                road=road,
                position=road.network.get_lane((_from, _to, _id)).position(x0, 0),
                heading=0,
                velocity=velocity,
                config=config
                )
        return v

    @classmethod
    def create_from(cls,
                    vehicle
                   ):
        """
            Create a new vehicle from an existing one.
            Only the vehicle dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        v = cls(vehicle.road, vehicle.position,
                vehicle.heading, vehicle.velocity)
        return v


    @staticmethod
    def update_from_v2v(from_vehicle,
                        to_vehicle,
                       ):
        """
            Create a new vehicle from an existing one.
            Only the vehicle dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        #to_vehicle.road = from_vehicle.road
        to_vehicle.position = copy.deepcopy(from_vehicle.position)
        to_vehicle.heading = from_vehicle.heading
        to_vehicle.velocity = from_vehicle.velocity


    def act(self, action=None, **kwargs):
        """
            Store an action to be repeated.

        :param action: the input action
        """
        if action is not None:
            self.action = action
        elif self.control_action is not None:
            self.action = self.control_action
        
        throttle_brake = self.action["acceleration"]

        if self.velocity > self.config["MAX_VELOCITY"]:
            throttle_brake = min(throttle_brake, 1.0*(self.config["MAX_VELOCITY"] - self.velocity))
        elif self.velocity < -self.config["MAX_VELOCITY"]:
            throttle_brake = max(throttle_brake, 1.0*(self.config["MAX_VELOCITY"] - self.velocity))

        if self.PRNDL is "R": # Reverse 
            if self.velocity < -self.config["max_abs_speed_when_stopped"]:
                if throttle_brake > 0:
                    self.braking = throttle_brake
                    self.throttle = 0.0
                    self.reverse = True
                else:
                    self.braking = 0.0
                    self.throttle = abs(throttle_brake)
                    self.reverse = True
            else:
                if throttle_brake > 0.1: # Gear change
                    self.braking =  abs(throttle_brake)
                    self.throttle = 0.0
                    self.reverse = False
                else:
                    self.braking = 0.0
                    self.throttle = abs(throttle_brake)
                    self.reverse = True
        elif self.PRNDL is "D": # Forward
            if self.velocity > self.config["max_abs_speed_when_stopped"]:
                if throttle_brake > 0:
                    self.braking = 0.0
                    self.throttle = throttle_brake
                    self.reverse = False
                else:
                    self.braking = abs(throttle_brake)
                    self.throttle = 0.0
                    self.reverse = False
            else: 
                if throttle_brake > -0.1: 
                    self.braking = 0.0
                    self.throttle = throttle_brake
                    self.reverse = False
                else: # Gear change
                    self.braking = abs(throttle_brake)
                    self.throttle = 0.0
                    self.reverse = True
        elif self.PRNDL is "N":
            self.braking = abs(throttle_brake)
            self.throttle = 0.0

        #self.action["acceleration"] = self.throttle - self.braking


    def step(self, dt):
        """
            Propagate the vehicle state given its actions.

            Integrate a modified bicycle model with a 1st-order response on the steering wheel dynamics.
            If the vehicle is crashed, the actions are overridden with erratic steering and braking until complete stop.
            The vehicle's current lane is updated.

        :param dt: timestep of integration of the model [s]
        """
        if self.LGAgent is not None:
            self.position = np.array([
                                      self.LGAgent.state.transform.position.z - self.config["map_offset"][0],
                                      self.LGAgent.state.transform.position.x - self.config["map_offset"][1], 
                                    ]).astype('float')
            self.velocity = np.sqrt(self.LGAgent.state.velocity.x**2 + self.LGAgent.state.velocity.z**2)
            if self.reverse:
                self.velocity *= -abs(self.velocity)
            self.heading = np.deg2rad(self.LGAgent.state.transform.rotation.y - self.config["map_offset"][2])
        else:
            if self.crashed:
                self.action['steering'] = 0
                self.action['acceleration'] = -1.0*self.velocity
            if self.velocity > self.config["MAX_VELOCITY"]:
                self.action['acceleration'] = min(
                    self.action['acceleration'], 1.0*(self.config["MAX_VELOCITY"] - self.velocity))
            elif self.velocity < -self.config["MAX_VELOCITY"]:
                self.action['acceleration'] = max(
                    self.action['acceleration'], 1.0*(self.config["MAX_VELOCITY"] - self.velocity))

            
            v = self.velocity * np.array([np.cos(self.heading), np.sin(self.heading)])
            self.position += v * dt
            self.heading += self.velocity * np.tan(self.action['steering']) / self.LENGTH * dt
            self.velocity += self.acceleration * dt
            self.jerk = (self.action['acceleration'] - self.acceleration)/dt
            alpha = 0.1
            self.acceleration = self.action['acceleration'] + alpha*(self.acceleration - self.action['acceleration'])

        if self.road:
            self.lane_index = self.road.network.get_closest_lane_index(self.position, self.heading)
            self.lane = self.road.network.get_lane(self.lane_index)
        
        if self.PRNDL is "D" and self.reverse:
            self.PRNDL = "N"
            self.shift_start_time = time()
        elif self.PRNDL is "R" and not self.reverse:
            self.PRNDL = "N"
            self.shift_start_time = time()
        elif self.PRNDL is "N":
            if time() - self.shift_start_time > 1.0:
                self.PRNDL = "R" if self.reverse else "D"
        elif self.PRNDL is "R" and self.reverse:
            self.PRNDL = "R"
            self.shift_start_time = 0.0
        elif self.PRNDL is "D" and not self.reverse:
            self.PRNDL = "D"
            self.shift_start_time = 0.0

        self.intent_prediction()
        self.steps += 1
        #print("For vehicle ", self.Id(), " self.steps ", self.steps)




    def lane_distance_to(self, vehicle):
        """
            Compute the signed distance to another vehicle along current lane.

        :param vehicle: the other vehicle
        :return: the distance to the other vehicle [m]
        """
        if not vehicle:
            return np.nan
        other_center_on_lane = self.lane.local_coordinates(vehicle.position)[0]
        # assuming heading along lane
        other_edge_1_on_lane = self.lane.local_coordinates(vehicle.position)[0]-vehicle.LENGTH/2
        other_edge_2_on_lane = self.lane.local_coordinates(vehicle.position)[0]+vehicle.LENGTH/2
        ego_center_on_lane = self.lane.local_coordinates(self.position)[0]
        other_edge_1_wrt_ego_on_lane = other_edge_1_on_lane - ego_center_on_lane
        other_edge_2_wrt_ego_on_lane = other_edge_2_on_lane - ego_center_on_lane
        if (other_edge_1_wrt_ego_on_lane*other_edge_2_wrt_ego_on_lane < 0):
            return 0
        if(abs(other_edge_1_wrt_ego_on_lane) < abs(other_edge_2_wrt_ego_on_lane)):
            return other_edge_1_wrt_ego_on_lane
        return self.lane.local_coordinates(vehicle.position)[0] - self.lane.local_coordinates(self.position)[0]
    
    def distance_to(self, vehicle):
        return ((self.position[0]-vehicle.position[0])**2+(self.position[1]-vehicle.position[1])**2)**(1/2)

    def _on_collision(self, agent1, agent2, contact):
        name1 = "STATIC OBSTACLE" if agent1 is None else agent1.name
        name2 = "STATIC OBSTACLE" if agent2 is None else agent2.name
        print("{} collided with {} at {}".format(name1, name2, contact))
        self.crashed = True


    def check_collision(self, other, SCALE=1.1):
        """
            Check for collision with another vehicle.

        :param other: the other vehicle
        """

        if self is other:
            return
        
        if self.config["_predict_only"]:
            SCALE = 1.0
            
        if self.virtual and other.virtual:
            return 

        if self.is_projection or other.is_projection:
            return

        if not self.config["COLLISIONS_ENABLED"] or not other.config["COLLISIONS_ENABLED"] or self.crashed or other is self:
            return


        # Fast spherical pre-check
        if np.linalg.norm(other.position - self.position) > self.LENGTH:
            return

        if self.LGAgent is not None and other.LGAgent is not None:
            if self.is_ego:
                self.LGAgent.on_collision(self._on_collision)
            elif other.is_ego:
                other.LGAgent.on_collision(other._on_collision)  

        # Accurate rectangular check
        if utils.rotated_rectangles_intersect((self.position, self.LENGTH, self.WIDTH, self.heading),
                                              (other.position, SCALE*other.LENGTH, SCALE*other.WIDTH, other.heading)):
            #self.velocity = other.velocity = min(self.velocity, other.velocity)
            self.crashed = other.crashed = True
            #if self.is_ego_vehicle:
            #    print("ego crashed")

    @property
    def direction(self):
        return np.array([np.cos(self.heading), np.sin(self.heading)])

    def to_dict(self, relative_features=[], origin_vehicle=None):
        if origin_vehicle is None:
            origin_vehicle = self
        origin_lane = None
        if hasattr(origin_vehicle, 'route_lane_index'):
            origin_lane = origin_vehicle.road.network.get_lane(origin_vehicle.route_lane_index)
        velocity = self.velocity * self.direction
        if origin_lane is not None:
            x = origin_lane.local_coordinates(self.position)[0]
            y = origin_lane.local_coordinates(self.position)[1]+origin_lane.DEFAULT_WIDTH
            vx = np.dot(velocity, origin_lane.direction)
            vy = np.dot(velocity, origin_lane.direction_lateral)
        else:
            x = self.position[0]
            y = self.position[1]
            vx = velocity[0]
            vy = velocity[1]

        cos_h = self.direction[0]
        sin_h = self.direction[1]
        psi = self.heading
        d = {
            'x': utils.remap(x, [- self.config["x_position_range"], self.config["x_position_range"]], [-1, 1]),
            'y': utils.remap(y, [- self.config["y_position_range"], self.config["y_position_range"]], [-1, 1]),
            'vx': utils.remap(vx, [- self.config["velocity_range"], self.config["velocity_range"]], [-1, 1]),
            'vy': utils.remap(vy, [- self.config["velocity_range"], self.config["velocity_range"]], [-1, 1]),
            'cos_h': cos_h,
            'sin_h': sin_h,
            'length': self.LENGTH/100,
            'width': self.WIDTH/10,
            'psi': psi/(2*np.pi),
            'lane_psi': self.lane.heading_at(self.position[0])/(2*np.pi),
        }
        if (origin_vehicle is not self):
            origin_dict = origin_vehicle.to_dict([], origin_vehicle)
            for key in relative_features:
                d[key] -= origin_dict[key]
        else:
            for key in relative_features:
                d[key] = 0             
        return d

    def dump(self):
        """
            Update the internal log of the vehicle, containing:
                - its kinematics;
                - some metrics relative to its neighbour vehicles.
        """
        data = {
            'x': self.position[0],
            'y': self.position[1],
            'psi': self.heading,
            'lane_psi': self.lane.heading_at(self.position[0]),
            'vx': self.velocity * np.cos(self.heading),
            'vy': self.velocity * np.sin(self.heading),
            'v': self.velocity,
            'acceleration': self.action['acceleration'],
            'steering': self.action['steering']}

        if self.road:
            for lane_index in self.road.network.side_lanes(self.lane_index):
                lane_coords = self.road.network.get_lane(
                    lane_index).local_coordinates(self.position)
                data.update({
                    'dy_lane_{}'.format(lane_index): lane_coords[1],
                    'psi_lane_{}'.format(lane_index): self.road.network.get_lane(lane_index).heading_at(lane_coords[0])
                })
            if front_vehicle:
                data.update({
                    'front_v': front_vehicle.velocity,
                    'front_distance': self.lane_distance_to(front_vehicle)
                })
            if rear_vehicle:
                data.update({
                    'rear_v': rear_vehicle.velocity,
                    'rear_distance': rear_vehicle.lane_distance_to(self)
                })

        self.log.append(data)

    def get_log(self):
        """
            Cast the internal log as a DataFrame.

        :return: the DataFrame of the Vehicle's log.
        """
        return pd.DataFrame(self.log)

    def __str__(self):
        return "{} #{}: {}".format(self.__class__.__name__, id(self) % 1000, self.position)

    def __repr__(self):
        return self.__str__()

    @abc.abstractmethod
    def Id(self):
        return str(id(self))[-3:]

    def set_as_projection_only(self):
        self.is_projection = True
    
    def intent_prediction(self):
        if self.is_projection:
            dprojection_step = self.config["SIMULATION_FREQUENCY"]//self.config["TRAJECTORY_FREQUENCY"]
            if (self.steps % dprojection_step)==0:
                self.projection.append(copy.deepcopy(self))


    def predict_trajectory(self, actions, action_duration, trajectory_timestep, dt, out_q=None, pred_horizon=-1, **kwargs):
        """
            Predict the future trajectory of the vehicle given a sequence of actions.

        :param actions: a sequence of future actions.
        :param action_duration: the duration of each action.
        :param trajectory_timestep: the duration between each save of the vehicle state.
        :param dt: the timestep of the simulation
        :return: the sequence of future states
        """
        return self.projection
        states = []
        v = copy.deepcopy(self)
        v.set_as_projection_only()
        #v.virtual = True
        t = 0
        for action in actions:  # only used to iterate (MDP # of actions)
            # v.act(action)  # High-level decision
            acceleration = action[0].item() * v.config['max_acceleration']
            steering = action[1].item() * v.config['max_steer_angle']
            control_action = (
                                        {
                                            "acceleration": acceleration,
                                            "steering": steering                                                    
                                        }
                             )            
            for _ in range(int(action_duration / dt)):
                t += 1
                v.act(control_action)  # Low-level control action
                v.step(dt)
                if (t % int(trajectory_timestep / dt)) == 0:
                    states.append(copy.deepcopy(v))

                if pred_horizon > 0 and t > pred_horizon//dt:
                    break
            else:
                continue
            break
        del(v)
        if out_q is not None:
            out_q.append(states)
        return states

    def __deepcopy__(self, memo=None):
        """
            Perform a deep copy but without copying the environment viewer.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        if memo is not None:
            memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k not in ['LGAgent', 'projection']:
                setattr(result, k, copy.deepcopy(v, memo))
            elif isinstance(v, list):
                setattr(result, k, [])
            else:
                setattr(result, k, None)
        return result

class Obstacle(Vehicle):
    """
        A motionless obstacle at a given position.
    """

    def __init__(self, road, position, heading=0, config=None, color=None, **kwargs):
        super(Obstacle, self).__init__(
                                       road=road, 
                                       position=position, 
                                       heading=heading, 
                                       color=color, 
                                       config=config, 
                                       **kwargs)
        self.target_velocity = 0
        self.velocity = 0
        self.projection = []
        if self.road:
            self.lane_index = self.road.network.get_closest_lane_index(self.position, self.heading)
            self.lane = self.road.network.get_lane(self.lane_index)
        #self.LENGTH = self.WIDTH
        

    def Id(self):
        return super(Obstacle, self).Id()

    def predict_trajectory(self, actions, action_duration, trajectory_timestep, dt, out_q=None, pred_horizon=-1, **kwargs):
        pass

    def step(self, dt):
        if self.LGAgent is not None:
            self.position = np.array([self.LGAgent.state.transform.position.z - self.config["map_offset"][0], 
                                      self.LGAgent.state.transform.position.x - self.config["map_offset"][1]]).astype('float')
            self.velocity = np.sqrt(self.LGAgent.state.velocity.x**2 + self.LGAgent.state.velocity.z**2)
            self.heading = np.deg2rad(self.LGAgent.state.transform.rotation.y  - self.config["map_offset"][2])
        
        
    def intent_prediction(self):
        if self.is_projection:
            if not self.projection:
                self.projection.append(copy.deepcopy(self))
    
    def act(self, action=None, **kwargs):
        pass


class Pedestrian(Vehicle):
    """
      
    """

    def __init__(self, road, position, heading=0, config=None, color=None, **kwargs):
        super(Pedestrian, self).__init__(
                                       road=road, 
                                       position=position, 
                                       heading=heading, 
                                       color=color, 
                                       config=config, 
                                       length=1.0,
                                       width=5.0,
                                       **kwargs)
        #self.target_velocity = 0
        #self.LENGTH = self.WIDTH

    def Id(self):
        return super(Pedestrian, self).Id()

    def predict_trajectory(self, actions, action_duration, trajectory_timestep, dt, out_q=None, pred_horizon=-1, **kwargs):
        return None

    def check_collision(self, other, SCALE = 1.0):
        if other.is_ego():
            if np.linalg.norm(other.position - self.position) < 10: # Don't Move
                self.position -= self.delta_position
            super(Pedestrian, self).check_collision(other, SCALE)

    def step(self, dt):
        self.delta_position = self.velocity * np.array([np.cos(self.heading), np.sin(self.heading)]) * dt
        self.position += self.delta_position

