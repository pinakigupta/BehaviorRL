######################################################################
#          Deep Reinforcement Learning for Autonomous Driving
#                  Created/Modified on: February 5, 2019
#                      Author:   Pinaki Gupta
#######################################################################
from __future__ import division, print_function
import abc
import numpy as np
import copy
from urban_env import utils
from urban_env.vehicle.dynamics import Vehicle
from importlib import reload
import settings

from ray.rllib.rollout import default_policy_agent_mapping, DefaultMapping
from settings import retrieved_agent_policy
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID

from urban_env.envdict import ACTIONS_DICT


class ControlledVehicle(Vehicle):
    """
        A vehicle piloted by two low-level controller, allowing high-level actions
        such as cruise control and lane changes.

        - The longitudinal controller is a velocity controller;
        - The lateral controller is a heading controller cascaded with a lateral position controller.
    """

    TAU_A = 0.6  # [s]
    TAU_DS = 0.2  # [s]
    PURSUIT_TAU = 1.5*TAU_DS  # [s]
    KP_A = 1 / TAU_A
    KP_HEADING = 1 / TAU_DS
    KP_LATERAL = 1 / 0.5  # [1/s]
    MAX_STEERING_ANGLE = np.pi / 3  # [rad]

    DELTA_VELOCITY = 5  # [m/s]

    def __init__(self,
                 road,
                 position,
                 heading=0,
                 velocity=0,
                 lane_index=None,
                 target_lane_index=None,
                 target_velocity=None,
                 route=None,
                 config=None,
                 **kwargs):
        super(ControlledVehicle, self).__init__(road=road,
                                                position=position,
                                                lane_index=lane_index,
                                                heading=heading,
                                                velocity=velocity,
                                                config=config)
        self.target_lane_index = target_lane_index or self.lane_index
        self.route_lane_index = self.target_lane_index  or self.lane_index# assume this is not changing unless route is changing
        self.target_velocity = self.velocity if target_velocity is None else target_velocity
        self.lane_target_velocity = self.target_velocity
        self.route = route
        self.front_vehicle = None
        self.rear_vehicle = None
        

    @classmethod
    def create_from(cls, vehicle):
        """
            Create a new vehicle from an existing one.
            The vehicle dynamics and target dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        v = cls(vehicle.road, vehicle.position, heading=vehicle.heading, velocity=vehicle.velocity,
                target_lane_index=vehicle.target_lane_index, target_velocity=vehicle.target_velocity,
                route=vehicle.route)
        return v

    def plan_route_to(self, destination):
        """
            Plan a route to a destination in the road network

        :param destination: a node in the road network
        """
        path = self.road.network.shortest_path(self.lane_index[1], destination)
        if path:
            self.route = [self.lane_index] + \
                [(path[i], path[i + 1], None) for i in range(len(path) - 1)]
        else:
            self.route = [self.lane_index]
        return self

    def act(self, action=None, **kwargs):
        """
            Perform a low-level action to change the desired lane or velocity.

            - If a high-level action is provided, update the target velocity and lane;
            - then, perform longitudinal and lateral control.

        :param action: a high-level action
        """
        self.front_vehicle, self.rear_vehicle = self.road.neighbour_vehicles(self)
        is_aggressive_lcx = False
        self.follow_road()

        if action == "LANE_RIGHT":
            _from, _to, _id = self.target_lane_index
            target_lane_index = _from, _to, np.clip(
                _id + 1, 0, len(self.road.network.graph[_from][_to]) - 1)
            if self.road.network.get_lane(target_lane_index).is_reachable_from(self.position):
                self.target_lane_index = target_lane_index
        elif action == "LANE_LEFT":
            _from, _to, _id = self.target_lane_index
            target_lane_index = _from, _to, np.clip(
                _id - 1, 0, len(self.road.network.graph[_from][_to]) - 1)
            if self.road.network.get_lane(target_lane_index).is_reachable_from(self.position):
                self.target_lane_index = target_lane_index
        elif action == "LANE_RIGHT_AGGRESSIVE":
            _from, _to, _id = self.target_lane_index
            target_lane_index = _from, _to, np.clip(
                _id + 1, 0, len(self.road.network.graph[_from][_to]) - 1)
            if self.road.network.get_lane(target_lane_index).is_reachable_from(self.position):
                self.target_lane_index = target_lane_index
                is_aggressive_lcx = True
            # print("LANE_RIGHT_AGGRESSIVE")
        elif action == "LANE_LEFT_AGGRESSIVE":
            _from, _to, _id = self.target_lane_index
            target_lane_index = _from, _to, np.clip(
                _id - 1, 0, len(self.road.network.graph[_from][_to]) - 1)
            if self.road.network.get_lane(target_lane_index).is_reachable_from(self.position):
                self.target_lane_index = target_lane_index
                is_aggressive_lcx = True

        default_offset = 0
        '''if 'target_lane_index' in locals() and 'action' in locals(): # Means lane change
            if (self.target_lane_index == self.lane_index):
                if action == "LANE_LEFT_AGGRESSIVE" or action == "LANE_LEFT_AGGRESSIVE":
                    default_offset = 4
                else :
                    default_offset = -4
                self.action_validity = True'''

        steering = self.steering_control(
            self.target_lane_index, is_aggressive_lcx, default_offset)
        acceleration = self.velocity_control(self.target_velocity)
        self.control_action = {'steering': steering,
                               'acceleration': acceleration}

        super(ControlledVehicle, self).act(self.control_action)

    def follow_road(self):
        """
           At the end of a lane, automatically switch to a next one.
        """
        if self.road.network.get_lane(self.target_lane_index).after_end(self.position):
            self.target_lane_index = self.road.network.next_lane(self.target_lane_index,
                                                                 route=self.route,
                                                                 position=self.position,
                                                                 np_random=self.road.np_random)

    def steering_control(self, target_lane_index, is_agressive=False, default_offset=0):
        """
            Steer the vehicle to follow the center of an given lane.

        1. Lateral position is controlled by a proportional controller yielding a lateral velocity command
        2. Lateral velocity command is converted to a heading reference
        3. Heading is controlled by a proportional controller yielding a heading rate command
        4. Heading rate command is converted to a steering angle

        :param target_lane_index: index of the lane to follow
        :return: a steering wheel angle command [rad]
        """
        target_lane = self.road.network.get_lane(target_lane_index)
        lane_coords = target_lane.local_coordinates(self.position)
        lane_next_coords = lane_coords[0] + self.velocity * self.PURSUIT_TAU
        lane_future_heading = target_lane.heading_at(lane_next_coords)
        lane_current_heading = target_lane.heading_at(self.position)


        # Lateral position control
        target_offset = default_offset + lane_coords[1]
        lateral_velocity_command = (- 2 * self.KP_LATERAL *
                                    target_offset) if is_agressive else (- self.KP_LATERAL * target_offset)

        # Lateral velocity to heading
        heading_command = np.arcsin(
            np.clip(lateral_velocity_command/utils.not_zero(self.velocity), -1, 1))
        heading_ref = lane_future_heading  + \
            np.clip(heading_command, -np.pi/4, np.pi/4)
        # Heading control
        heading_rate_command = self.KP_HEADING * \
            utils.wrap_to_pi(heading_ref - self.heading)
        heading_rate_command = (
            2*heading_rate_command) if is_agressive else heading_rate_command
        # Heading rate to steering angle
        steering_angle = self.LENGTH / \
            utils.not_zero(self.velocity) * np.arctan(heading_rate_command)
        steering_angle = np.clip(
            steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)

        '''if isinstance(self, MDPVehicle) and is_agressive:
            print("Id: ",self.Id(), " lane_future_heading ", lane_future_heading,\
                  "steering_angle ", steering_angle," lateral_velocity_command ", lateral_velocity_command)
            print("lane_coords ",lane_coords,"self.action ", self.action)'''

        return steering_angle

    def velocity_control(self, target_velocity):
        """
            Control the velocity of the vehicle.

            Using a simple proportional controller.

        :param target_velocity: the desired velocity
        :return: an acceleration command [m/s2]
        """

        return self.KP_A * (target_velocity - self.velocity)

    def set_route_at_intersection(self, _to):
        """
            Set the road to be followed at the next intersection.
            Erase current planned route.
        :param _to: index of the road to follow at next intersection, in the road network
        """

        if not self.route:
            return
        for index in range(min(len(self.route), 3)):
            try:
                next_destinations = self.road.network.graph[self.route[index][1]]
            except KeyError:
                continue
            if len(next_destinations) >= 2:
                break
        else:
            return
        next_destinations_from = list(next_destinations.keys())
        if _to == "random":
            _to = self.road.np_random.randint(0, len(next_destinations_from))
        next_index = _to % len(next_destinations_from)
        self.route = self.route[0:index+1] + \
            [(self.route[index][1], next_destinations_from[next_index], self.route[index][2])]

    def check_collision(self, other, SCALE=1.1):
        # if not self.is_ego() or other.is_ego(): #MDP is not learning from
            # collision experiences of other vehicles, yet
        #    return
        return super(ControlledVehicle, self).check_collision(other, SCALE)


    def Id(self):
        return super(ControlledVehicle, self).Id()

    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError()


class MDPVehicle(ControlledVehicle):
    """
        A controlled vehicle with a specified discrete range of allowed target velocities.
    """

    SPEED_COUNT = 5  # []
    SPEED_MIN = 0  # [m/s]
    SPEED_MAX = 30  # [m/s]

    def __init__(self,
                 road,
                 position,
                 heading=0,
                 velocity=0,
                 lane_index=None,
                 target_lane_index=None,
                 target_velocity=None,
                 route=None,
                 config=None,
                 **kwargs):
        super(MDPVehicle, self).__init__(road=road,
                                         position=position,
                                         heading=heading,
                                         velocity=velocity,
                                         lane_index=lane_index,
                                         target_lane_index=target_lane_index,
                                         target_velocity=target_velocity,
                                         route=route,
                                         config=config)
        self.velocity_index = self.speed_to_index(self.target_velocity)
        self.target_velocity = self.index_to_speed(self.velocity_index)
        self.discrete_action = None

    def act(self, action=None, **kwargs):
        """
            Perform a high-level action.

            If the action is a velocity change, choose velocity from the allowed discrete range.
            Else, forward action to the ControlledVehicle handler.

        :param action: a high-level action
        """
        self.discrete_action = action
        if action == "FASTER":
            self.velocity_index = self.speed_to_index(self.velocity) + 1
            self.velocity_index = np.clip(
                self.velocity_index, 0, self.SPEED_COUNT - 1)
            self.target_velocity = self.index_to_speed(self.velocity_index)
            super(MDPVehicle, self).act(action)
        elif action == "SLOWER":
            self.velocity_index = self.speed_to_index(self.velocity) - 1
            self.velocity_index = np.clip(
                self.velocity_index, 0, self.SPEED_COUNT - 1)
            self.target_velocity = self.index_to_speed(self.velocity_index)
            super(MDPVehicle, self).act(action)
        else:
            alpha = 0.8
            self.target_velocity = self.lane_target_velocity + alpha * \
                (self.target_velocity - self.lane_target_velocity)
            super(MDPVehicle, self).act(action)

    @classmethod
    def index_to_speed(cls, index):
        """
            Convert an index among allowed speeds to its corresponding speed
        :param index: the speed index []
        :return: the corresponding speed [m/s]
        """
        if cls.SPEED_COUNT > 1:
            return cls.SPEED_MIN + index * (cls.SPEED_MAX - cls.SPEED_MIN) / (cls.SPEED_COUNT - 1)
        else:
            return cls.SPEED_MIN

    @classmethod
    def speed_to_index(cls, speed):
        """
            Find the index of the closest speed allowed to a given speed.
        :param speed: an input speed [m/s]
        :return: the index of the closest speed allowed []
        """
        x = (speed - cls.SPEED_MIN) / (cls.SPEED_MAX - cls.SPEED_MIN)
        return np.int(np.clip(np.round(x * (cls.SPEED_COUNT - 1)), 0, cls.SPEED_COUNT - 1))

    def speed_index(self):
        """
            The index of current velocity
        """
        return self.speed_to_index(self.velocity)



    def Id(self):
        return super(MDPVehicle, self).Id()

    def __str__(self):
        str = ""
        str = "vehicle = " + self.Id()
        if self.front_vehicle is not None:
            str += " front_vehicle = " + self.front_vehicle.Id()
        if self.rear_vehicle is not None:
            str += " rear_vehicle = " + self.rear_vehicle.Id()
        return str


class IDMDPVehicle(MDPVehicle):
    """
        A MDP vehicle with a learned policy
    """

    def __init__(self,
                 road,
                 position,
                 heading=0,
                 velocity=0,
                 lane_index=None,
                 target_lane_index=None,
                 target_velocity=None,
                 route=None,
                 config=None,
                 **kwargs):
        super(IDMDPVehicle, self).__init__(road=road,
                                           position=position,
                                           heading=heading,
                                           velocity=velocity,
                                           lane_index=lane_index,
                                           target_lane_index=target_lane_index,
                                           target_velocity=target_velocity,
                                           route=route,
                                           config=config)

        self.retrieved_agent_policy = None
        self.observation = None
        self.sim_steps = 0
        


    def act(self, observations=None, **kwargs):
        if self.observation is None:
            self.observation = observations[self]
        obs = self.observation.observe()

            
        if self.retrieved_agent_policy is None:
            import settings
            retrieved_agent_policy = settings.retrieved_agent_policy
            self.retrieved_agent_policy = copy.copy(retrieved_agent_policy)

        if self.sim_steps >= self.config["SIMULATION_FREQUENCY"]//self.config["POLICY_FREQUENCY"]:
            if self.retrieved_agent_policy is not None:
                self.discrete_action = ACTIONS_DICT[self.retrieved_agent_policy.compute_single_action(obs, [])[0]]
                #self.discrete_action = ACTIONS_DICT[0]
                self.sim_steps = 0
                #print("ID", self.Id(), "action ", self.discrete_action," steps ", self.sim_steps)

        
        if self.discrete_action is not None:
            super(IDMDPVehicle, self).act(self.discrete_action)
        self.sim_steps += 1




