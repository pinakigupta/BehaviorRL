######################################################################
#          Deep Reinforcement Learning for Autonomous Driving
#                  Created/Modified on: February 5, 2019
#                      Author:   Pinaki Gupta
#######################################################################

from __future__ import division, print_function, absolute_import
import numpy as np
import gym
from gym import GoalEnv
from urban_env import utils
from urban_env.envs.abstract import AbstractEnv
from urban_env.road.lane import AbstractLane
from urban_env.road.lane import LineType, StraightLane, SineLane
from urban_env.road.road import Road, RoadNetwork
from urban_env.envs.graphics import EnvViewer
from urban_env.vehicle.control import ControlledVehicle, MDPVehicle, IDMDPVehicle
from urban_env.vehicle.behavior import IDMVehicle
from urban_env.vehicle.dynamics import Obstacle
from handle_model_files import is_predict_only
from urban_env.envs.graphics import EnvViewer
import random
import pprint
import sys

class TwoWayEnv(AbstractEnv):
    """
        A risk management task: the agent is driving on a two-way lane with icoming traffic.
        It must balance making progress by overtaking and ensuring safety.

        These conflicting objectives are implemented by a reward signal and a constraint signal,
        in the CMDP/BMDP framework.
    """
    ROAD_LENGTH = 2000
    ROAD_SPEED = 25
    
    
    DEFAULT_CONFIG = {**AbstractEnv.DEFAULT_CONFIG, 
        **{
            "observation": {
                "type": "Kinematics",
                # "features": {"vehicles": ['x', 'y', 'vx', 'vy', 'psi']},
                "features":  ['x', 'y', 'vx', 'vy', 'psi'],
                "relative_features": ['x'],
                "obs_count": 6,
                "obs_size": 6,
                "goals_size": None,
                "goals_count": None              
            },

            "obstacle_type": "urban_env.vehicle.dynamics.Obstacle",
            "other_vehicles_type": "urban_env.vehicle.behavior.IDMVehicle", 
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
            "x_position_range": AbstractEnv.DEFAULT_CONFIG["PERCEPTION_DISTANCE"],
            "y_position_range": AbstractLane.DEFAULT_WIDTH * 2,
            "velocity_range": MDPVehicle.SPEED_MAX,   
            "MAX_VELOCITY": MDPVehicle.SPEED_MAX,  
            "closest_lane_dist_thresh": 500,       
            },
        **{  # Frequency related
            "SIMULATION_FREQUENCY": 10,  # The frequency at which the system dynamics are simulated [Hz],
            "PREDICTION_SIMULATION_FREQUENCY": 10,  # The frequency at which the system dynamics are predicted [Hz],
            "POLICY_FREQUENCY": 2,  # The frequency at which the agent can take actions [Hz]
            "TRAJECTORY_FREQUENCY": 0.5, # The frequency at which the agent trajectory is generated, mainly for visualization
            "TRAJECTORY_HORIZON": 10,
          },
        **{  # RL Policy model related
            "LOAD_MODEL_FOLDER": "20190516-120321",
            "RESTORE_COND": "RESTORE", 
            "retrieved_agent_policy": 0,
          },
    }

    def __init__(self, config=DEFAULT_CONFIG):
        super(TwoWayEnv, self).__init__(config)
        #self.goal_achieved = False
        EnvViewer.SCREEN_HEIGHT = self.config['screen_height']
        EnvViewer.SCREEN_WIDTH = self.config['screen_width']         
        self.ego_x0 = None
        self.reset()
        
    def step(self, action):
        self.steps += 1
        self.previous_action = action
        obs, rew, done, info = super(TwoWayEnv, self).step(action)
        self.episode_travel = self.vehicle.position[0] - self.ego_x0 

        #self.print_obs_space(ref_vehicle=self.idmdp_opp_vehicle)
        #self.print_obs_space(ref_vehicle=self.vehicle)
        #self.print_obs_space(ref_vehicle=self.idmdp_vehicle)
        #self._set_curriculam(curriculam_reward_threshold=0.6*self.GOAL_REWARD)
        return (obs, rew, done, info)

    def _on_route(self, veh=None):
        if veh is None:
            veh = self.vehicle
        lane_ID = veh.lane_index[2]
        onroute = (lane_ID == 1)
        return onroute
    
    def _on_road(self, veh=None):
        if veh is None:
            veh = self.vehicle
        return (veh.position[0] < self.ROAD_LENGTH) and (veh.position[0] > 0)

    def _goal_achieved(self, veh=None):
        if veh is None:
            veh = self.vehicle
        return (veh.position[0] > self.config["GOAL_LENGTH"] and \
                self._on_route(veh))

    def _reward(self, action):
        """
            The vehicle is rewarded for driving with high velocity
        :param action: the action performed
        :return: the reward of the state-action transition
        """

        #neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        collision_reward = self.config["COLLISION_REWARD"] * self.vehicle.crashed
        velocity_reward = self.config["VELOCITY_REWARD"] * (self.vehicle.velocity_index -1) / (self.vehicle.SPEED_COUNT - 1)
        if (velocity_reward > 0):
            velocity_reward *= self._on_route()
        goal_reward = self.config["GOAL_REWARD"]
        if self.vehicle.crashed:
            reward = collision_reward + min(0.0, velocity_reward)
        elif self._goal_achieved():
            reward = goal_reward + velocity_reward
        else:
            reward = velocity_reward
        if not self.vehicle.action_validity:
            reward = reward + self.config["INVALID_ACTION_REWARD"]
        return reward

    def _is_terminal(self):
        """
            The episode is over if the ego vehicle crashed or the time is out.
        """
        terminal = self.vehicle.crashed or \
                   self._goal_achieved() or \
                  (not self._on_road()) or \
                  (self.steps >= self.config["duration"]) 
                 # or  (self.vehicle.action_validity == False)
        #if terminal:
        #    print("self.steps ",self.steps," terminal ", terminal, " crashed ", self.vehicle.crashed , " goal_achieved ",self._goal_achieved() )
        #print("self.episode_reward ", self.episode_reward)
        return terminal

    def _constraint(self, action):
        """
            The constraint signal is the time spent driving on the opposite lane, and occurence of collisions.
        """
        return float(self.vehicle.crashed) + float(self.vehicle.lane_index[2] == 0)/15

    def reset(self):
        self.steps = 0
        self._make_road()
        self._make_vehicles()
        return super(TwoWayEnv, self).reset()

    def _make_road(self, length = 800):
        """
            Make a road composed of a two-way road.
        :return: the road
        """
        net = RoadNetwork()
        length = self.ROAD_LENGTH
        # Lanes
        net.add_lane("a", "b", StraightLane([0, 0], [length, 0],
                                            line_types=[LineType.CONTINUOUS_LINE, LineType.STRIPED]))
        net.add_lane("a", "b", StraightLane([0, StraightLane.DEFAULT_WIDTH], [length, StraightLane.DEFAULT_WIDTH],
                                            line_types=[LineType.NONE, LineType.CONTINUOUS_LINE]))
        net.add_lane("b", "a", StraightLane([length, StraightLane.DEFAULT_WIDTH], [0, StraightLane.DEFAULT_WIDTH],
                                            line_types=[LineType.NONE, LineType.NONE]))                                             
        net.add_lane("b", "a", StraightLane([length, 0], [0, 0],
                                            line_types=[LineType.NONE, LineType.NONE]))
                                           

        road = Road(network=net, np_random=self.np_random, config=self.config)
        self.road = road


    def _make_vehicles(self):
        """
            Populate a road with several vehicles on the road
        :return: the ego-vehicle
        """
        scene_complexity = 3
        if 'DIFFICULTY_LEVELS' in self.config:
            scene_complexity = self.config['DIFFICULTY_LEVELS']

        if '_predict_only' in self.config:
            if self.config['_predict_only']:
                scene_complexity = 4
        
        road = self.road
        lane_idx = ("a", "b", 1)
        ego_lane = road.network.get_lane(lane_idx)
        low = 400 if self.config["_predict_only"] else max(0, (700 - 30*scene_complexity))
        ego_init_position = ego_lane.position(np.random.randint(low=low, 
                                                                high=low+60
                                                                ),
                                               0
                                             )
        x0 = ego_init_position[0]
        self.ego_x0 = x0
        #self.ego_x0 -= self.ROAD_LENGTH - 150
        ego_init_position = ego_lane.position(x0, 0)
        ego_vehicle = MDPVehicle(
                                 self.road,
                                 position=ego_init_position,
                                 velocity= np.random.randint(low=15, high=25),
                                 target_velocity=self.ROAD_SPEED,
                                 heading=ego_lane.heading_at(x0),
                                 config=self.config
                                 )
        ego_vehicle.is_ego_vehicle = True
        self.road.vehicles.append(ego_vehicle)
        self.road.ego_vehicle = ego_vehicle
        self.vehicle = ego_vehicle

        '''idmdp_init_position = ego_init_position
        idmdp_init_position[0] += 40
        lane_index = ("a", "b", 1)
        idmdp_vehicle = IDMDPVehicle(road=self.road,
                                     position=idmdp_init_position,
                                     velocity=np.random.randint(low=15, high=25),
                                     heading=road.network.get_lane(lane_index).heading_at(0),
                                     target_velocity=self.ROAD_SPEED,
                                     #target_lane_index=lane_index,
                                     #lane_index=lane_index,
                                     config=self.config
                                    )

        self.road.add_vehicle(idmdp_vehicle)
        self.idmdp_vehicle = idmdp_vehicle

        lane_index = ("b", "a", 1)
        x0 = self.ROAD_LENGTH-self.ego_x0 - 150
        idmdp_opp_init_position = road.network.get_lane(lane_index).position(x0, 0)
        idmdp_opp_vehicle = IDMDPVehicle(
                                         road=self.road,
                                         position=idmdp_opp_init_position,
                                         velocity=np.random.randint(low=15, high=25),
                                         heading=road.network.get_lane(lane_index).heading_at(x0),
                                         target_velocity=self.ROAD_SPEED,
                                         #target_lane_index=lane_index,
                                         #lane_index=lane_index,
                                         config=self.config
                                        )

        self.road.add_vehicle(idmdp_opp_vehicle)
        self.idmdp_opp_vehicle = idmdp_opp_vehicle'''

        vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

        def lcx(scene_complexity):
            percent = scene_complexity*10-20
            return random.randrange(100) < percent

        lane_index=("a", "b", 1)
        # stationary vehicles
        stat_veh_x0 = []
        rand_stat_veh_count = np.random.randint(low=0, high=2*scene_complexity)
        for i in range(rand_stat_veh_count):
            x0 = self.ego_x0 + 90 + 90*i + 10*self.np_random.randn()
            stat_veh_x0.append(x0)
            self.road.add_vehicle(
                                    Obstacle(self.road,
                                             position=road.network.get_lane(lane_index)
                                             .position(x0, 1),
                                             heading=road.network.get_lane(lane_index).heading_at(x0),
                                             velocity=0,
                                             target_velocity=0,
                                             target_lane_index=lane_index,
                                             lane_index=lane_index,                             
                                             enable_lane_change=False,
                                             config=self.config
                                             )
                                 )
            
        rand_veh_count = np.random.randint(low=0, high=2*scene_complexity)
        for i in range(rand_veh_count):
            x0 = self.ego_x0+ 90+40*i + 10*self.np_random.randn()
            v =     IDMVehicle(road,
                               position=road.network.get_lane(lane_index)
                               .position(x0, 0),
                               heading=road.network.get_lane(lane_index).heading_at(x0),
                               velocity=max(0,10 + 2*self.np_random.randn()),
                               target_velocity=self.ROAD_SPEED,
                               #target_lane_index=lane_index, 
                               #lane_index=lane_index,                             
                               enable_lane_change=False,
                               config=self.config
                               )
            front_vehicle, _ = self.road.neighbour_vehicles(v)
            d = v.lane_distance_to(front_vehicle) 
            if (d<5):
                continue
            elif(d<20):
                v.velocity = max(0, 2.5 + 0.5*self.np_random.randn())
            self.road.add_vehicle(v)
        
        
        lane_index=("b", "a", 1)
        # stationary vehicles Left Lane
        #if (rand_stat_veh_count == 0):
        rand_oncoming_stat_veh_count = np.random.randint(low=0, high=2*scene_complexity)
        #else:
        #    rand_stat_veh_count = 0
        for i in range(rand_oncoming_stat_veh_count):
            x0 = self.ROAD_LENGTH-self.ego_x0-100-120*i + 10*self.np_random.randn()
            x0_wrt_ego_lane = self.ROAD_LENGTH - x0
            min_offset = 1e6

            if stat_veh_x0:
                dist_from_ego_lane_parked_vehs = [x - x0_wrt_ego_lane for x in stat_veh_x0]
                min_offset = min([abs(y) for y in dist_from_ego_lane_parked_vehs])
            if (min_offset < 10):
                break
            else:
                v = Obstacle(
                             road=road,
                             position=road.network.get_lane(lane_index).position(x0, 1),
                             heading=road.network.get_lane(lane_index).heading_at(x0),
                             velocity=0,
                             target_velocity=0,
                             target_lane_index=lane_index,
                             lane_index=lane_index,
                             enable_lane_change=False,
                             config=self.config
                             )
                v.target_lane_index = lane_index
                v.lane_index = lane_index
                self.road.add_vehicle(v)
       
        lane_change = (scene_complexity)
        for i in range(np.random.randint(low=0,high=2*scene_complexity)):
            x0 = self.ROAD_LENGTH-self.ego_x0-20-120*i + 10*self.np_random.randn()
            v =    IDMVehicle(
                              road,
                              position=road.network.get_lane(lane_index)
                              .position(x0, 0.1),
                              heading=road.network.get_lane(lane_index).heading_at(x0),
                              velocity=0.5*max(0, 20 + 5*self.np_random.randn()),
                              target_velocity=self.ROAD_SPEED,
                              target_lane_index=lane_index,
                              lane_index=lane_index,
                              #enable_lane_change=False,
                              config=self.config
                              )
            v.target_lane_index = lane_index
            v.lane_index = lane_index
            front_vehicle, _ = self.road.neighbour_vehicles(v)
            d = v.lane_distance_to(front_vehicle)
            if(d < 5):
                continue 
            elif(d < 20):
                v.velocity = max(0, 4 + self.np_random.randn())
            self.road.add_vehicle(v)
        
            # ------------------------------------------------------------------------------------------------------------------------------

            # Add the virtual obstacles/constraints
        lane_index = ("b", "a", 1)
        lane = self.road.network.get_lane(lane_index)
        x0 = lane.length/2
        position = lane.position(x0, StraightLane.DEFAULT_WIDTH)
        lane_index = self.road.network.get_closest_lane_index(
                                                            position=position,
                                                            heading=0  
                                                             )  
        virtual_obstacle_left = Obstacle(road=self.road,
                                         position=position,
                                         heading=lane.heading_at(x0),
                                         velocity=0,
                                         target_velocity=0,
                                         lane_index=lane_index,
                                         target_lane_index=lane_index,                     
                                         enable_lane_change=False,
                                         config=self.config
                                         )
        virtual_obstacle_left.virtual = True
        virtual_obstacle_left.LENGTH = lane.length
        self.road.add_vehicle(virtual_obstacle_left)
        self.road.add_virtual_vehicle(virtual_obstacle_left)

        lane_index = ("a", "b", 1)
        lane = self.road.network.get_lane(lane_index)
        x0 = lane.length/2
        position = lane.position(x0, StraightLane.DEFAULT_WIDTH)
        virtual_obstacle_right = Obstacle(road=self.road,
                                          position=position,
                                          heading=lane.heading_at(x0),
                                          velocity=0,
                                          target_velocity=0,
                                          lane_index=lane_index,
                                          target_lane_index=lane_index,                
                                          enable_lane_change=False,
                                          config=self.config
                                          )
        virtual_obstacle_right.virtual = True                                       
        virtual_obstacle_right.LENGTH = lane.length
        self.road.add_vehicle(virtual_obstacle_right)
        self.road.add_virtual_vehicle(virtual_obstacle_right)

        '''lane_index = ("b", "a", 0)
        lane = self.road.network.get_lane(lane_index)
        x0 = 0
        position = lane.position(x0, 0)
        end_obstacle_right = vehicles_type(self.road,
                                           position=position,
                                           heading=lane.heading_at(x0),
                                           velocity=0,
                                           target_velocity=0,
                                           lane_index=lane_index,
                                           target_lane_index=lane_index,                
                                           enable_lane_change=False)
        end_obstacle_right.LENGTH = 4
        self.road.add_vehicle(end_obstacle_right)'''                                    

    def print_obs_space(self, ref_vehicle):
        if not ref_vehicle:
            return
        print("-------------- start obs ", ref_vehicle.Id(), "  ----------------------")
        print("obs space, step ", self.steps)
        if ref_vehicle.discrete_action is not None:
            print("reference discrete action ", ref_vehicle.discrete_action)
        if ref_vehicle.control_action is not None:
            print("reference accel = ", 
                    ref_vehicle.control_action['acceleration'],
                    " steering = ", ref_vehicle.control_action['steering'])
        
        #sys.stdout.flush()
        pp = pprint.PrettyPrinter(indent=4)
        numoffeatures = len(self.config["observation"]["features"])
        numfofobs = len(self.obs)
        numofvehicles = numfofobs//numoffeatures
        close_vehicle_ids = [int(ref_vehicle.Id())]
        modified_obs = self.observations[ref_vehicle].observe()
        close_vehicles = self.road.closest_vehicles_to(ref_vehicle,
                                                           numofvehicles - 1,
                                                           7.0 * MDPVehicle.SPEED_MAX
                                                      )
        for v in close_vehicles:
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
        print("\n\n\n")



