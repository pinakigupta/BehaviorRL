######################################################################
#          Deep Reinforcement Learning for Autonomous Driving
#                  Created/Modified on: February 5, 2019
#                      Author: Munir Jojo-Verge, Pinaki Gupta
#######################################################################

import numpy as np

OBS_SCALE = 1

REWARD_WEIGHTS = [7/100, 7/100, 1/100, 1/100, 9/10, 9/10]
#REWARD_WEIGHTS = [10/100, 10/100, 0/100, 0/100, 9/10, 9/10]
#REWARD_WEIGHTS = [5/100, 5/100, 1/100, 1/100, 5/10, 5/10]
#REWARD_WEIGHTS = [1/100, 1/100, 1/100, 1/100, 1/10, 1/10]


def distance_2_goal_reward(achieved_goal, desired_goal, p=0.5):
    return - np.power(np.dot(OBS_SCALE * np.abs(achieved_goal - desired_goal), REWARD_WEIGHTS), p)

if __name__ == "__main__":
    # lets suppose an error of 10 cm in x & y, 0.1m/s error and about 2 degrees = 0.0872665    
    err_angle = np.deg2rad(5)

    achieved_goal = np.array([0.25, 0.25, 0.2, 0.2, np.cos(err_angle), np.sin(err_angle)])
    desired_goal  = np.array([0.0, 0.0, 0.0, 0.0, np.cos(0), np.sin(0)])
    print (distance_2_goal_reward(achieved_goal, desired_goal))
