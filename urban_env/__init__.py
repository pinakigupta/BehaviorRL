from gym.envs.registration import register

register(
    id='multilane-v0',
    entry_point='urban_env.envs:MultilaneEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 250}
)

register(
    id='merge-v0',
    entry_point='urban_env.envs:MergeEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 250}
)

register(
    id='roundabout-v0',
    entry_point='urban_env.envs:RoundaboutEnv',    
)

register(
    id='two-way-v0',
    entry_point='urban_env.envs:TwoWayEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 500}
)

register(
    id='parking-v0',
    entry_point='urban_env.envs:ParkingEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 20}
)

register(
    id='parking_2outs-v0',
    entry_point='urban_env.envs:ParkingEnv_2outs',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 25}
)

register(
    id='LG-SIM-ENV-v0',
    entry_point='urban_env.envs:LG_Sim_Env',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 25}
)

