from baselines import logger
import gym
from gym.envs.registration import registry

def baselines_play(env, policy):

    # env = gym.make(env_id)

    logger.configure()
    # logger.log("Running trained model")
    obs = env.reset()

    state = policy.initial_state if hasattr(policy, 'initial_state') else None
    dones = np.zeros((1,))

    def print_action_and_obs():
        print('episode_rew={}'.format(episode_rew), '  episode_len={}'.format(episode_len),
              'episode_travel = ', episode_travel)
        env.print_obs_space()
        if "extra_obs" in info:
            extra_obs = pp.pformat(info["extra_obs"])
            print(extra_obs)

        #print("Optimal action ",AbstractEnv.ACTIONS[actions[0]], "\n")

    episode_rew = 0
    episode_len = 0
    ego_x0 = None
    while True:
        if state is not None:
            actions, _, state, _ = policy.step(obs, S=state, M=dones)
        else:
            actions, _, _, _ = policy.step(obs)

        if len(actions) == 1:
            actions = actions[0]
        obs, rew, done, info = env.step(actions)
        if ego_x0 is None:
            ego_x0 = env.vehicle.position[0]
        episode_travel = env.vehicle.position[0]-ego_x0
        if (episode_len % 10 == 0) and is_predict_only():
            print_action_and_obs()
       # print(env._max_episode_step)
        episode_rew += rew
        episode_len += 1
        env.render()
        done = done.any() if isinstance(done, np.ndarray) else done

        if done:
            print_action_and_obs()
            episode_rew = 0
            episode_len = 0
            env.close()
            break
