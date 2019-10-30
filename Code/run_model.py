import gym
import numpy as np
import constants
import log_class


env = gym.make(constants.game_name)


def run_games(model):
    env.reset()
    scores = []
    for each_game in range(constants.games_to_play):
        score = 0
        prev_obs = []
        env.reset()
        for _ in range(constants.goal_steps):
            #env.render()

            # at start we take random action because we don't have any prv observation
            if len(prev_obs) == 0:
                action = env.action_space.sample()
            else:
                action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0])

            new_observation, reward, done, info = env.step(action)
            prev_obs = new_observation
            score += reward
            if done: break

        scores.append(score)

    log_class.logger.info("scores: " + repr(scores))
    log_class.logger.info('Average Score:' + repr(sum(scores) / len(scores)))