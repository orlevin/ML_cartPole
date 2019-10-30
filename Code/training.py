import gym
import random
import numpy as np
from statistics import median, mean
from collections import Counter
import dnnmodel as dnn
import constants
import log_class

env = gym.make(constants.game_name)
env.reset()

def convert_one_hot_decoding(data):

    if data[1] == 1:
        output = [0, 1]
    elif data[1] == 0:
        output = [1, 0]

    return output

def initial_data_for_traninig():
    # array of Q-VALUES [observation,action]
    training_data = []

    # all scores, even those wich didnt pass the treshold
    all_scores = []

    # scores that met our threshold
    accepted_scores = []

    for _ in range(constants.number_of_games):
        score = 0

        # array that save the moves(Q-values) for the specific game:
        game_memory = []

        prev_observation = []

        for _ in range(constants.goal_steps):

            # choose random action left or right(0 or 1)
            action = env.action_space.sample()

            # take step by the action and return parameters
            observation, reward, done, _ = env.step(action)

            if done and score < 199: break

            # the observation is returned after we took an action
            # so to adjust the observation to his action,
            # we save the action with the prev_observation.
            if (len(prev_observation) > 0):
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score += reward
            if done: break

        # If the score that we get is higher than our score_threshold, we'd like to save
        # every move we made
        if score >= constants.score_threshold:
            accepted_scores.append(score)
            for data in game_memory:
                # convert to one-hot(this is the output layer for our neural network) - change to func
                output = convert_one_hot_decoding(data)
                training_data.append([data[0], output])

        # reset env to play again
        env.reset()
        all_scores.append(score)

    save_training_data = np.array(training_data)
    np.save('saved.npy', save_training_data)

    log_class.logger.info('data on accepted score: ')
    log_class.logger.info('Average: ' + repr(mean(accepted_scores)))
    log_class.logger.info('Median : ' + repr(median(accepted_scores)))
    log_class.logger.info('counter : ' + repr(Counter(accepted_scores)))

    return training_data


def train_model(training_data):

    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
    y = [i[1] for i in training_data]
    model = dnn.bulidModel(input_size=len(X[0]))
    log_class.logger.info('the input size is:' + repr(len(X[0])))
    model.fit({'input': X}, {'targets': y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id='openai_learning')
    return model


