import gym
from statistics import median, mean
from collections import Counter

env = gym.make("CartPole-v0")
env.reset()

def run_random_game():
    scores = []
    for episode in range(1500):
        env.reset()
        score = 0
        for t in range(200):
            #env.render()
            action = env.action_space.sample()
            observation, reward, done, _ = env.step(action)
            score += reward
            if done:
                break
        scores.append(score)
    print('random running game: ')
    print('Average: ' , mean(scores))
    print('Median : ' , median(scores))
    print('counter : ' , Counter(scores))


run_random_game()