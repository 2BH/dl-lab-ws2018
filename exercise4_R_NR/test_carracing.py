from __future__ import print_function

import os
from datetime import datetime
import json
import gym
from dqn.dqn_agent import DQNAgent
from train_carracing import run_episode
from dqn.networks import *
import numpy as np

np.random.seed(0)

if __name__ == "__main__":

    env = gym.make("CarRacing-v0").unwrapped

    #TODO: Define networks and load agent
    # ....
    state_dim = (96, 96)
    history_length = 2
    num_actions = 5
    skip_frames = 2
    method = "DQL"
    # method = "CQL"
    game = "carracing"
    epsilon = 0.2
    epsilon_decay = 0.999
    epsilon_min = 0.05
    explore_type = "epsilon_greedy"
    tau = 0.5
    
    Q = ConvolutionNeuralNetwork(state_dim=state_dim, num_actions=num_actions, history_length=history_length, hidden=300, lr=3e-4)
    Q_target = CNNTargetNetwork(state_dim=state_dim, num_actions=num_actions, history_length=history_length, hidden=300, lr=3e-4)
    agent = DQNAgent(Q, Q_target, num_actions, method=method, discount_factor=0.98, batch_size=64, epsilon=epsilon, epsilon_decay=epsilon_decay, explore_type=explore_type, game=game, tau=tau, epsilon_min=epsilon_min)
    # agent = DQNAgent(Q, Q_target, num_actions, method=method, discount_factor=0.6, batch_size=64, epsilon=epsilon, epsilon_decay=epsilon_decay, explore_type=explore_type, game=game, tau=tau)
    agent.load("./models_carracing/dqn_agent.ckpt")
    n_test_episodes = 15

    episode_rewards = []
    for i in range(n_test_episodes):
        stats = run_episode(env, agent, history_length=history_length, deterministic=True, do_training=False, rendering=True)
        episode_rewards.append(stats.episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
 
    if not os.path.exists("./results"):
        os.mkdir("./results")  

    fname = "./results/carracing_results_dqn-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)
            
    env.close()
    print('... finished')

