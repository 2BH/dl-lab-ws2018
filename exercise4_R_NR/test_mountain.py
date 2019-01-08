import os
from datetime import datetime
import gym
import json
from dqn.dqn_agent import DQNAgent
from train_cartpole import run_episode
from dqn.networks import *
import numpy as np

np.random.seed(0)

if __name__ == "__main__":

    # TODO: load DQN agent
    # ...
    env = gym.make("MountainCar-v0").unwrapped

    # TODO: 
    # 1. init Q network and target network (see dqn/networks.py)
    # 2. init DQNAgent (see dqn/dqn_agent.py)
    # 3. train DQN agent with train_online(...)
    num_actions = 3
    state_dim = 2
    method = "DQL"
    game = "mountaincar"
    model_dir = "./models_mountaincar"
    epsilon = 0.3
    epsilon_decay = 0.99
    explore_type = "epsilon_greedy"
    epsilon_min = 0.04
    tau = 2
    Q = NeuralNetwork(state_dim=state_dim, num_actions=num_actions, hidden=200, lr=3e-4)
    Q_target = TargetNetwork(state_dim=state_dim, num_actions=num_actions, hidden=200, lr=3e-4)
    agent = DQNAgent(Q, Q_target, num_actions,
                     method=method,
                     discount_factor=0.98,
                     batch_size=64,
                     epsilon=epsilon,
                     epsilon_decay=epsilon_decay,
                     explore_type=explore_type,
                     epsilon_min=epsilon_min,
                     game=game, tau=tau)
    agent.load("./models_mountaincar/dqn_agent.ckpt")
 
    n_test_episodes = 15

    episode_rewards = []
    for i in range(n_test_episodes):
        stats = run_episode(env, agent, deterministic=True, do_training=False, rendering=True)
        episode_rewards.append(stats.episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
 
    if not os.path.exists("./results"):
        os.mkdir("./results")  

    fname = "./results/mountaincar_results_dqn-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)
            
    env.close()
    print('... finished')

