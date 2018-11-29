from __future__ import print_function

from datetime import datetime
import numpy as np
import gym
import os
import json

from model import Model
from utils import *

history_length = 5
def run_episode(env, agent, rendering=True, max_timesteps=1000 ):

    episode_reward = 0
    step = 0
    state = env.reset()
    state_history = np.zeros((1, state.shape[0], state.shape[1], history_length))
    while True:
        
        # TODO: preprocess the state in the same way than in in your preprocessing in train_agent.py
        state = rgb2gray(state)
        state_history[0,:,:,0:history_length-1] = state_history[0,:,:,1:]
        state_history[0,:,:,-1] = state
        # state = state.reshape(1, 96, 96, 1)
        # TODO: get the action from your agent! If you use discretized actions you need to transform them to continuous
        # actions again. a needs to have a shape like np.array([0.0, 0.0, 0.0])
        a = agent.sess.run(agent.output, feed_dict={agent.x_input:state_history})[0]
        next_state, r, done, info = env.step(a)
        episode_reward += r       
        state = next_state
        step += 1
        
        if rendering:
            env.render()

        if done or step > max_timesteps: 
            break

    return episode_reward


if __name__ == "__main__":

    rendering = True                      # set rendering=False if you want to evaluate faster
    
    n_test_episodes = 15                  # number of episodes to test

    # TODO: load agent
    agent = Model(history_length=history_length)
    agent.load("models/agent.ckpt")

    env = gym.make('CarRacing-v0').unwrapped

    episode_rewards = []
    for i in range(n_test_episodes):
        episode_reward = run_episode(env, agent, rendering=rendering)
        episode_rewards.append(episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
 
    fname = "results/results_bc_agent-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)
            
    env.close()
    print('... finished')
