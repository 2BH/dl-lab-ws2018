# export DISPLAY=:0 

import sys
sys.path.append("../") 

import numpy as np
import gym
from dqn.dqn_agent import DQNAgent
from dqn.networks import ConvolutionNeuralNetwork, CNNTargetNetwork
from tensorboard_evaluation import *
import itertools as it
from utils import *
import os

def run_episode(env, agent, deterministic, skip_frames=0,  do_training=True, rendering=False, max_timesteps=1000, history_length=0):
    """
    This methods runs one episode for a gym environment. 
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """

    stats = EpisodeStats()

    # Save history
    image_hist = []

    step = 0
    state = env.reset()

    # fix bug of corrupted states without rendering in gym environment
    env.viewer.window.dispatch_events() 

    # append image history to first state
    state = state_preprocessing(state)
    image_hist.extend([state] * (history_length + 1))
    state = np.array(image_hist).reshape(96, 96, history_length + 1)
    
    while True:

        # TODO: get action_id from agent
        # Hint: adapt the probabilities of the 5 actions for random sampling so that the agent explores properly. 
        # action_id = agent.act(...)
        # action = your_id_to_action_method(...)

        action_id = agent.act(state=state, deterministic=deterministic)
        action = id_to_action(action_id)
        # Hint: frame skipping might help you to get better results.
        reward = 0
        for _ in range(skip_frames + 1):
            next_state, r, terminal, info = env.step(action)
            reward += r

            if rendering:
                env.render()

            if terminal: 
                 break

        next_state = state_preprocessing(next_state)
        image_hist.append(next_state)
        image_hist.pop(0)
        next_state = np.array(image_hist).reshape(96, 96, history_length + 1)

        if do_training:
            agent.train(state, action_id, next_state, reward, terminal)

        stats.step(reward, action_id)

        state = next_state
        
        if terminal or (step * (skip_frames + 1)) > max_timesteps : 
            break

        step += 1

    return stats


def train_online(env, agent, num_episodes, skip_frames=0, max_timesteps=1000, history_length=0, model_dir="./models_carracing", tensorboard_dir="./tensorboard"):
   
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
 
    print("... train agent")
    tensorboard = Evaluation(os.path.join(tensorboard_dir, "train"), ["episode_reward", "valid_episode_reward", "straight", "left", "right", "accel", "brake", "left_acce", "right_acce"])

    for i in range(num_episodes):
        print("epsiode %d" % i)

        # Hint: you can keep the episodes short in the beginning by changing max_timesteps (otherwise the car will spend most of the time out of the track)
        max_timesteps_reduced = int(np.max([300, max_timesteps * i / num_episodes]))
        stats = run_episode(env, agent, history_length=history_length, skip_frames=skip_frames, max_timesteps=max_timesteps_reduced, deterministic=False, do_training=True, rendering=False)

        tensorboard.write_episode_data(i, eval_dict={ "episode_reward" : stats.episode_reward, 
                                                      "straight" : stats.get_action_usage(STRAIGHT),
                                                      "left" : stats.get_action_usage(LEFT),
                                                      "right" : stats.get_action_usage(RIGHT),
                                                      "accel" : stats.get_action_usage(ACCELERATE),
                                                      "brake" : stats.get_action_usage(BRAKE),
                                                      "left_acce" : stats.get_action_usage(LEFT_ACCELERATE),
                                                      "right_acce" : stats.get_action_usage(RIGHT_ACCELERATE)
                                                      })

        # TODO: evaluate agent with deterministic actions from time to time

        if i % 100 == 0 or (i >= num_episodes - 1):
            valid_episode_reward = 0
            for j in range(3):
                valid_stats = run_episode(env, agent, history_length=history_length, skip_frames=skip_frames, max_timesteps=max_timesteps, deterministic=True, do_training=False, rendering=False)
                valid_episode_reward += valid_stats.episode_reward
            tensorboard.write_valid_episode_data(i, eval_dict={ "valid_episode_reward" : valid_episode_reward/3})
            agent.saver.save(agent.sess, os.path.join(model_dir, "dqn_agent.ckpt")) 

    tensorboard.close_session()

def state_preprocessing(state):
    return rgb2gray(state).reshape(96, 96) / 255.0

if __name__ == "__main__":

    env = gym.make('CarRacing-v0').unwrapped
    
    # TODO: Define Q network, target network and DQN agent
    # ...


    state_dim = (96, 96)
    history_length = 2
    num_actions = 5
    skip_frames = 3
    method = "CQL"
    # method = "CQL"
    game = "carracing"
    epsilon = 0.05
    epsilon_decay = 1
    epsilon_min = 0.03
    explore_type = "epsilon_greedy"
    tau = 0.5
    Q = ConvolutionNeuralNetwork(state_dim=state_dim, num_actions=num_actions, history_length=history_length, hidden=300, lr=3e-4)
    Q_target = CNNTargetNetwork(state_dim=state_dim, num_actions=num_actions, history_length=history_length, hidden=300, lr=3e-4)
    agent = DQNAgent(Q, Q_target, num_actions, method=method, discount_factor=0.98, batch_size=64, epsilon=epsilon, epsilon_decay=epsilon_decay, explore_type=explore_type, game=game, tau=tau, epsilon_min=epsilon_min)
    train_online(env, agent, skip_frames=skip_frames, num_episodes=1200, max_timesteps=1000, history_length=history_length, model_dir="./models_carracing")
    """
    os.system('python test_carracing.py')

    
    state_dim = (96, 96)
    history_length = 2
    num_actions = 5
    skip_frames = 3
    method = "DQL"
    # method = "CQL"
    game = "carracing"
    epsilon = 0.3
    epsilon_decay = 0.99
    epsilon_min = 0.03
    explore_type = "epsilon_greedy"
    tau = 2
    Q = ConvolutionNeuralNetwork(state_dim=state_dim, num_actions=num_actions, history_length=history_length, hidden=300, lr=3e-4)
    Q_target = CNNTargetNetwork(state_dim=state_dim, num_actions=num_actions, history_length=history_length, hidden=300, lr=3e-4)
    agent = DQNAgent(Q, Q_target, num_actions, method=method, discount_factor=0.98, batch_size=64, epsilon=epsilon, epsilon_decay=epsilon_decay, explore_type=explore_type, game=game, tau=tau, epsilon_min=epsilon_min)
    train_online(env, agent, skip_frames=skip_frames, num_episodes=1500, max_timesteps=1000, history_length=history_length, model_dir="./models_carracing")
    
    os.system('python test_carracing.py')"""
