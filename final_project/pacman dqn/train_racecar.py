import numpy as np
import gym
from dqn.dqn_agent import DQNAgent
from tensorboard_evaluation import Evaluation
from dqn.conv_networks import CNN, CNNTargetNetwork
from utils import *
import os
import json

MODEL_TEST_INTERVAL = 10  # after this number of episodes, test agent with deterministic actions
MODEL_SAVE_INTERVAL = 25  # yep


def state_preprocessing(state):
    return np.squeeze(rgb2gray(state))[::2, ::2] / 255.0
    #return state / 255.0


def run_episode(env,
                agent,
                deterministic,
                softmax=False,
                skip_frames=0,
                do_training=True,
                rendering=False,
                max_timesteps=99999,
                history_length=0,
                diff_history=False,
                apply_lane_penalty=False):
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

    # append image history to first state
    state = state_preprocessing(state)
    image_hist.extend([state] * (history_length + 1))
    # state = np.array(image_hist).reshape(96, 96, history_length + 1)
    state = np.array(image_hist[::-1]).transpose(1, 2, 0)

    if diff_history:
        # current image is at zero, so go through 1...n and subtract current from them
        for i in range(1, state.shape[-1]):
            state[..., i] -= state[..., 0]

    while True:

        action = agent.act(state=state, deterministic=deterministic)

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
        # next_state = np.array(image_hist).reshape(96, 96, history_length + 1)
        next_state = np.array(image_hist[::-1]).transpose(1, 2, 0)

        # so state and next_state are now both np.arrays, of the right length
        
        if diff_history:
            # current image is at zero, so go through 1...n and subtract current from them
            for i in range(1, next_state.shape[-1]):
                next_state[..., i] -= next_state[..., 0]

        if do_training:
            agent.train(state, action, next_state, reward, terminal)

        stats.step(reward, action)

        state = next_state

        if terminal or (step * (skip_frames + 1)) > max_timesteps:
            print(step)
            break

        step += 1

    return stats


def train_online(name,
                 env,
                 num_episodes=10000,
                 lr=1e-4,
                 discount_factor=0.99,
                 batch_size=64,
                 epsilon=0.05,
                 epsilon_decay=0.0,
                 boltzmann=False,
                 tau=0.01,
                 double_q=False,
                 buffer_capacity=5e5,
                 history_length=0,
                 diff_history=False,
                 skip_frames=0,
                 big=False,
                 try_resume=False):

    print("AGENT: " + name)
    print("\t... creating agent")

    # prepare folders
    model_path = os.path.join(base_path, name)
    ckpt_path = os.path.join(model_path, "ckpt")
    tensorboard_path = os.path.join(base_path, "tensorboard")

    for d in [model_path, ckpt_path, tensorboard_path]:
        if not os.path.exists(d):
            os.mkdir(d)

    agent = make_pacman_agent(
        name,
        model_path,
        lr,
        discount_factor,
        batch_size,
        epsilon,
        epsilon_decay,
        boltzmann,
        tau,
        double_q,
        buffer_capacity,
        history_length,
        diff_history,
        big,
        save_hypers=True)

    print("... training agent")

    # todo? make this better
    tensorboard = Evaluation(
        os.path.join(tensorboard_path, agent.name + "_train"),
        ["episode_reward", "NOOP", "up", "right", "left", "down"])
    tensorboard_test = Evaluation(
        os.path.join(tensorboard_path, agent.name + "_test"),
        ["episode_reward", "NOOP", "up", "right", "left", "down"])

    start_episode = 0

    if try_resume:
        possible_file = os.path.join(model_path, "epstrained.json")
        if os.path.exists(possible_file):
            # get the last ep trained; start at the next one
            with open(possible_file, "r") as fh:
                start_episode = json.load(fh) + 1

            #load up model from previous training session
            agent.load(os.path.join(model_path, 'ckpt', 'dqn_agent.ckpt'))

    # training
    for i in range(start_episode, num_episodes):
        print("episode: ", i)

        max_timesteps = 99999

        stats = run_episode(
            env,
            agent,
            max_timesteps=max_timesteps,
            deterministic=False,
            softmax=False,
            do_training=True,
            rendering=False,
            skip_frames=skip_frames,
            history_length=history_length,
            diff_history=diff_history)

        tensorboard.write_episode_data(
            i,
            eval_dict={
                "episode_reward": stats.episode_reward,
                "NOOP": stats.get_action_usage(NOOP),
                "up": stats.get_action_usage(UP),
                "right": stats.get_action_usage(RIGHT),
                "left": stats.get_action_usage(LEFT),
                "down": stats.get_action_usage(DOWN)
            })

        if i % MODEL_TEST_INTERVAL == 0 or i >= (num_episodes - 1):

            stats_test = run_episode(
                env,
                agent,
                max_timesteps=max_timesteps,
                deterministic=True,
                softmax=False,
                do_training=False,
                rendering=False,
                skip_frames=2,
                history_length=history_length,
                diff_history=diff_history)

            tensorboard_test.write_episode_data(
                i,
                eval_dict={
                    "episode_reward": stats.episode_reward,
                    "NOOP": stats.get_action_usage(NOOP),
                    "up": stats.get_action_usage(UP),
                    "right": stats.get_action_usage(RIGHT),
                    "left": stats.get_action_usage(LEFT),
                    "down": stats.get_action_usage(DOWN)
                })

        # store model every 100 episodes and in the end.
        if i % MODEL_SAVE_INTERVAL == 0 or i >= (num_episodes - 1):
            agent.saver.save(agent.sess,
                             os.path.join(ckpt_path, "dqn_agent.ckpt"))

        # write an episode counter, so that we can resume training later
        with open(os.path.join(model_path, "epstrained.json"), "w") as fh:
            json.dump(i, fh)

    tensorboard.close_session()
    tensorboard_test.close_session()
    


def make_pacman_agent(name, model_path, lr, discount_factor,
                       batch_size, epsilon, epsilon_decay, boltzmann, tau,
                       double_q, buffer_capacity, history_length, diff_history, big=False,
                       save_hypers=False):

    hypers = locals()

    num_actions = 5

    # save hypers into folder -- used for reconstructing model at test time
    if save_hypers:
        with open(os.path.join(model_path, "hypers.json"), "w") as fh:
            json.dump(hypers, fh)

    # using -1 for unused parameters. fix later.
    Q_current = CNN(
        num_actions=num_actions,
        lr=lr,
        history_length=history_length,
        diff_history=diff_history,
        big=big)

    Q_target = CNNTargetNetwork(
        num_actions=num_actions,
        lr=lr,
        tau=tau,
        history_length=history_length,
        diff_history=diff_history,
        big=big)

    # 2. init DQNAgent (see dqn/dqn_agent.py)
    agent = DQNAgent(
        name,
        Q_current,
        Q_target,
        num_actions,
        discount_factor,
        batch_size,
        epsilon,
        epsilon_decay,
        boltzmann,
        double_q,
        buffer_capacity)

    return agent


if __name__ == "__main__":

    env = gym.make('MsPacman-v0')

    # prepare a model folder
    base_path = os.path.join('.', 'pacman')
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    train_online('1_basic', env)

    #train_online('9_diffpenalty', env, history_length=1, diff_history=True, apply_lane_penalty=True, epsilon_decay=1e-3, try_resume=True)

    #train_online('10_dpbig', env, history_length=1, diff_history=True, apply_lane_penalty=True, epsilon_decay=1e-3, big=True, try_resume=True)

    #train_online('12_dpnoskip', env, history_length=1, diff_history=True, apply_lane_penalty=True, epsilon_decay=1e-3, skip_frames=0, num_episodes=1000, try_resume=True)


    env.close()
