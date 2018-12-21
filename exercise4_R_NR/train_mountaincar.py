import numpy as np
import gym
import itertools as it
from dqn.dqn_agent import DQNAgent
from tensorboard_evaluation import *
from dqn.networks import NeuralNetwork, TargetNetwork
from utils import EpisodeStats


def run_episode(env, agent, deterministic, do_training=True, rendering=False, max_timesteps=1000):
    """
    This methods runs one episode for a gym environment. 
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """
    
    stats = EpisodeStats()        # save statistics like episode reward or action usage
    state = env.reset()
    step = 0
    while True:
        
        action_id = agent.act(state=state, deterministic=deterministic)
        next_state, reward, terminal, info = env.step(action_id)

        if do_training:  
            agent.train(state, action_id, next_state, reward, terminal)

        stats.step(reward, action_id)

        state = next_state
        
        if rendering:
            env.render()

        if terminal or step > max_timesteps: 
            break

        step += 1

    return stats

def train_online(env, agent, num_episodes, model_dir="./models_mountaincar", tensorboard_dir="./tensorboard"):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
 
    print("... train agent")

    tensorboard = Evaluation(os.path.join(tensorboard_dir, "train"), ["episode_reward", "valid_episode_reward", "a_0", "a_1", "a_2"])

    # training
    for i in range(num_episodes):
        print("episode: ",i)
        stats = run_episode(env, agent, deterministic=False, do_training=True)
        tensorboard.write_episode_data(i, eval_dict={"episode_reward" : stats.episode_reward, 
                                                     "a_0" : stats.get_action_usage(0),
                                                     "a_1" : stats.get_action_usage(1),
                                                     "a_2" : stats.get_action_usage(2)})

        # TODO: evaluate your agent once in a while for some episodes using run_episode(env, agent, deterministic=True, do_training=False) to 
        # check its performance with greedy actions only. You can also use tensorboard to plot the mean episode reward.
        # store model every 100 episodes and in the end.
        if i % 100 == 0 or i >= (num_episodes - 1):
            valid_reward = 0
            for j in range(5):
                valid_stats = run_episode(env, agent, deterministic=True, do_training=False, rendering=False)
                valid_reward += valid_stats.episode_reward
            tensorboard.write_valid_episode_data(i, eval_dict={"valid_episode_reward" : valid_reward/5})
            agent.saver.save(agent.sess, os.path.join(model_dir, "dqn_agent.ckpt"))
   
    tensorboard.close_session()


if __name__ == "__main__":

    # You find information about cartpole in 
    # https://github.com/openai/gym/wiki/CartPole-v0
    # Hint: CartPole is considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.

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
    train_online(env, agent, 1600)
