import tensorflow as tf
import numpy as np
from dqn.replay_buffer import ReplayBuffer

class DQNAgent:

    def __init__(self, Q, Q_target, num_actions, game="cartpole" ,explore_type="epsilon_greedy", epsilon_decay=1, epsilon_min=0.05, tau=1, method="CQL", discount_factor=0.99, batch_size=64, epsilon=0.05):
        """
         Q-Learning agent for off-policy TD control using Function Approximation.
         Finds the optimal greedy policy while following an epsilon-greedy policy.

         Args:
            Q: Action-Value function estimator (Neural Network)
            Q_target: Slowly updated target network to calculate the targets.
            num_actions: Number of actions of the environment.
            discount_factor: gamma, discount factor of future rewards.
            batch_size: Number of samples per batch.
            epsilon: Chance to sample a random action. Float betwen 0 and 1.
        """
        self.Q = Q      
        self.Q_target = Q_target
        # now support cartpole or carracing two games
        self.game = game
        # self.state_dim = Q.
        self.epsilon = epsilon
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        # now support CQL(classical Q) or DQL(Double Q) 
        self.method = method
        self.explore_type = explore_type
        # for epsilon annealing
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        # for boltzmann exploration
        self.tau = tau
        # define replay buffer
        self.replay_buffer = ReplayBuffer()
        
        # start tensorflow session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
        self.saver = tf.train.Saver()


    def train(self, state, action, next_state, reward, terminal):
        """
        This method stores a transition to the replay buffer and updates the Q networks.
        """

        # TODO:
        # 1. add current transition to replay buffer
        # 2. sample next batch and perform batch update: 
        #       2.1 compute td targets: 
        #              td_target =  reward + discount * argmax_a Q_target(next_state_batch, a)
        #       2.2 update the Q network
        #              self.Q.update(...)
        #       2.3 call soft update for target network
        #              self.Q_target.update(...)
        self.replay_buffer.add_transition(state, action, next_state, reward, terminal)
        batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones = self.replay_buffer.next_batch(self.batch_size)
        td_target = batch_rewards
        if self.method == "CQL":
            td_target[np.logical_not(batch_dones)] += self.discount_factor * np.max(self.Q_target.predict(self.sess, batch_next_states), 1)[np.logical_not(batch_dones)]
            self.Q.update(self.sess, batch_states, batch_actions, td_target)
            self.Q_target.update(self.sess)
        elif self.method == "DQL":
            best_action = np.argmax(self.Q.predict(self.sess, batch_next_states)[np.logical_not(batch_dones)], 1)
            td_target[np.logical_not(batch_dones)] += self.discount_factor * self.Q_target.predict(self.sess, batch_next_states)[np.logical_not(batch_dones), best_action]
            self.Q.update(self.sess, batch_states, batch_actions, td_target)
            self.Q_target.update(self.sess)




    def act(self, state, deterministic):
        """
        This method creates an epsilon-greedy policy based on the Q-function approximator and epsilon (probability to select a random action)    
        Args:
            state: current state input
            deterministic:  if True, the agent should execute the argmax action (False in training, True in evaluation)
        Returns:
            action id
        """
        if deterministic:
            action_id = np.argmax(self.Q.predict(self.sess, [state]))
        else:
            if self.explore_type == "epsilon_greedy":
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
                r = np.random.uniform()
                if r > self.epsilon:
                    # TODO: take greedy action (argmax)
                    action_id = np.argmax(self.Q.predict(self.sess, [state]))
                else:
                    # TODO: sample random action
                    # Hint for the exploration in CarRacing: sampling the action from a uniform distribution will probably not work. 
                    # You can sample the agents actions with different probabilities (need to sum up to 1) so that the agent will prefer to accelerate or going straight.
                    # To see how the agent explores, turn the rendering inthe training on and look what the agent is doing.
                    if self.game == "cartpole" or self.game == "mountaincar":
                        action_id = np.random.randint(self.num_actions)
                    elif self.game == "carracing":
                        # action_probability = np.array([1, 2, 2, 10, 1, 1, 1])
                        action_probability = np.array([2, 5, 5, 10, 1])
                        action_probability = action_probability / np.sum(action_probability)
                        action_id = np.random.choice(self.num_actions, p=action_probability)
                    else:
                        print("Invalid game")
            elif self.explore_type == "boltzmann":
                action_value = self.Q.predict(self.sess, [state])[0]
                prob = self.softmax(action_value/self.tau)
                action_id = np.random.choice(self.num_actions, p=prob)
            else:
                print("Invalid Exploration Type")
        return action_id

    def softmax(self, input):
        """
        Safe Softmax function to avoid overflow
        Args:
            input: input vector
        Returns:
            prob: softmax of input
        """
        input_max = np.max(input)
        e = np.exp(input-input_max)
        prob = e / np.sum(e)
        return prob
        
    def load(self, file_name):
        self.saver.restore(self.sess, file_name)
