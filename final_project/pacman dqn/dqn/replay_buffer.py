from collections import namedtuple
import numpy as np
import os
import gzip
import pickle


class ReplayBuffer:

    # DONE: implement a capacity for the replay buffer (FIFO, capacity: 1e5 - 1e6)

    # Replay buffer for experience replay. Stores transitions.
    def __init__(self, capacity=1e5):
        self._data = namedtuple(
            "ReplayBuffer",
            ["states", "actions", "next_states", "rewards", "dones"])
        # this seems weird - define _data as a tuple, and then redefine it as a new instance of that tuple??
        # why are we using a named tuple anyway?
        self._data = self._data(
            states=[], actions=[], next_states=[], rewards=[], dones=[])

        self.capacity = capacity

    def add_transition(self, state, action, next_state, reward, done):
        """
        This method adds a transition to the replay buffer.
        """
        # check if we're over capacity
        while len(self._data.states) >= self.capacity:
            for d in self._data:
                del d[0]  #FIFO

        self._data.states.append(state)
        self._data.actions.append(action)
        self._data.next_states.append(next_state)
        self._data.rewards.append(reward)
        self._data.dones.append(done)

    def next_batch(self, batch_size):
        """
        This method samples a batch of transitions.
        """
        batch_indices = np.random.choice(len(self._data.states), batch_size)
        batch_states = np.array([self._data.states[i] for i in batch_indices])
        batch_actions = np.array(
            [self._data.actions[i] for i in batch_indices])
        batch_next_states = np.array(
            [self._data.next_states[i] for i in batch_indices])
        batch_rewards = np.array(
            [self._data.rewards[i] for i in batch_indices])
        batch_dones = np.array([self._data.dones[i] for i in batch_indices])
        return batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones
