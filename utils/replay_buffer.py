# utils/replay_buffer.py
"""
Simple replay buffer for off-policy RL algorithms.

Stores transitions in a deque and supports random minibatch sampling.
Used by DQN, Double DQN, Dueling DQN, and any future agents.
"""

import random
import numpy as np
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity):
        """
        Args:
            capacity: maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Store a transition in the buffer.

        Args:
            state:      np.ndarray (C,H,W)
            action:     int
            reward:     float
            next_state: np.ndarray (C,H,W)
            done:       bool
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Randomly sample a batch of transitions.

        Returns:
            states, actions, rewards, next_states, dones
            Each item is a NumPy array of shape (batch_size, ...)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
