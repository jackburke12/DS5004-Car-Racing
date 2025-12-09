# agents/base_agent.py
"""
BaseAgent: Shared utilities for all DQN-style agents.

This class handles:
    - network creation (subclasses provide model class)
    - epsilon-greedy action selection
    - replay buffer integration
    - saving/loading checkpoints
    - global step counting
    - storing transitions

Algorithm-specific logic (target updates, loss computation)
is implemented inside subclass update() methods.

Used by:
    - DQNAgent
    - DoubleDQNAgent
    - DuelingDQNAgent
"""

import os
import torch
import torch.nn as nn
import random
import numpy as np

from utils.replay_buffer import ReplayBuffer
from utils.action_mapping import NUM_ACTIONS
from utils.action_mapping import ACTIONS


class BaseAgent:
    """
    Base class for all agents. Subclasses must override:
        - build_model()
        - update()
    """

    def __init__(
        self,
        model_class,
        replay_size=100000,
        batch_size=64,
        gamma=0.99,
        lr=1e-4,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_steps=500000,
        device=None,
        in_channels=4,
        img_h=84,
        img_w=84,
    ):
        # Device handling
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hyperparameters
        self.gamma = gamma
        self.batch_size = batch_size
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.lr = lr
        self.is_continuous = False

        # Exploration tracking
        self.total_steps = 0
        self.eps = eps_start

        # Replay buffer
        self.replay = ReplayBuffer(replay_size)

        # Build online & target networks
        self.online_net = model_class(
            num_actions=NUM_ACTIONS,
            in_channels=in_channels,
            img_h=img_h,
            img_w=img_w
        ).to(self.device)

        self.target_net = model_class(
            num_actions=NUM_ACTIONS,
            in_channels=in_channels,
            img_h=img_h,
            img_w=img_w
        ).to(self.device)

        # Sync initial parameters
        self.target_net.load_state_dict(self.online_net.state_dict())

        # Optimizer
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.lr)

    # Action selection
    def select_action(self, state, eval_mode=False):
        """
        Args:
            state: np.array (C,H,W)
            eval_mode: if True, uses tiny epsilon (near-greedy)

        Returns:
            action_idx: int in [0, NUM_ACTIONS)
        """
        # Compute current epsilon
        if eval_mode:
            eps = 0.001
        else:
            eps = max(
                self.eps_end,
                self.eps_start - (self.total_steps / self.eps_decay_steps) * (self.eps_start - self.eps_end)
            )

        self.eps = eps

        # Epsilon-greedy policy
        if (not eval_mode) and random.random() < eps:
            return random.randrange(NUM_ACTIONS)

        state_t = torch.tensor(state[None, :], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            qvals = self.online_net(state_t)
        return int(qvals.argmax(dim=1).item())

    #Experience Replay
    def store_transition(self, state, action, reward, next_state, done):
        self.replay.push(state, action, reward, next_state, done)

    # Target network update
    def hard_update_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    # Checkpointing
    def save(self, path):
        """Save online network parameters."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.online_net.state_dict(), path)

    def load(self, path):
        """Load online network parameters."""
        self.online_net.load_state_dict(torch.load(path, map_location=self.device))
        self.hard_update_target()

    # Subclasses must override update()
    def update(self):
        """
        Abstract method.
        Must return loss (float) or None.
        """
        raise NotImplementedError("update() must be implemented in subclasses.")
