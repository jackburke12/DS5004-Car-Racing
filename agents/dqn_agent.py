# agents/dqn_agent.py
"""
DQNAgent: Implements the standard DQN learning algorithm.
"""

import torch
import torch.nn.functional as F
import numpy as np

from agents.base_agent import BaseAgent
from models.dqn import DQN


class DQNAgent(BaseAgent):
    def __init__(
        self,
        replay_size=150000,
        batch_size=64,
        gamma=0.99,
        lr=1e-4,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_steps=500000,
        in_channels=4,
        img_h=84,
        img_w=84,
        device=None,
    ):
        super().__init__(
            model_class=DQN,
            replay_size=replay_size,
            batch_size=batch_size,
            gamma=gamma,
            lr=lr,
            eps_start=eps_start,
            eps_end=eps_end,
            eps_decay_steps=eps_decay_steps,
            device=device,
            in_channels=in_channels,
            img_h=img_h,
            img_w=img_w,
        )

    # -------------------------------------------------------------------------
    # Standard DQN update rule
    # -------------------------------------------------------------------------
    def update(self):
        """Perform one DQN update step. Returns loss or None."""
        if len(self.replay) < self.batch_size:
            return None

        # Sample minibatch
        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)

        # Convert everything to tensors
        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones.astype(np.uint8), dtype=torch.float32, device=self.device)

        # Q(s,a) using the online network
        q_values = self.online_net(states_t)                         # (B, A)
        q_s_a = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)  # (B,)

        # DQN target: max over Q_target(next_state)
        with torch.no_grad():
            next_q_values = self.target_net(next_states_t)          # (B, A)
            next_q_max = next_q_values.max(dim=1)[0]                # (B,)
            target = rewards_t + (1.0 - dones_t) * self.gamma * next_q_max

        # Compute Huber loss
        loss = F.smooth_l1_loss(q_s_a, target)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        return loss.item()
