# agents/ddpg_agent.py
"""
DDPGAgent for CarRacing-v3 (continuous control).

Uses:
- DDPGActor / DDPGCritic (models/ddpg.py)
- ReplayBuffer (utils/replay_buffer.py)

Public API matches other agents enough to plug into train.py:
    - select_action(state, eval_mode=False)
    - store_transition(...)
    - update()
    - hard_update_target()
    - attributes: batch_size, replay, total_steps, eps
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.replay_buffer import ReplayBuffer
from models.ddpg import DDPGActor, DDPGCritic


class DDPGAgent:
    def __init__(
        self,
        replay_size=150000,
        batch_size=64,
        gamma=0.99,
        lr=1e-4,
        eps_start=1.0,       # unused, present for compatibility
        eps_end=0.05,        # unused
        eps_decay_steps=500000,  # unused
        in_channels=4,
        img_h=84,
        img_w=84,
        device=None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gamma = gamma
        self.batch_size = batch_size
        self.replay = ReplayBuffer(replay_size)

        self.action_dim = 3
        self.max_action = 1.0
        self.is_continuous = True

        # Actor & Critic + targets
        self.actor = DDPGActor(in_channels=in_channels, img_h=img_h, img_w=img_w,
                               action_dim=self.action_dim).to(self.device)
        self.actor_target = DDPGActor(in_channels=in_channels, img_h=img_h, img_w=img_w,
                                      action_dim=self.action_dim).to(self.device)
        self.critic = DDPGCritic(in_channels=in_channels, img_h=img_h, img_w=img_w,
                                 action_dim=self.action_dim).to(self.device)
        self.critic_target = DDPGCritic(in_channels=in_channels, img_h=img_h, img_w=img_w,
                                        action_dim=self.action_dim).to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # Exploration noise (Gaussian, decaying)
        self.noise_std_start = 0.3
        self.noise_std_end = 0.05
        self.noise_decay_steps = 500_000

        # Soft update factor
        self.tau = 0.005

        # For compatibility with train.py logging
        self.total_steps = 0
        self.eps = 0.0  # DDPG is not epsilon-greedy, but train.py prints this

    # ----------------------------------------------------------------------
    # Utilities
    # ----------------------------------------------------------------------
    def _current_noise_std(self):
        frac = min(1.0, self.total_steps / self.noise_decay_steps)
        return self.noise_std_start + frac * (self.noise_std_end - self.noise_std_start)

    def _soft_update(self, online, target):
        for p, tp in zip(online.parameters(), target.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    # ----------------------------------------------------------------------
    # API used by train.py
    # ----------------------------------------------------------------------
    def select_action(self, state, eval_mode=False):
        """
        Args:
            state: np.ndarray (C, H, W)

        Returns:
            continuous action np.ndarray (3,) in env space:
                steer ∈ [-1, 1]
                gas   ∈ [0, 1]
                brake ∈ [0, 1]
        """
        self.actor.eval()
        with torch.no_grad():
            s_t = torch.tensor(state[None, :], dtype=torch.float32, device=self.device)
            action = self.actor(s_t).cpu().numpy()[0]  # in [-1, 1]
        self.actor.train()

        if not eval_mode:
            std = self._current_noise_std()
            noise = np.random.normal(0.0, std, size=action.shape)
            action = action + noise
            action = np.clip(action, -1.0, 1.0)

        # Map to CarRacing action space:
        #   steer: [-1,1] -> [-1,1]
        #   gas:   [-1,1] -> [0,1]
        #   brake: [-1,1] -> [0,1]
        steer = float(action[0])
        gas = float((action[1] + 1.0) / 2.0)
        brake = float((action[2] + 1.0) / 2.0)

        gas = np.clip(gas, 0.0, 1.0)
        brake = np.clip(brake, 0.0, 1.0)

        return np.array([steer, gas, brake], dtype=np.float32)

    def store_transition(self, state, action, reward, next_state, done):
        # action is continuous np.array([3,])
        self.replay.push(state, action, reward, next_state, done)

    def hard_update_target(self):
        """Not really needed for DDPG, but used by train.py; we'll just hard copy."""
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    # ----------------------------------------------------------------------
    # DDPG update
    # ----------------------------------------------------------------------
    def update(self):
        if len(self.replay) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.float32, device=self.device)  # (B, 3)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones_t = torch.tensor(dones.astype(np.uint8), dtype=torch.float32, device=self.device).unsqueeze(1)

        # ---------------- Critic update ----------------
        with torch.no_grad():
            next_actions = self.actor_target(next_states_t)            # [-1,1]
            # map to same range as critic expects (we keep critic in [-1,1] range)
            target_q = self.critic_target(next_states_t, next_actions)
            target_y = rewards_t + (1.0 - dones_t) * self.gamma * target_q

        current_q = self.critic(states_t, actions_t)
        critic_loss = F.mse_loss(current_q, target_y)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_opt.step()

        # ---------------- Actor update ----------------
        pred_actions = self.actor(states_t)
        actor_loss = -self.critic(states_t, pred_actions).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_opt.step()

        # ---------------- Target soft-update ----------------
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        # Return critic loss for logging if desired
        return float(critic_loss.item())
