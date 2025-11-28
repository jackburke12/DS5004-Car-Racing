# agents/td3_agent.py
"""
TD3Agent for CarRacing-v3 (continuous control).

- Uses the same pixel-based CNN actor/critic as DDPG (models.ddpg)
- Two critics (Q1, Q2) + target networks
- Policy smoothing and delayed policy updates (TD3 paper)

Public API matches the other agents so it plugs into train.py:
    - select_action(state, eval_mode=False) -> continuous action [steer, gas, brake]
    - store_transition(...)
    - update()
    - hard_update_target() (for compatibility with train.py)
    - attributes: batch_size, replay, total_steps, eps, is_continuous
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.replay_buffer import ReplayBuffer
from models.ddpg import DDPGActor, DDPGCritic


class TD3Agent:
    def __init__(
        self,
        replay_size=150000,
        batch_size=64,
        gamma=0.99,
        lr=1e-4,
        eps_start=1.0,          # unused, for signature compatibility
        eps_end=0.05,            # unused
        eps_decay_steps=500000,  # unused
        in_channels=4,
        img_h=84,
        img_w=84,
        device=None,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        tau=0.005,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gamma = gamma
        self.batch_size = batch_size
        self.replay = ReplayBuffer(replay_size)

        self.action_dim = 3
        self.max_action = 1.0

        # TD3 components
        self.actor = DDPGActor(in_channels=in_channels, img_h=img_h, img_w=img_w,
                               action_dim=self.action_dim).to(self.device)
        self.actor_target = DDPGActor(in_channels=in_channels, img_h=img_h, img_w=img_w,
                                      action_dim=self.action_dim).to(self.device)

        self.critic1 = DDPGCritic(in_channels=in_channels, img_h=img_h, img_w=img_w,
                                  action_dim=self.action_dim).to(self.device)
        self.critic2 = DDPGCritic(in_channels=in_channels, img_h=img_h, img_w=img_w,
                                  action_dim=self.action_dim).to(self.device)

        self.critic1_target = DDPGCritic(in_channels=in_channels, img_h=img_h, img_w=img_w,
                                         action_dim=self.action_dim).to(self.device)
        self.critic2_target = DDPGCritic(in_channels=in_channels, img_h=img_h, img_w=img_w,
                                         action_dim=self.action_dim).to(self.device)

        # Initialize targets
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_opt = torch.optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_opt = torch.optim.Adam(self.critic2.parameters(), lr=lr)

        # Exploration noise for behavior policy
        self.noise_std_start = 0.3
        self.noise_std_end = 0.05
        self.noise_decay_steps = 500_000

        # TD3 hyperparameters
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.tau = tau

        # Bookkeeping for train.py
        self.total_steps = 0
        self.eps = 0.0              # not epsilon-greedy
        self.is_continuous = True   # IMPORTANT: used by train/evaluate

        # Internal update counter (if you prefer this over total_steps)
        self._update_step = 0

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _current_noise_std(self):
        frac = min(1.0, self.total_steps / self.noise_decay_steps)
        return self.noise_std_start + frac * (self.noise_std_end - self.noise_std_start)

    def _soft_update(self, online, target):
        for p, tp in zip(online.parameters(), target.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def select_action(self, state, eval_mode=False):
        """
        Args:
            state: np.ndarray (C, H, W)

        Returns:
            np.ndarray [steer, gas, brake] in CarRacing-v3 action space
        """
        self.actor.eval()
        with torch.no_grad():
            s_t = torch.tensor(state[None, :], dtype=torch.float32, device=self.device)
            action = self.actor(s_t).cpu().numpy()[0]  # in [-1, 1]
        self.actor.train()

        if not eval_mode:
            std = self._current_noise_std()
            noise = np.random.normal(0.0, std, size=action.shape)
            action = np.clip(action + noise, -1.0, 1.0)

        # Map to CarRacing: steer [-1,1], gas/brake [0,1]
        steer = float(action[0])
        gas = float((action[1] + 1.0) / 2.0)
        brake = float((action[2] + 1.0) / 2.0)

        gas = np.clip(gas, 0.0, 1.0)
        brake = np.clip(brake, 0.0, 1.0)

        return np.array([steer, gas, brake], dtype=np.float32)

    def store_transition(self, state, action, reward, next_state, done):
        # store continuous action
        self.replay.push(state, action, reward, next_state, done)

    def hard_update_target(self):
        """For compatibility with train.py; TD3 also uses soft updates each step."""
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

    # ------------------------------------------------------------------ #
    # TD3 Update
    # ------------------------------------------------------------------ #
    def update(self):
        if len(self.replay) < self.batch_size:
            return None

        self._update_step += 1

        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.float32, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones_t = torch.tensor(dones.astype(np.uint8), dtype=torch.float32, device=self.device).unsqueeze(1)

        # ---------------- Target policy smoothing ----------------
        with torch.no_grad():
            next_action = self.actor_target(next_states_t)

            # Add clipped noise to target action
            noise = torch.normal(
                mean=0.0,
                std=self.policy_noise,
                size=next_action.shape,
                device=self.device,
            )
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            next_action = torch.clamp(next_action + noise, -1.0, 1.0)

            # Compute target Q-values
            target_q1 = self.critic1_target(next_states_t, next_action)
            target_q2 = self.critic2_target(next_states_t, next_action)
            target_q = torch.min(target_q1, target_q2)

            target_y = rewards_t + (1.0 - dones_t) * self.gamma * target_q

        # ---------------- Critic updates ----------------
        current_q1 = self.critic1(states_t, actions_t)
        current_q2 = self.critic2(states_t, actions_t)

        critic1_loss = F.mse_loss(current_q1, target_y)
        critic2_loss = F.mse_loss(current_q2, target_y)
        critic_loss = critic1_loss + critic2_loss

        self.critic1_opt.zero_grad()
        self.critic2_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic1.parameters(), 10.0)
        nn.utils.clip_grad_norm_(self.critic2.parameters(), 10.0)
        self.critic1_opt.step()
        self.critic2_opt.step()

        # ---------------- Delayed policy updates ----------------
        if self._update_step % self.policy_freq == 0:
            # Actor aims to maximize Q1
            pred_actions = self.actor(states_t)
            actor_loss = -self.critic1(states_t, pred_actions).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
            self.actor_opt.step()

            # Soft update targets
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic1, self.critic1_target)
            self._soft_update(self.critic2, self.critic2_target)

        return float(critic_loss.item())
