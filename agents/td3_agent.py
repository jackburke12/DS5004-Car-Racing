# agents/td3_agent.py
"""
Stable TD3 Agent for CarRacing-v3 (pixel observations)

Improvements included:
    - Separate actor/critic CNN encoders (from DDPGActor/DDPGCritic)
    - Actor uses stop-gradient features (in DDPGActor)
    - TD3 soft target updates only
    - Critic target Q-values detached safely
    - Actor loss uses a detached critic output (prevents instability)
    - Actor/critic mode switching for extra LayerNorm safety
    - Correct action scaling between raw & env action space
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
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_steps=500000,
        in_channels=4,
        img_h=84,
        img_w=84,
        device=None,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=4,   # better for image-based TD3
        tau=0.005,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gamma = gamma
        self.batch_size = batch_size
        self.replay = ReplayBuffer(replay_size)

        self.action_dim = 3
        self.max_action = 1.0
        self.is_continuous = True

        # TD3 uses only soft updates (train.py checks this flag)
        self.uses_soft_update = True

        # ----------------------------
        # Networks
        # ----------------------------
        self.actor = DDPGActor(in_channels, img_h, img_w, self.action_dim).to(self.device)
        self.actor_target = DDPGActor(in_channels, img_h, img_w, self.action_dim).to(self.device)

        self.critic1 = DDPGCritic(in_channels, img_h, img_w, self.action_dim).to(self.device)
        self.critic2 = DDPGCritic(in_channels, img_h, img_w, self.action_dim).to(self.device)

        self.critic1_target = DDPGCritic(in_channels, img_h, img_w, self.action_dim).to(self.device)
        self.critic2_target = DDPGCritic(in_channels, img_h, img_w, self.action_dim).to(self.device)

        # Copy initial weights
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_opt = torch.optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_opt = torch.optim.Adam(self.critic2.parameters(), lr=lr)

        # TD3 hyperparameters
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.tau = tau

        # Noise schedule for exploration
        self.noise_std_start = 0.6
        self.noise_std_end = 0.1
        self.noise_decay_steps = 500_000

        self.total_steps = 0
        self._update_step = 0
        self.eps = 0.0

    # --------------------------------------------------------------
    def _current_noise_std(self):
        frac = min(1.0, self.total_steps / self.noise_decay_steps)
        return self.noise_std_start + frac * (self.noise_std_end - self.noise_std_start)

    # --------------------------------------------------------------
    def _soft_update(self, online, target):
        for p, tp in zip(online.parameters(), target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    # --------------------------------------------------------------
    # ACTION SELECTION (raw → env)
    # --------------------------------------------------------------
    def select_action(self, state, eval_mode=False):
        """
        state: np.ndarray (C,H,W) already normalized
        returns: env action [steer, gas, brake]
        """
        self.actor.eval()
        with torch.no_grad():
            s = torch.tensor(state[None, :], dtype=torch.float32, device=self.device)
            action_raw = self.actor(s).cpu().numpy()[0]
        self.actor.train()

        # Add exploration noise
        if not eval_mode:
            noise = np.random.normal(0.0, self._current_noise_std(), size=action_raw.shape)
            action_raw = np.clip(action_raw + noise, -1.0, 1.0)
            self.total_steps += 1

        # Convert raw action → env action
        steer = float(action_raw[0])
        gas = float((action_raw[1] + 1.0) / 2.0)
        brake = float((action_raw[2] + 1.0) / 2.0)

        gas = np.clip(gas, 0.0, 1.0)
        brake = np.clip(brake, 0.0, 1.0)

        return np.array([steer, gas, brake], dtype=np.float32)

    # --------------------------------------------------------------
    # STORE TRANSITION (env → raw)
    # --------------------------------------------------------------
    def store_transition(self, state, action, reward, next_state, done):
        """
        action is env action: [steer, gas, brake]
        convert gas/brake back to raw [-1,1] range
        """
        steer, gas, brake = float(action[0]), float(action[1]), float(action[2])
        gas_raw = gas * 2.0 - 1.0
        brake_raw = brake * 2.0 - 1.0
        action_raw = np.array([steer, gas_raw, brake_raw], dtype=np.float32)

        self.replay.push(state, action_raw, reward, next_state, done)

    # --------------------------------------------------------------
    # TD3 UPDATE
    # --------------------------------------------------------------
    def update(self):
        if len(self.replay) < self.batch_size:
            return

        self._update_step += 1

        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.float32, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones_t = torch.tensor(dones.astype(np.uint8), dtype=torch.float32, device=self.device).unsqueeze(1)

        # --------------------------------------------------------------
        # TARGET ACTION (with smoothing)
        # --------------------------------------------------------------
        with torch.no_grad():
            next_action_raw = self.actor_target(next_states_t)

            noise = torch.normal(0.0, self.policy_noise, next_action_raw.shape, device=self.device)
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)

            next_action_raw = torch.clamp(next_action_raw + noise, -1.0, 1.0)

            target_q1 = self.critic1_target(next_states_t, next_action_raw).detach()
            target_q2 = self.critic2_target(next_states_t, next_action_raw).detach()
            target_q = torch.min(target_q1, target_q2)

            target_y = rewards_t + (1.0 - dones_t) * self.gamma * target_q

        # --------------------------------------------------------------
        # CRITIC UPDATES
        # --------------------------------------------------------------
        self.actor.eval()  # just for LayerNorm safety

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

        self.actor.train()

        # --------------------------------------------------------------
        # DELAYED ACTOR UPDATE
        # --------------------------------------------------------------
        if self._update_step % self.policy_freq == 0:

            pred_action_raw = self.actor(states_t)

            # VERY important: detach Q to avoid interfering with critic CNN learning
            q_for_actor = self.critic1(states_t, pred_action_raw).detach()
            actor_loss = -q_for_actor.mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
            self.actor_opt.step()

            # Soft target updates
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic1, self.critic1_target)
            self._soft_update(self.critic2, self.critic2_target)

    # --------------------------------------------------------------
    def save(self, path):
        ckpt = {
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic1_target": self.critic1_target.state_dict(),
            "critic2_target": self.critic2_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic1_opt": self.critic1_opt.state_dict(),
            "critic2_opt": self.critic2_opt.state_dict(),
            "total_steps": self.total_steps,
            "_update_step": self._update_step,
        }
        torch.save(ckpt, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic1.load_state_dict(ckpt["critic1"])
        self.critic2.load_state_dict(ckpt["critic2"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic1_target.load_state_dict(ckpt["critic1_target"])
        self.critic2_target.load_state_dict(ckpt["critic2_target"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic1_opt.load_state_dict(ckpt["critic1_opt"])
        self.critic2_opt.load_state_dict(ckpt["critic2_opt"])
        self.total_steps = ckpt.get("total_steps", 0)
        self._update_step = ckpt.get("_update_step", 0)
