# agents/ddpg_agent.py
"""
DDPGAgent for CarRacing-v3 (continuous control).

Uses:
- DDPGActor / DDPGCritic (models/ddpg.py)
- ReplayBuffer (utils/replay_buffer.py)

Public API matches other agents enough to plug into train.py:
    - select_action(state, eval_mode=False) -> env action [steer, gas, brake]
    - store_transition(state, action, reward, next_state, done)
    - update()
    - hard_update_target()
    - save(path), load(path)
    - attributes: batch_size, replay, total_steps, eps, is_continuous
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
        eps_start=1.0,          # unused, present for compatibility
        eps_end=0.05,           # unused
        eps_decay_steps=500000, # unused
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
        self.is_continuous = True  # used by train/evaluate to branch action handling

        # Actor & Critic + targets
        self.actor = DDPGActor(
            in_channels=in_channels,
            img_h=img_h,
            img_w=img_w,
            action_dim=self.action_dim,
        ).to(self.device)

        self.actor_target = DDPGActor(
            in_channels=in_channels,
            img_h=img_h,
            img_w=img_w,
            action_dim=self.action_dim,
        ).to(self.device)

        self.critic = DDPGCritic(
            in_channels=in_channels,
            img_h=img_h,
            img_w=img_w,
            action_dim=self.action_dim,
        ).to(self.device)

        self.critic_target = DDPGCritic(
            in_channels=in_channels,
            img_h=img_h,
            img_w=img_w,
            action_dim=self.action_dim,
        ).to(self.device)

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
            state: np.ndarray (C, H, W) — already preprocessed/stacked frames

        Returns:
            env_action: np.ndarray(3,) in CarRacing action space:
                steer ∈ [-1, 1]
                gas   ∈ [0, 1]
                brake ∈ [0, 1]

        NOTE:
            - Internally, the actor operates in a "raw" action space a_raw ∈ [-1,1]^3.
            - We add exploration noise in raw space.
            - We map to env space only for env.step().
            - For training, we store *raw* actions (reconstructed in store_transition).
        """
        self.actor.eval()
        with torch.no_grad():
            s_t = torch.tensor(state[None, :], dtype=torch.float32, device=self.device)
            action_raw = self.actor(s_t).cpu().numpy()[0]  # in [-1, 1]
        self.actor.train()

        if not eval_mode:
            std = self._current_noise_std()
            noise = np.random.normal(0.0, std, size=action_raw.shape)
            action_raw = action_raw + noise
            action_raw = np.clip(action_raw, -1.0, 1.0)
            self.total_steps += 1  # count only during training

        # Map raw action -> CarRacing env space:
        #   steer: [-1,1] -> [-1,1]
        #   gas:   [-1,1] -> [0,1]
        #   brake: [-1,1] -> [0,1]
        steer = float(action_raw[0])
        gas = float((action_raw[1] + 1.0) / 2.0)
        brake = float((action_raw[2] + 1.0) / 2.0)

        gas = np.clip(gas, 0.0, 1.0)
        brake = np.clip(brake, 0.0, 1.0)

        env_action = np.array([steer, gas, brake], dtype=np.float32)
        return env_action

    def store_transition(self, state, action, reward, next_state, done):
        """
        Store a transition.

        Args:
            state:      np.ndarray (C,H,W)
            action:     np.ndarray(3,) in *env* space [steer ∈ [-1,1], gas/brake ∈ [0,1]]
            reward:     float
            next_state: np.ndarray (C,H,W)
            done:       bool

        Internally, we convert env-action back to the actor's raw space in [-1,1]^3
        so that the critic sees a consistent action distribution.
        """
        # Convert env action back to raw actor space:
        # steer_raw = steer (already [-1,1])
        # gas_raw   = gas * 2 - 1   (maps [0,1] -> [-1,1])
        # brake_raw = brake * 2 - 1
        steer_env, gas_env, brake_env = float(action[0]), float(action[1]), float(action[2])
        gas_raw = gas_env * 2.0 - 1.0
        brake_raw = brake_env * 2.0 - 1.0
        action_raw = np.array([steer_env, gas_raw, brake_raw], dtype=np.float32)

        self.replay.push(state, action_raw, reward, next_state, done)

    def hard_update_target(self):
        """For compatibility with train.py; we also do soft updates each step."""
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
        actions_t = torch.tensor(actions, dtype=torch.float32, device=self.device)  # (B, 3) in [-1,1]
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones_t = torch.tensor(dones.astype(np.uint8), dtype=torch.float32, device=self.device).unsqueeze(1)

        # ---------------- Critic update ----------------
        with torch.no_grad():
            next_actions_raw = self.actor_target(next_states_t)  # [-1,1]^3
            target_q = self.critic_target(next_states_t, next_actions_raw)
            target_y = rewards_t + (1.0 - dones_t) * self.gamma * target_q

        current_q = self.critic(states_t, actions_t)
        critic_loss = F.mse_loss(current_q, target_y)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_opt.step()

        # ---------------- Actor update ----------------
        pred_actions_raw = self.actor(states_t)
        actor_loss = -self.critic(states_t, pred_actions_raw).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_opt.step()

        # ---------------- Target soft-update ----------------
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        return float(critic_loss.item())

    # ----------------------------------------------------------------------
    # Save / Load
    # ----------------------------------------------------------------------
    def save(self, path):
        ckpt = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "total_steps": self.total_steps,
        }
        torch.save(ckpt, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic_opt.load_state_dict(ckpt["critic_opt"])

        self.total_steps = ckpt.get("total_steps", 0)
