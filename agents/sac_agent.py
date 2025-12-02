# agents/sac_agent.py
"""
Soft Actor-Critic (SAC) agent for CarRacing-v3 from pixels.

Architecture:
    - CNN encoder for actor and for each critic, using CNNFeatureExtractor
    - Tanh-squashed Gaussian policy (stochastic)
    - Twin Q-critics with target networks
    - Automatic entropy temperature tuning (alpha)

Conventions:
    - Actor outputs raw actions in [-1, 1]^3
    - Env expects [steer ∈ [-1,1], gas ∈ [0,1], brake ∈ [0,1]]
      We map gas/brake from raw [-1,1] → [0,1] in select_action()
    - Replay buffer stores *raw* actions in [-1,1]^3 (like TD3)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from utils.replay_buffer import ReplayBuffer
from models.common_networks import CNNFeatureExtractor


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


# -----------------------------------------------------------------------------
# Networks
# -----------------------------------------------------------------------------
class SACActor(nn.Module):
    def __init__(self, in_channels=4, img_h=84, img_w=84, action_dim=3):
        super().__init__()

        self.cnn = CNNFeatureExtractor(in_channels, img_h, img_w)
        feat_dim = self.cnn.output_dim

        self.fc_body = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        self.mean_head = nn.Linear(256, action_dim)
        self.log_std_head = nn.Linear(256, action_dim)

    def forward(self, x):
        """
        Returns:
            mean, log_std: tensors of shape (B, action_dim)
        """
        feat = self.cnn(x)
        h = self.fc_body(feat)

        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, x, deterministic: bool = False):
        """
        Sample tanh-squashed Gaussian action and its log-prob.

        Args:
            x: (B,C,H,W)
            deterministic: if True, use mean instead of sampling

        Returns:
            action: (B, action_dim) in [-1,1]
            log_prob: (B,1)
        """
        mean, log_std = self.forward(x)
        std = log_std.exp()

        if deterministic:
            z = mean
        else:
            normal = Normal(mean, std)
            z = normal.rsample()  # reparameterization

        action = torch.tanh(z)

        # Log prob with tanh correction
        # log pi(a) = log pi(z) - sum log(1 - tanh(z)^2)
        # Add small epsilon for numerical stability
        log_prob = None
        if not deterministic:
            normal = Normal(mean, std)
            log_prob = normal.log_prob(z)  # (B, action_dim)
            log_prob = log_prob - torch.log(1.0 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=1, keepdim=True)  # (B,1)

        return action, log_prob


class SACCritic(nn.Module):
    def __init__(self, in_channels=4, img_h=84, img_w=84, action_dim=3):
        super().__init__()

        self.cnn = CNNFeatureExtractor(in_channels, img_h, img_w)
        feat_dim = self.cnn.output_dim

        self.q_net = nn.Sequential(
            nn.Linear(feat_dim + action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x, action):
        """
        x: (B,C,H,W)
        action: (B, action_dim)
        """
        feat = self.cnn(x)
        x_cat = torch.cat([feat, action], dim=1)
        return self.q_net(x_cat)


# -----------------------------------------------------------------------------
# SAC Agent
# -----------------------------------------------------------------------------
class SACAgent:
    def __init__(
        self,
        replay_size=150000,
        batch_size=64,
        gamma=0.99,
        lr=3e-4,
        eps_start=1.0,          # unused, but kept for train.py interface
        eps_end=0.05,           # unused
        eps_decay_steps=500000, # unused
        in_channels=4,
        img_h=84,
        img_w=84,
        device=None,
        tau=0.005,
        target_entropy=None,
        actor_lr = None,
        critic_lr = None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gamma = gamma
        self.batch_size = batch_size
        self.replay = ReplayBuffer(replay_size)

        self.action_dim = 3
        self.max_action = 1.0

        self.is_continuous = True
        self.uses_soft_update = True  # like TD3, we only use soft updates
        self.eps = 0.0  # for logging in train.py

        # ----------------------------
        # Networks
        # ----------------------------
        self.actor = SACActor(in_channels, img_h, img_w, self.action_dim).to(self.device)
        self.critic1 = SACCritic(in_channels, img_h, img_w, self.action_dim).to(self.device)
        self.critic2 = SACCritic(in_channels, img_h, img_w, self.action_dim).to(self.device)

        self.critic1_target = SACCritic(in_channels, img_h, img_w, self.action_dim).to(self.device)
        self.critic2_target = SACCritic(in_channels, img_h, img_w, self.action_dim).to(self.device)

        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        if actor_lr is None:
            actor_lr = lr
        if critic_lr is None:
            critic_lr = lr

        # Optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_opt = torch.optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_opt = torch.optim.Adam(self.critic2.parameters(), lr=lr)

        # ----------------------------
        # Entropy temperature (alpha)
        # ----------------------------
        if target_entropy is None:
            # Standard heuristic: -dim(A)
            target_entropy = -float(self.action_dim)
        self.target_entropy = target_entropy

        self.log_alpha = torch.zeros(1, device=self.device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)

        self.tau = tau
        self.total_steps = 0

    # -------------------------------------------------------------------------
    def _soft_update(self, online, target):
        for p, tp in zip(online.parameters(), target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    # -------------------------------------------------------------------------
    # ACTION SELECTION (raw → env)
    # -------------------------------------------------------------------------
    def select_action(self, state, eval_mode=False):
        """
        state: np.ndarray (C,H,W) normalized
        returns: env action [steer, gas, brake]
        """
        self.actor.eval()
        with torch.no_grad():
            s = torch.tensor(state[None, :], dtype=torch.float32, device=self.device)
            # deterministic during eval; stochastic during training
            action_raw, _ = self.actor.sample(s, deterministic=eval_mode)
            action_raw = action_raw.cpu().numpy()[0]
        self.actor.train()

        # Map raw [-1,1] → env [steer, gas, brake]
        steer = float(action_raw[0])
        gas = float((action_raw[1] + 1.0) / 2.0)
        brake = float((action_raw[2] + 1.0) / 2.0)

        gas = np.clip(gas, 0.0, 1.0)
        brake = np.clip(brake, 0.0, 1.0)

        return np.array([steer, gas, brake], dtype=np.float32)

    # -------------------------------------------------------------------------
    # STORE TRANSITION (env → raw)
    # -------------------------------------------------------------------------
    def store_transition(self, state, action, reward, next_state, done):
        """
        action is env action: [steer, gas, brake]
        convert gas/brake back to raw [-1,1] range before storing
        """
        steer, gas, brake = float(action[0]), float(action[1]), float(action[2])
        gas_raw = gas * 2.0 - 1.0
        brake_raw = brake * 2.0 - 1.0
        action_raw = np.array([steer, gas_raw, brake_raw], dtype=np.float32)

        self.replay.push(state, action_raw, reward, next_state, done)

    # -------------------------------------------------------------------------
    # SAC UPDATE
    # -------------------------------------------------------------------------
    def update(self):
        if len(self.replay) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.float32, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones_t = torch.tensor(dones.astype(np.uint8), dtype=torch.float32, device=self.device).unsqueeze(1)

        # ------------------------------
        # Critic targets
        # ------------------------------
        with torch.no_grad():
            next_action_raw, next_log_prob = self.actor.sample(next_states_t, deterministic=False)

            q1_next = self.critic1_target(next_states_t, next_action_raw)
            q2_next = self.critic2_target(next_states_t, next_action_raw)
            q_next = torch.min(q1_next, q2_next)

            alpha = self.log_alpha.exp()
            target_q = q_next - alpha * next_log_prob  # (B,1)
            target_y = rewards_t + (1.0 - dones_t) * self.gamma * target_q

        # ------------------------------
        # Critic losses
        # ------------------------------
        q1 = self.critic1(states_t, actions_t)
        q2 = self.critic2(states_t, actions_t)

        critic1_loss = F.mse_loss(q1, target_y)
        critic2_loss = F.mse_loss(q2, target_y)
        critic_loss = critic1_loss + critic2_loss

        self.critic1_opt.zero_grad()
        self.critic2_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic1.parameters(), 10.0)
        nn.utils.clip_grad_norm_(self.critic2.parameters(), 10.0)
        self.critic1_opt.step()
        self.critic2_opt.step()

        # ------------------------------
        # Actor loss
        # ------------------------------
        new_action_raw, log_prob = self.actor.sample(states_t, deterministic=False)
        q1_pi = self.critic1(states_t, new_action_raw)
        q2_pi = self.critic2(states_t, new_action_raw)
        q_pi = torch.min(q1_pi, q2_pi)

        alpha = self.log_alpha.exp()
        actor_loss = (alpha * log_prob - q_pi).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_opt.step()

        # ------------------------------
        # Temperature (alpha) update
        # ------------------------------
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # ------------------------------
        # Soft target updates
        # ------------------------------
        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)

    # -------------------------------------------------------------------------
    # Checkpointing (compatible with TD3 style)
    # -------------------------------------------------------------------------
    def save(self, path):
        ckpt = {
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "critic1_target": self.critic1_target.state_dict(),
            "critic2_target": self.critic2_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic1_opt": self.critic1_opt.state_dict(),
            "critic2_opt": self.critic2_opt.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_opt": self.alpha_opt.state_dict(),
            "total_steps": self.total_steps,
        }
        torch.save(ckpt, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic1.load_state_dict(ckpt["critic1"])
        self.critic2.load_state_dict(ckpt["critic2"])
        self.critic1_target.load_state_dict(ckpt["critic1_target"])
        self.critic2_target.load_state_dict(ckpt["critic2_target"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic1_opt.load_state_dict(ckpt["critic1_opt"])
        self.critic2_opt.load_state_dict(ckpt["critic2_opt"])

        if "log_alpha" in ckpt:
            self.log_alpha = ckpt["log_alpha"].to(self.device)
            self.log_alpha.requires_grad_(True)
        if "alpha_opt" in ckpt:
            self.alpha_opt.load_state_dict(ckpt["alpha_opt"])

        self.total_steps = ckpt.get("total_steps", 0)
