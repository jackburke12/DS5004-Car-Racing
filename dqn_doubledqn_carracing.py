"""
Double DQN for Gymnasium CarRacing-v3
Discrete action set (5 actions):
0: do nothing
1: steer right
2: steer left
3: gas
4: brake

PyTorch implementation with Double DQN, replay buffer, frame preprocessing & stacking.

Usage: python double_dqn_carracing.py
"""

import random
import numpy as np
import pandas as pd
import cv2
import time
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import os

# -------------------------
# Hyperparameters
# -------------------------
ENV_ID = "CarRacing-v3"   # target environment version
SEED = 42

NUM_EPISODES = 1500
MAX_STEPS_PER_EPISODE = 1000
REPLAY_BUFFER_SIZE = 150000
BATCH_SIZE = 64
GAMMA = 0.99
LEARNING_RATE = 1e-4
TARGET_UPDATE_EVERY = 500   # hard update every 500 environment steps
WARMUP_STEPS = 5000
TRAIN_EVERY = 1
UPDATES_PER_STEP = 1
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 500000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_H = 84
IMG_W = 84
NUM_STACK = 4

# -------------------------
# Define 5 discrete actions
# -------------------------
ACTIONS = [
    np.array([0.0, 0.0, 0.0], dtype=np.float32),   # 0: do nothing
    np.array([+1.0, 0.0, 0.0], dtype=np.float32),  # 1: steer right
    np.array([-1.0, 0.0, 0.0], dtype=np.float32),  # 2: steer left
    np.array([0.0, 1.0, 0.0], dtype=np.float32),   # 3: gas
    np.array([0.0, 0.0, 0.8], dtype=np.float32),   # 4: brake
]
NUM_ACTIONS = len(ACTIONS)

# -------------------------
# Frame preprocessing
# -------------------------
def preprocess_frame(frame):
    """
    Convert RGB frame (H, W, 3) -> grayscale resized to (IMG_H x IMG_W), 
    and normalized to [0,1] (of type float32).
    """
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)   # convert to grayscale
    img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)   # resize
    img = img.astype(np.float32) / 255.0   # normalize to [0,1]
    return img   # shape (IMG_H, IMG_W)

# -------------------------
# Replay Buffer
# -------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):   # random minibatch sampling from replay buffer
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))   # states shape: (B, C, H, W)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

# -------------------------
# Q-network (CNN-FFN)
# -------------------------
class QNetwork(nn.Module):
    def __init__(self, num_actions, in_channels=NUM_STACK):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        # compute conv output size dynamically
        with torch.no_grad():
            test_in = torch.zeros(1, in_channels, IMG_H, IMG_W)
            conv_out_size = self.conv(test_in).view(1, -1).shape[1]   # flatten the convolution output into a one-row tensor (vector) and get its length

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):   # input x: preprocessed image stack by FrameStack
        x = self.conv(x)   # output x shape from the convolution: (B, C, H, W)
        x = x.view(x.size(0), -1)   # convert x into shape: (B, C*H*W)
        return self.fc(x)   # get the logits for the action values

# ----------------------------
# Agent with DQN or Double DQN
# ----------------------------
class Agent:
    def __init__(self, double_dqn=False):
        self.online_net = QNetwork(NUM_ACTIONS).to(DEVICE)   # online policy network
        self.target_net = QNetwork(NUM_ACTIONS).to(DEVICE)   # target policy network
        self.target_net.load_state_dict(self.online_net.state_dict())   # function to update the target network parameters from online network
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=LEARNING_RATE)
        self.replay = ReplayBuffer(REPLAY_BUFFER_SIZE)
        self.total_steps = 0
        self.eps = EPS_START   # epsilon greedy starting rate
        self.double_dqn = double_dqn

    def select_action(self, state, eval_mode=False):
        """
        state: numpy array shape (C,H,W)
        returns action index (int)
        """
        if eval_mode:
            eps = 0.001
        else:
            eps = max(EPS_END, EPS_START - (EPS_START - EPS_END) * (self.total_steps / EPS_DECAY_STEPS))  # decayed epsilon greedy schedular
        self.eps = eps

        if random.random() < eps:
            return random.randrange(NUM_ACTIONS)   # random action selection
        
        state_t = torch.tensor(state[None, :], dtype=torch.float32, device=DEVICE)  # reshape the state as input tensor to online network (1,C,H,W)
        with torch.no_grad():
            qvals = self.online_net(state_t)  # action values estimation by the online network, shape: (1, NUM_ACTIONS)
        return int(qvals.argmax(dim=1).item())   # action section based on action values

    def store(self, *args):
        self.replay.push(*args)   # update replay buffer

    def update(self):
        if len(self.replay) < BATCH_SIZE:
            return None

        states, actions, rewards, next_states, dones = self.replay.sample(BATCH_SIZE)   # random minibatch sampling

        # convert to tensors
        states_t = torch.tensor(states, dtype=torch.float32, device=DEVICE)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=DEVICE)
        actions_t = torch.tensor(actions, dtype=torch.long, device=DEVICE)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
        dones_t = torch.tensor(dones.astype(np.uint8), dtype=torch.float32, device=DEVICE)

        # current Q(s,a) calculated by the online network
        q_values = self.online_net(states_t)  # (B, A)
        q_s_a = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)  # (B,)

        # DQN or Double DQN target Q(s,a) calculation by the target network
        with torch.no_grad():
            if self.double_dqn:
                next_q_online = self.online_net(next_states_t)           # (B,A)   online network deciding next action: argmax_aQ(S_t+1; a; θ_t);
                next_actions = next_q_online.argmax(dim=1, keepdim=True) # (B,1)
                next_q_target = self.target_net(next_states_t)           # (B,A)   target network deciding next state action values
                next_q_value = next_q_target.gather(1, next_actions).squeeze(1)  # (B,)   get the next state action value Q(s',a) based on next action
                target = rewards_t + (1.0 - dones_t) * GAMMA * next_q_value   # Bellman equation to get target Q(s,a)
            else:
                next_q_value = self.target_net(next_states_t).max(1)[0]   # directly use target nextwork and Bellman equation to get target Q(s,a)
                target = rewards_t + (1.0 - dones_t) * GAMMA * next_q_value

        # Loss
        loss = F.smooth_l1_loss(q_s_a, target)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)

        # Online network parameter update
        self.optimizer.step()

        return loss.item()

    def hard_update_target(self):   # Sync the target network parameters from online network
        self.target_net.load_state_dict(self.online_net.state_dict())

# -------------------------
# Frame stack helper
# -------------------------
class FrameStack:
    def __init__(self, k):
        self.k = k
        self.deque = deque(maxlen=k)

    def reset(self, frame):
        processed = preprocess_frame(frame)
        for _ in range(self.k):
            self.deque.append(processed)
        return np.stack(self.deque, axis=0).astype(np.float32)

    def append(self, frame):
        processed = preprocess_frame(frame)
        self.deque.append(processed)
        return np.stack(self.deque, axis=0).astype(np.float32)

# -------------------------
# Training loop
# -------------------------
def train(agent, env):
    agent = agent
    env = env
    name = "double_dqn" if agent.double_dqn else "dqn"
    fs = FrameStack(NUM_STACK)
    global_step = 0
    episode_rewards = []
    avg50_rewards = []

    print(f"\nStarting training the {name} agent...")
    checkpoint_dir = f"./{name}_carracing_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    for ep in range(1, NUM_EPISODES + 1):
        obs, info = env.reset()   # get the initial 96x96 RGB image observation from environment
        state = fs.reset(obs)   # preprocess the image
        ep_reward = 0.0

        for step in range(MAX_STEPS_PER_EPISODE):
            action_idx = agent.select_action(state)
            cont_action = ACTIONS[action_idx]

            next_obs, reward, terminated, truncated, info = env.step(cont_action)
            done = terminated or truncated

            next_state = fs.append(next_obs)   # preprocess the 96x96 RGB image from environment
            agent.store(state, action_idx, reward, next_state, done)

            state = next_state
            ep_reward += reward
            global_step += 1
            agent.total_steps = global_step

            # training step(s)
            if global_step > WARMUP_STEPS and global_step % TRAIN_EVERY == 0:
                for _ in range(UPDATES_PER_STEP):
                    agent.update()

            # target network hard update
            if global_step % TARGET_UPDATE_EVERY == 0:
                agent.hard_update_target()

            if done:
                break

        episode_rewards.append(ep_reward)
        avg50 = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 1 else ep_reward
        avg50_rewards.append(avg50)
        print(f"Episode: {ep:4d} | EpisodeReward: {ep_reward:7.1f} | Avg50EpReward: {avg50:7.2f} | Epsilon: {agent.eps:.3f}")

        if ep % 50 == 0:
            torch.save(agent.online_net.state_dict(), f"./{checkpoint_dir}/{name}_carracing_v3_ep{ep}.pth")
            print(f"{name} checkpoint saved at episode {ep}")

    training_results = {"EpisodeReward": episode_rewards, "Avg50EpReward": avg50_rewards}
    results_df = pd.DataFrame(training_results)
    results_df.to_csv(f'{name}_training_results.csv', index=False)
    print(f"\nTraining the {name} agent finished! Training_results saved to '{name}_training_results.csv'")

    env.close()
    return agent

# -------------------------
# Evaluation / play
# -------------------------
def evaluate(agent, env, episodes=3, render=True):
    agent = agent
    env = env
    name = "double_dqn" if agent.double_dqn else "dqn"
    fs = FrameStack(NUM_STACK)

    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        state = fs.reset(obs)
        total_reward = 0.0
        done = False
        steps = 0
        while not done and steps < MAX_STEPS_PER_EPISODE:
            action_idx = agent.select_action(state, eval_mode=True)
            cont_action = ACTIONS[action_idx]
            obs, reward, terminated, truncated, info = env.step(cont_action)
            done = terminated or truncated
            state = fs.append(obs)
            total_reward += reward
            steps += 1
            if render:
                time.sleep(0.01)
        print(f"Evaluation of trained {name} agent - Episode {ep} reward: {total_reward:.1f}")
    env.close()

# -------------------------
# Run training + evaluation
# -------------------------
if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # create env without rendering for training
    env = gym.make(ENV_ID, render_mode=None)
    env.reset(seed=SEED)
    
    agent = Agent(double_dqn=False)

    train(agent, env)

    print("\nRunning evaluation with rendering.")
    env = gym.make(ENV_ID, render_mode="human")
    evaluate(agent, env, episodes=3)


    



